from copy import deepcopy
import logging
from operator import concat
import os
import time

from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
import numpy as np
import wandb
import numpy as np

from classifier import Classifier
from image_list import ImageList, mixup_data
from moco.builder import hwc_MoCo
from utils import (
    adjust_learning_rate,
    concat_all_gather,
    get_distances,
    is_master,
    save_checkpoint,
    use_wandb,
    AverageMeter,
    CustomDistributedDataParallel,
    ProgressMeter,
)
import pandas as pd
from sklearn.mixture import GaussianMixture  ## numpy version
from target import (
    eval_and_label_dataset, 
    get_augmentation_versions, 
    refine_predictions, 
    get_target_optimizer, 
    noise_detect_cls, 
    get_center_proto,
    prototype_cluster
)

@torch.no_grad()
def update_labels(banks, idxs, features, logits, args):
    # 1) avoid inconsistency among DDP processes, and
    # 2) have better estimate with more data points
    if args.distributed:
        idxs = concat_all_gather(idxs)
        features = concat_all_gather(features)
        logits = concat_all_gather(logits)
    
    probs = F.softmax(logits, dim=1)
    
    start = banks["ptr"]
    end = start + len(features)
    banks["features"] = torch.cat([banks["features"], features], dim=0)
    banks["probs"] = torch.cat([banks["probs"], probs], dim=0)
    banks["logit"] = torch.cat([banks["logit"], logits], dim=0)
    banks["index"] = torch.cat([banks["index"], idxs], dim=0)
    banks["ptr"] = end % len(banks["features"])

def train_target_domain(args):
    logging.info(
        f"Start target training on {args.data.src_domain}-{args.data.tgt_domain}..."
    )

    # if not specified, use the full length of dataset.
    if args.learn.queue_size == -1:
        if args.data.dataset.lower() == 'pacs': 
            label_file = os.path.join(
                args.data.image_root, f"{args.data.tgt_domain}_test_kfold.txt"
            )
        else: 
            label_file = os.path.join(
                args.data.image_root, f"{args.data.tgt_domain}_list.txt"
            )
        dummy_dataset = ImageList(args.data.image_root, label_file)
        data_length = len(dummy_dataset)
        args.learn.queue_size = data_length
        del dummy_dataset

    checkpoint_path = os.path.join(
        args.model_tta.src_log_dir,
        f"best_{args.data.src_domain}_{args.seed}.pth.tar",
    )
    train_target = (args.data.src_domain != args.data.tgt_domain)
    src_model = Classifier(args.model_src, train_target, checkpoint_path)
    momentum_model = Classifier(args.model_src, train_target, checkpoint_path)
    
    # val_transform = get_augmentation("test")
    val_transform = get_augmentation_versions(args, False)
    if args.data.dataset.lower() == 'pacs': 
        label_file = os.path.join(args.data.image_root, f"{args.data.tgt_domain}_test_kfold.txt")
    else: 
        label_file = os.path.join(args.data.image_root, f"{args.data.tgt_domain}_list.txt")
    val_dataset = ImageList(
        image_root=args.data.image_root,
        label_file=label_file,
        transform=val_transform,
    )
    model = hwc_MoCo(
        src_model,
        momentum_model,
        K=args.model_tta.queue_size,
        m=args.model_tta.m,
        T_moco=args.model_tta.T_moco,
        dataset_legth=len(val_dataset),
        args=args
    ).cuda()
    
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = CustomDistributedDataParallel(model, device_ids=[args.gpu])
    logging.info(f"1 - Created target model")

    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=256, sampler=val_sampler, num_workers=2
    )
    if args.learn.sep_gmm:
        gm = [GaussianMixture(n_components=2, random_state=0) for i in range(model.src_model.num_classes+1)]
    else: 
        gm = GaussianMixture(n_components=2, random_state=0,max_iter=50)
    
    banks = {
        "features": torch.tensor([], device='cuda'),
        "probs": torch.tensor([], device='cuda'),
        "logit": torch.tensor([], device='cuda'),
        "index": torch.tensor([], device='cuda'),
        "ptr": 0,
        'gm':gm
    }


    logging.info("2 - Computed initial pseudo labels")
    
    args.num_clusters = model.src_model.num_classes
    # Training data
    train_transform = get_augmentation_versions(args)
    train_dataset = ImageList(
        image_root=args.data.image_root,
        label_file=label_file,  # uses pseudo labels
        transform=train_transform,
        # pseudo_item_list=pseudo_item_list,
    )
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.data.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.data.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=False,
    )

    args.learn.full_progress = args.learn.epochs * len(train_loader)
    logging.info("3 - Created train/val loader")

    # define loss function (criterion) and optimizer
    optimizer = get_target_optimizer(model, args)
    logging.info("4 - Created optimizer")

    logging.info("Start training...")
    for epoch in range(args.learn.start_epoch, args.learn.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            
        train_epoch_sfda(train_loader, model, banks,
                    optimizer, epoch, args)
        
    _, banks = eval_and_label_dataset(val_loader, model, banks, epoch, banks['gm'], args)

    if is_master(args):
        filename = f"checkpoint_{epoch:04d}_{args.data.src_domain}-{args.data.tgt_domain}-{args.sub_memo}_{args.seed}.pth.tar"
        save_path = os.path.join(args.log_dir, filename)
        save_checkpoint(model, optimizer, epoch, save_path=save_path)
        logging.info(f"Saved checkpoint {save_path}")
        

def train_epoch_sfda(train_loader, model, banks,

                     optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    loss_meter = AverageMeter("Loss", ":.4f")
    top1_ins = AverageMeter("SSL-Acc@1", ":6.2f")
    top1_psd = AverageMeter("CLS-Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, loss_meter, top1_ins, top1_psd],
        prefix=f"Epoch: [{epoch}]",
    )
    first_X_samples = 0 
    # make sure to switch to train mode
    model.train()

    end = time.time()
    zero_tensor = torch.tensor([0.0]).to("cuda")
    for i, data in enumerate(train_loader):
        # unpack and move data
        images, _, idxs = data
        idxs = idxs.to("cuda")
        images_w, images_q, images_k = (
            images[0].to("cuda"),
            images[1].to("cuda"),
            images[2].to("cuda"),
        )
        
        # per-step scheduler
        step = i + epoch * len(train_loader)
        adjust_learning_rate(optimizer, step, args)
        
        # weak aug model output
        feats_w, logits_w = model(images_w, banks, idxs, cls_only=True)
        with torch.no_grad():
            probs_w = F.softmax(logits_w, dim=1)
            if use_proto := first_X_samples >= 1024:
                args.learn.refine_method = "nearest_neighbors"
            else:
                args.learn.refine_method = None
                first_X_samples += len(feats_w)
                
            pseudo_labels_w, probs_w, _ = refine_predictions(
                model, feats_w, probs_w, banks, args=args, return_index=args.learn.return_index
            )
        
        # similarity btw prototype(mean) and feats_w
        if use_proto: 
            # use confidence center
            cur_probs   = torch.cat([banks['probs'], probs_w], dim=0)
            cur_feats   = torch.cat([banks['features'], feats_w.detach()], dim=0) ## current 
            cur_pseudo  = torch.cat([banks['logit'].argmax(dim=1), pseudo_labels_w], dim=0) ## current 
            
            confidence, _ = noise_detect_cls(cur_probs, cur_pseudo, cur_feats, banks, args)
            origin_idx = torch.arange(banks['probs'].size(0), banks['probs'].size(0)+probs_w.size(0))
            ignore_idx = confidence[origin_idx] < 0.5 # select noise label

            args.learn.use_ce_weight = True
            args.learn.use_proto_loss_v2 = True
            model.confidence = confidence[torch.arange(banks['probs'].size(0))]
            aug_prototypes = get_center_proto(banks, use_confidence=False)
            prototypes = get_center_proto(banks, use_confidence=False)
        else: 
            args.learn.do_noise_detect = False
            aug_prototypes = None
            prototypes = None 
            ignore_idx = None
            args.learn.use_proto_loss_v2 = False

        # strong aug model output 
        feats_q, logits_q, logits_ins, feats_k, logits_k, logits_neg_near, loss_proto = model(images_q, banks, idxs, images_k, pseudo_labels_w, epoch, 
                                                                                            prototypes_q=prototypes, prototypes_k=aug_prototypes, 
                                                                                            ignore_idx=ignore_idx, args=args)
        
        # mixup
        alpha = 1.0
        inputs_w, targets_w_a, targets_w_b, lam_w, mix_w_idx = mixup_data(images_w, pseudo_labels_w,
                                                        alpha, use_cuda=True)
        inputs_w, targets_w_a, targets_w_b = map(Variable, (inputs_w,
                                                    targets_w_a, targets_w_b))
        
        targets_w_mix = lam_w * torch.eye(args.num_clusters).cuda()[targets_w_a] + (1-lam_w) * torch.eye(args.num_clusters).cuda()[targets_w_b]
        
        feats_w_mix, target_w_mix_logit = model(inputs_w, banks, idxs, cls_only=True)
        
        if args.learn.use_mixup_ws:
        
            inputs_q, targets_q_a, targets_q_b, lam_q, mix_q_idx = mixup_data(images_q, pseudo_labels_w,
                                                            alpha, use_cuda=True)
            inputs_q, targets_q_a, targets_q_b = map(Variable, (inputs_q,
                                                        targets_q_a, targets_q_b))
            
            targets_q_mix = lam_q * torch.eye(args.num_clusters).cuda()[targets_q_a] + (1-lam_q) * torch.eye(args.num_clusters).cuda()[targets_q_b]
            
            feats_q_mix, target_q_mix_logit = model(inputs_q, banks, idxs, cls_only=True)
        
        # similarity btw prototype and mixup input
        if args.learn.use_mixup_weight and args.learn.do_noise_detect:
            weight_a = confidence[origin_idx]
            weight_b = weight_a[mix_w_idx]
            weight_mix_w = lam_w * weight_a + (1-lam_w) * weight_b
            loss_mix = KLLoss(target_w_mix_logit, targets_w_mix, epsilon=1e-8, weight_mix=weight_mix_w)
            if args.learn.use_mixup_ws:
                weight_mix_q = lam_q * weight_a + (1-lam_q) * weight_b
                loss_mix += KLLoss(target_q_mix_logit, targets_q_mix, epsilon=1e-8, weight_mix=weight_mix_q)
            # loss_mix = KLLoss(target_w_mix_logit, targets_w_mix, epsilon=1e-8)
        else:
            loss_mix = mixup_criterion(target_w_mix_logit, targets_w_a, targets_w_b, lam=lam_w, weight_a=None, weight_b=None)
            
        # Calculate reliable degree of pseudo labels
        if args.learn.use_ce_weight and args.learn.do_noise_detect:
            CE_weight = confidence[origin_idx]
            CE_weight[ignore_idx] = 0 
        else:
            CE_weight = 1.
        
        # update key features and corresponding pseudo labels
        model.update_memory(feats_k, pseudo_labels_w, probs_w, idxs)

        if epoch > 0 and args.learn.use_confidence_instance_loss: 
            # sfda instance loss
            q = F.normalize(feats_q, dim=1)
            proto_sim = q @ F.normalize(prototypes, dim=1).T/0.25 ## (B x feature)x (Features class)= B, class 
            proto_sim = F.softmax(proto_sim, dim=1)
            # proto_label = torch.argmax(proto_sim, axis=1) 
            
            loss_ins, accuracy_ins = confi_instance_loss(
                logits_ins=logits_ins,
                pseudo_labels=pseudo_labels_w,
                mem_labels=model.mem_labels,
                logits_neg_near=logits_neg_near,
                confidence=model.confidence, 
                proto_sim=proto_sim, 
                contrast_type=args.learn.contrast_type,
            )
        else: 
            # sfda instance loss
            loss_ins, accuracy_ins = instance_loss(
                logits_ins=logits_ins,
                pseudo_labels=pseudo_labels_w,
                mem_labels=model.mem_labels,
                logits_neg_near=logits_neg_near,
                contrast_type=args.learn.contrast_type,
            )
            
        # instance accuracy shown for only one process to give a rough idea
        top1_ins.update(accuracy_ins.item(), len(logits_ins))

        # classification
        loss_cls, accuracy_psd = classification_loss(
            logits_w, logits_q, pseudo_labels_w, CE_weight, args
        )
        top1_psd.update(accuracy_psd.item(), len(logits_w))

        # diversification
        loss_div = (
            diversification_loss(logits_w, logits_q, args)
            if args.learn.eta > 0
            else zero_tensor
        )
        
       
        loss = (
            args.learn.alpha * loss_cls
            + args.learn.beta * loss_ins
            + args.learn.eta * loss_div
            + loss_mix
        )
        if loss_proto: loss += loss_proto

        loss_meter.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # use slow feature to update neighbor space
        with torch.no_grad():
            feats_w, logits_w = model.momentum_model(images_w, return_feats=True)
            feats_k, logits_k = model.momentum_model(images_k, return_feats=True)

        update_labels(banks, idxs, feats_w, logits_w, args)
        

        if use_wandb(args):
            wandb_dict = {
                "loss_cls": args.learn.alpha * loss_cls.item(),
                "loss_ins": args.learn.beta * loss_ins.item(),
                "loss_div": args.learn.eta * loss_div.item(),
                "loss_mix": loss_mix.item(),
                "acc_ins": accuracy_ins.item(),
                
                "lr": optimizer.param_groups[0]['lr']
            }
            
            if loss_proto: wandb_dict.update({"loss_proto": loss_proto.item()})

            wandb.log(wandb_dict, commit=(i != len(train_loader) - 1))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.learn.print_freq == 0:
            progress.display(i)
            

def prototype(banks, features):
    feature_bank = banks["features"]
    probs_bank = banks["probs"]
    
    prototypes = probs_bank.T.mm(feature_bank)
    
    similarity = F.normalize(features, dim=1) @ F.normalize(prototypes, dim=1).T
    similarity = F.softmax(similarity, dim=1)
    
    return prototypes, similarity
   
@torch.no_grad()
def calculate_acc(logits, labels):
    preds = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean() * 100
    return accuracy

def instance_loss(logits_ins, pseudo_labels, mem_labels, logits_neg_near, contrast_type):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()

    # in class_aware mode, do not contrast with same-class samples
    if contrast_type == "class_aware" and pseudo_labels is not None:
        mask = torch.ones_like(logits_ins, dtype=torch.bool)
        mask[:, 1:] = pseudo_labels.reshape(-1, 1) != mem_labels  # (B, K)
        logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())
    elif contrast_type == "nearest" and pseudo_labels is not None:
        mask = torch.ones_like(logits_ins, dtype=torch.bool)        
        _, idx_near = torch.topk(logits_neg_near, k=10, dim=-1, largest=True)
        mask[:, 1:] = torch.all(pseudo_labels.unsqueeze(1).repeat(1,10).unsqueeze(1) != mem_labels[idx_near].unsqueeze(0), dim=2) # (B, K)
        logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())

    loss = F.cross_entropy(logits_ins, labels_ins)

    accuracy = calculate_acc(logits_ins, labels_ins)

    return loss, accuracy

def confi_instance_loss(logits_ins, pseudo_labels, mem_labels, logits_neg_near, confidence, proto_sim, contrast_type):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()
    proto_label = torch.argmax(proto_sim, axis=1) 
    # in class_aware mode, do not contrast with same-class samples
    if contrast_type == "class_aware" and pseudo_labels is not None:
        mask = torch.ones_like(logits_ins, dtype=torch.bool)
        d_mask = torch.ones_like(logits_ins, dtype=torch.bool)
        
        mask[:, 1:] = pseudo_labels.reshape(-1, 1) != mem_labels  # (B, K) diff : 밀어낸다., same : 무시.
        d_mask[:, 1:] = proto_label.reshape(-1, 1) != mem_labels  # (B, K)
        
        # clean_confi = confidence < 0.8 # (1, K) # Clean : 무시., Noise 밀어낸다.

        clean_confi = confidence > 0.5 # (1, K) # Clean : 밀어낸다., Noise 일단 무시. 

        ## 일단 clean 안에서 본다면 어떻게 될까? 
        clean_confi = clean_confi.unsqueeze(0).repeat(mask.size(0),1)

        # d_mask[:, 1:] = d_mask[:, 1:] * clean_confi # (B, K)
        # mask[:,1:] = mask[:,1:] * d_mask[:, 1:] # (B, K) 
        mask[:,1:] = mask[:,1:] * clean_confi # (B, K) 
        logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())
    elif contrast_type == "nearest" and pseudo_labels is not None:
        mask = torch.ones_like(logits_ins, dtype=torch.bool)
        mask[:, 1:] = pseudo_labels.reshape(-1, 1) != mem_labels  # (B, K)
        
        _, idx_near = torch.topk(logits_neg_near, k=5, dim=-1, largest=True)
        mask[:, 1:] *= torch.all(pseudo_labels.unsqueeze(1).repeat(1,5).unsqueeze(1) != mem_labels[idx_near].unsqueeze(0), dim=2) # (B, K)
        logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())

    loss = F.cross_entropy(logits_ins, labels_ins)

    accuracy = calculate_acc(logits_ins, labels_ins)

    return loss, accuracy


def classification_loss(logits_w, logits_s, target_labels, CE_weight, args):
    if not args.learn.do_noise_detect: CE_weight = 1.
    
    if args.learn.ce_sup_type == "weak_weak":
        loss_cls = (CE_weight * cross_entropy_loss(logits_w, target_labels, args)).mean()
        
    elif args.learn.ce_sup_type == "weak_strong":
        # loss_cls = (CE_weight * cross_entropy_loss(logits_s, target_labels, args)).mean()
        loss_cls = (CE_weight * cross_entropy_loss(logits_s, target_labels, args))
        loss_cls = loss_cls[loss_cls!=0].mean() # ignore zero
        accuracy = calculate_acc(logits_s, target_labels)
        
    elif args.learn.ce_sup_type == "weak_strong_kl":
        loss_cls = KLLoss(logits_s, target_labels,  epsilon=1e-8, weight_mix=CE_weight)
        target_labels = torch.argmax(target_labels, dim=1)
        accuracy = calculate_acc(logits_s, target_labels)
    else:
        raise NotImplementedError(
            f"{args.learn.ce_sup_type} CE supervision type not implemented."
        )
    return loss_cls, accuracy


def div(logits, epsilon=1e-8):
    probs = F.softmax(logits, dim=1)
    probs_mean = probs.mean(dim=0)
    loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))

    return loss_div

def diversification_loss(logits_w, logits_s, args):
    if args.learn.ce_sup_type == "weak_weak":
        loss_div = div(logits_w)
    elif args.learn.ce_sup_type == "weak_strong":
        loss_div = div(logits_s)
    else:
        loss_div = div(logits_w) + div(logits_s)

    return loss_div


def cross_entropy_loss(logits, labels, args):
    if args.learn.ce_type == "standard":
        return F.cross_entropy(logits, labels)
    elif args.learn.ce_type == "reduction_none":
        return torch.nn.CrossEntropyLoss(reduction='none')(logits, labels)
    raise NotImplementedError(f"{args.learn.ce_type} CE loss is not implemented.")

def entropy_minimization(logits):
    if len(logits) == 0:
        return torch.tensor([0.0]).cuda()
    probs = F.softmax(logits, dim=1)
    ents = -(probs * probs.log()).sum(dim=1)

    loss = ents.mean()
    return loss

def mixup_criterion(pred, y_a, y_b, lam, weight_a, weight_b):
    if weight_a is not None and weight_b is not None:
        return (weight_a * lam * torch.nn.CrossEntropyLoss(reduction='none')(pred, y_a) 
                + weight_b * (1 - lam) * torch.nn.CrossEntropyLoss(reduction='none')(pred, y_b)).mean()
    else:
        return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)

def symmetric_cross_entropy(x, x_ema):
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)

def KLLoss(input, target, epsilon=1e-8, weight_mix=None):
    softmax = F.softmax(input, dim=1)
    kl_loss = (- target * torch.log(softmax + epsilon)).sum(dim=1)
    if weight_mix is not None:
        kl_loss *= torch.exp(weight_mix)
    return kl_loss.mean(dim=0)
