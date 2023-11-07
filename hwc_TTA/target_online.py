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
    get_augmentation_versions, 
    refine_predictions, 
    get_target_optimizer, 
    noise_detect_cls, 
    get_center_proto,
    prototype_cluster,
    instance_loss,
    calculate_acc,
    confi_instance_loss,
    classification_loss,
    diversification_loss,
    cross_entropy_loss,
    mixup_criterion,
    KLLoss,
    prototype,
    remove_wrap_arounds,
    per_class_accuracy
)
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
@torch.no_grad()
def eval_and_label_dataset(dataloader, model, banks, epoch, gm, args):
    wandb_dict = dict()
    # make sure to switch to eval mode
    model.eval()
    # projector = model.src_model.projector_q
    clustering = model.src_model.classifier_q
    
    # run inference
    logits, gt_labels, indices, cluster_labels = [], [], [], []
    features, project_feats = [], []
    alpha = []
    logging.info("Eval and labeling...")
    iterator = tqdm(dataloader) if is_master(args) else dataloader
    for data in iterator:
        images, labels, idxs = data
        imgs = images[0].to("cuda")
        inputs_w, targets_w_a, targets_w_b, lam, _ = mixup_data(imgs, labels.to('cuda'), 1, use_cuda=True)

        # wimgs = images[1].permute(1,0,2,3,4) if (use_loop := images[1].ndim > 4)else images[1]

        # (B, D) x (D, K) -> (B, K)
        feats, logits_cls = model(imgs, banks, idxs, cls_only=True)

        mix_feats, mix_logits_cls = model(inputs_w, banks, idxs, cls_only=True)
        
        features.append(feats)
        # project_feats.append(F.normalize(projector(feats), dim=1))
        cluster_labels.append(F.softmax(clustering(feats), dim=1))
        logits.append(logits_cls)

        # label and index    
        gt_labels.append(labels)
        indices.append(idxs)

    # origin
    features = torch.cat(features)
    # project_feats  = torch.cat(project_feats)
    cluster_labels = torch.cat(cluster_labels)
    logits = torch.cat(logits)
    
    # label and index
    gt_labels = torch.cat(gt_labels).to("cuda")
    indices = torch.cat(indices).to("cuda")
    
    if args.distributed:
        # gather results from all ranks
        features = concat_all_gather(features)
        # project_feats = concat_all_gather(project_feats)
        cluster_labels = concat_all_gather(cluster_labels)
        logits = concat_all_gather(logits)

        # label and index
        gt_labels = concat_all_gather(gt_labels)
        indices = concat_all_gather(indices)

        # remove extra wrap-arounds from DDP
        ranks = len(dataloader.dataset) % dist.get_world_size()
        features = remove_wrap_arounds(features, ranks)
        # project_feats = remove_wrap_arounds(project_feats, ranks)
        cluster_labels = remove_wrap_arounds(cluster_labels, ranks)
        logits = remove_wrap_arounds(logits, ranks)

        gt_labels = remove_wrap_arounds(gt_labels, ranks)
        indices = remove_wrap_arounds(indices, ranks)

    assert len(logits) == len(dataloader.dataset)
    pred_labels = logits.argmax(dim=1)
    accuracy = (pred_labels == gt_labels).float().mean() * 100
    logging.info(f"Accuracy of direct prediction: {accuracy:.2f}")
    wandb_dict["Test Acc"] = accuracy
    
    if args.data.dataset == "VISDA-C":
        acc_per_class = per_class_accuracy(
            y_true=gt_labels.cpu().numpy(),
            y_pred=pred_labels.cpu().numpy(),
        )
        wandb_dict["Test Avg"] = acc_per_class.mean()
        wandb_dict["Test Per-class"] = acc_per_class
        class_name = ['Aeroplane', 'Bicycle', 'Bus', 'Car', 'Horse', 'Knife', 'Motorcycle', 'Person', 'Plant', 'Skateboard', 'Train', 'Truck']
        class_dict = {idx:name[:3]for idx, name in enumerate(class_name)}

    probs = F.softmax(logits, dim=1)
    rand_idxs = torch.randperm(len(features)).cuda()
    banks = {
        "features": features[rand_idxs][: args.learn.queue_size],
        "probs": probs[rand_idxs][: args.learn.queue_size],
        "logit": logits[rand_idxs][: args.learn.queue_size],
        "ptr": 0,
        "norm_features": F.normalize(features[rand_idxs][: args.learn.queue_size]),
    }
    if args.learn.add_gt_in_bank: 
        banks.update({"gt": gt_labels[rand_idxs]})

    if args.learn.return_index: 
        banks.update({"index": indices[rand_idxs]})

    banks.update({'gm':gm})
    
    # refine predicted labels
    pred_labels, _, acc = refine_predictions(
            model, features, probs, banks, args=args, gt_labels=gt_labels, return_index=args.learn.return_index
    )

    wandb_dict["Test Post Acc"] = acc
    if args.data.dataset == "VISDA-C":
        acc_per_class = per_class_accuracy(
            y_true=gt_labels.cpu().numpy(),
            y_pred=pred_labels.cpu().numpy(),
        )
        wandb_dict["Test Post Avg"] = acc_per_class.mean()
        wandb_dict["Test Post Per-class"] = acc_per_class

    pseudo_item_list = []
    for pred_label, idx in zip(pred_labels, indices):
        img_path, gt, img_file = dataloader.dataset.item_list[idx]
        pseudo_item_list.append((img_path, int(gt), img_file))
    logging.info(f"Collected {len(pseudo_item_list)} pseudo labels.")

    if use_wandb(args):
        wandb.log(wandb_dict)
        
    return pseudo_item_list, banks

@torch.no_grad()
def update_labels(banks, idxs, features, logits, label, args):
    # 1) avoid inconsistency among DDP processes, and
    # 2) have better estimate with more data points
    if args.distributed:
        idxs = concat_all_gather(idxs)
        features = concat_all_gather(features)
        logits = concat_all_gather(logits)
        label = concat_all_gather(label.to("cuda"))
    
    probs = F.softmax(logits, dim=1)
    
    start = banks["ptr"]
    end = start + len(features)
    banks["features"] = torch.cat([banks["features"], features], dim=0)
    banks["probs"] = torch.cat([banks["probs"], probs], dim=0)
    banks["logit"] = torch.cat([banks["logit"], logits], dim=0)
    banks["index"] = torch.cat([banks["index"], idxs], dim=0)
    banks["gt"] = torch.cat([banks["gt"], label], dim=0)
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
    gm = GaussianMixture(n_components=2, random_state=0,max_iter=50)
    
    banks = {
        "features": torch.tensor([], device='cuda'),
        "probs": torch.tensor([], device='cuda'),
        "logit": torch.tensor([], device='cuda'),
        "index": torch.tensor([], device='cuda'),
        "ptr": 0,
        'gm':gm
    }
    if args.learn.add_gt_in_bank: 
        banks.update({"gt": torch.tensor([], device='cuda')})

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
        images, labels, idxs = data
        idxs = idxs.to("cuda")
        images_w, images_q, images_k = (
            images[0].to("cuda"),
            images[1].to("cuda"),
            images[2].to("cuda"),
        )
        
        # per-step scheduler
        # step = i + epoch * len(train_loader)
        # adjust_learning_rate(optimizer, step, args)
        
        # weak aug model output
        feats_w, logits_w = model(images_w, banks, idxs, cls_only=True)
        with torch.no_grad():
            probs_w = F.softmax(logits_w, dim=1)
            if use_proto := first_X_samples >= args.learn.online_length:
                args.learn.refine_method = "nearest_neighbors"
            else:
                args.learn.refine_method = None
                first_X_samples += len(feats_w)
                
            pseudo_labels_w, probs_w, _ = refine_predictions(
                model, feats_w, probs_w, banks, args=args, return_index=args.learn.return_index
            )
        
        # similarity btw prototype(mean) and feats_w
        if use_proto  and args.learn.component != 'pr': 
            # use confidence center
            cur_probs   = torch.cat([banks['probs'], probs_w], dim=0)
            cur_feats   = torch.cat([banks['features'], feats_w.detach()], dim=0) ## current 
            cur_pseudo  = torch.cat([banks['logit'].argmax(dim=1), pseudo_labels_w], dim=0) ## current 
            # cur_gt      = torch.cat([banks['gt'], labels], dim=0) ## current 

            confidence, _ = noise_detect_cls(cur_probs, cur_pseudo, cur_feats, banks, args)
            origin_idx = torch.arange(banks['probs'].size(0), banks['probs'].size(0)+probs_w.size(0))
            ignore_idx = confidence[origin_idx] < args.learn.conf_filter # select noise label
            args.learn.use_ce_weight = True
            args.learn.use_proto_loss_v2 = True
            args.learn.do_noise_detect = True
            model.confidence = confidence[torch.arange(banks['probs'].size(0))]
            aug_prototypes = get_center_proto(banks, use_confidence=False)
            prototypes = get_center_proto(banks, use_confidence=False)
        else: 
            args.learn.use_ce_weight = False
            args.learn.use_proto_loss_v2 = False
            args.learn.do_noise_detect = False
            aug_prototypes = None
            prototypes = None 
            ignore_idx = None

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
                args=args
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
        
        if args.learn.component == 'pr': 
            loss = (
            )

        elif args.learn.component == 'cr': 
            loss = (
                args.learn.alpha * loss_cls
            )

        elif args.learn.component == 'ccp': 
            loss = (
                args.learn.alpha * loss_cls
                + loss_mix
            )
        elif args.learn.component == 'div': 
            loss = (
                args.learn.alpha * loss_cls
                + loss_mix
                + args.learn.eta * loss_div
            )
        elif args.learn.component == 'inst': 
            loss = (
                args.learn.alpha * loss_cls
                + loss_mix
                + args.learn.eta * loss_div
                + args.learn.beta * loss_ins
            )

        elif args.learn.component == 'all': 
            loss = (
                args.learn.alpha * loss_cls
                + loss_mix
                + args.learn.eta * loss_div
                + args.learn.beta * loss_ins
            )
            if loss_proto: loss += loss_proto

        loss_meter.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # use slow feature to update neighbor space
        with torch.no_grad():
            feats_w, logits_w = model.momentum_model(images_w, return_feats=True)

        update_labels(banks, idxs, feats_w, logits_w, labels, args)
        

        if use_wandb(args):
            wandb_dict = {
                "loss_cls": args.learn.alpha * loss_cls.item(),
                "loss_ins": args.learn.beta * loss_ins.item(),
                "loss_div": args.learn.eta * loss_div.item(),
                "loss_mix": loss_mix.item(),
                "acc_ins": accuracy_ins.item(),    
                "lr": optimizer.param_groups[0]['lr']
            }
            
            if use_proto:
                matching_label = (labels.to("cuda") == logits_w.argmax(dim=1))
                clean_idx = confidence[origin_idx] > args.learn.conf_filter
                noise_accuracy = (clean_idx == matching_label).float().mean()
                # loss_meter.update(noise_accuracy)
                pre, rec, f1, _ =  precision_recall_fscore_support(matching_label.cpu().numpy(), clean_idx.cpu().numpy(), average='macro')
                wandb_dict.update({"noise_acc_ins": noise_accuracy.item()})
                wandb_dict.update({"rec_ins": rec})
                wandb_dict.update({"pre_ins": pre})

            if loss_proto: wandb_dict.update({"loss_proto": loss_proto.item()})

            wandb.log(wandb_dict, commit=(i != len(train_loader) - 1))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.learn.print_freq == 0:
            progress.display(i)
            