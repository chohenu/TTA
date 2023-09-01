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
from moco.builder import AdaMoCo, AdaMixCo, hwc_MoCo
from moco.loader import NCropsTransform
from utils import (
    adjust_learning_rate,
    concat_all_gather,
    get_augmentation,
    get_distances,
    is_master,
    per_class_accuracy,
    remove_wrap_arounds,
    save_checkpoint,
    use_wandb,
    AverageMeter,
    CustomDistributedDataParallel,
    ProgressMeter,
    get_tsne_map,
)

import random
import pickle
from losses import ClusterLoss
import pandas as pd
from sklearn.metrics import roc_auc_score
import math
from sklearn.mixture import GaussianMixture  ## numpy version

@torch.no_grad()
def eval_and_label_dataset(dataloader, model, banks, epoch, args):
    wandb_dict = dict()

    # make sure to switch to eval mode
    model.eval()
    # projector = model.src_model.projector_q
    clustering = model.src_model.classifier_q
    
    # run inference
    logits, gt_labels, indices, cluster_labels = [], [], [], []
    features, project_feats = [], []
    logging.info("Eval and labeling...")
    iterator = tqdm(dataloader) if is_master(args) else dataloader
    for imgs, labels, idxs in iterator:
        imgs = imgs.to("cuda")

        # (B, D) x (D, K) -> (B, K)
        feats, logits_cls = model(imgs, banks, idxs, cls_only=True)

        features.append(feats)
        # project_feats.append(F.normalize(projector(feats), dim=1))
        cluster_labels.append(F.softmax(clustering(feats), dim=1))
        logits.append(logits_cls)
        gt_labels.append(labels)
        indices.append(idxs)

    features = torch.cat(features)
    # project_feats  = torch.cat(project_feats)
    cluster_labels = torch.cat(cluster_labels)
    logits = torch.cat(logits)
    gt_labels = torch.cat(gt_labels).to("cuda")
    indices = torch.cat(indices).to("cuda")

    if args.distributed:
        # gather results from all ranks
        features = concat_all_gather(features)
        # project_feats = concat_all_gather(project_feats)
        cluster_labels = concat_all_gather(cluster_labels)
        logits = concat_all_gather(logits)
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

    # if epoch > -1: ## wandb logging Error image
    #     num_images = 10
    #     y_true, y_pred = gt_labels.cpu().numpy(), pred_labels.cpu().numpy()
    #     diff_ = y_true != y_pred
    #     diff_y_true, diff_y_pred, diff_indices = y_true[diff_], y_pred[diff_], indices[diff_]
    #     class_num ,idx = np.unique(diff_y_true, return_inverse = True)
    #     for i,cls_name in zip(class_num,class_name):
    #         class_diff_index = diff_indices[idx==i][:num_images]
    #         diff_class = diff_y_pred[idx==i][:num_images]
    #         diff_class = np.array([class_dict[i] for i in diff_class])
    #         image_list = [dataloader.dataset.__getitem__(i)[0] for i in class_diff_index]
    #         wandb_dict.update({f'Error_class_{cls_name}':wandb.Image(
    #                                         torch.concat(image_list,dim=2),
    #                                         caption=f"Diff_class_{np.array2string(diff_class)}")
                                            # })


    probs = F.softmax(logits, dim=1)
    rand_idxs = torch.randperm(len(features)).cuda()
    banks = {
        "features": features[rand_idxs][: args.learn.queue_size],
        "probs": probs[rand_idxs][: args.learn.queue_size],
        "logit": logits[rand_idxs][: args.learn.queue_size],
        "ptr": 0,
        "noram_features": F.normalize(features[rand_idxs][: args.learn.queue_size]),
    }
    if args.learn.add_gt_in_bank: 
        banks.update({"gt": gt_labels[rand_idxs]})

    if args.learn.return_index: 
        banks.update({"index": indices[rand_idxs]})
        
    if args.learn.do_noise_detect:
        logging.info(
            "Do Noise Detection"
        )
        noise_labels = pred_labels
        is_clean = gt_labels.cpu().numpy() == noise_labels.cpu().numpy()
        
        confidence = noise_detect_cls(cluster_labels, noise_labels, 
                                                                features, args)
        noise_accuracy = ((confidence > 0.5) == (gt_labels == noise_labels)).float().mean()
        logging.info(
            f"noise_accuracy: {noise_accuracy}"
        )
        context_noise_auc = roc_auc_score(is_clean, confidence.cpu().numpy())
        logging.info(f"noise_accuracy: {noise_accuracy}")
        logging.info(f"roc_auc_score: {context_noise_auc}")
        banks.update({"confidence": confidence[rand_idxs]})
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
def soft_k_nearest_neighbors(features, features_bank, probs_bank, args):
    pred_probs = []
    for feats in features.split(64):
        distances = get_distances(feats, features_bank, args.learn.dist_type)
        _, idxs = distances.sort()
        idxs = idxs[:, : args.learn.num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)

    return pred_labels, pred_probs, None

@torch.no_grad()
def selective_soft_k_nearest_neighbors(features, features_bank, probs_bank, args):
    near_avg_feats = []
    for feats in features.split(64):
        distances = get_distances(feats, features_bank, args.learn.dist_type)
        _, idxs = distances.sort()
        idxs = idxs[:, : args.learn.num_neighbors]
        # select reliable nearest neighborhood
        near_feat = features_bank[idxs, :]
        labels = torch.argmax(probs_bank, dim=1)
        sim_ = F.normalize(near_feat, dim=1) @ F.normalize(features_bank, dim=1).T
        _, idx_near_ = torch.topk(sim_, dim=-1, largest=True, k = 10 + 1)
        idx_near_ = idx_near_[:, :, 1:]
        near_labels = labels[idx_near_]
        near_avg_feat = features_bank[idx_near_, :].mean(dim=2)
        near_avg_feats.append(near_avg_feat)
    near_feat = torch.cat(near_avg_feats)
    return near_feat



@torch.no_grad()
def update_labels(banks, idxs, features, logits, labels, args):
    # 1) avoid inconsistency among DDP processes, and
    # 2) have better estimate with more data points
    if args.distributed:
        idxs = concat_all_gather(idxs)
        features = concat_all_gather(features)
        logits = concat_all_gather(logits)
        labels = concat_all_gather(labels.to("cuda"))
    
    probs = F.softmax(logits, dim=1)
    
    if args.learn.return_index:
        origin_idx = torch.where(idxs.reshape(-1,1)==banks['index'])[1]
        banks["features"][origin_idx, :] = features
        banks["probs"][origin_idx, :] = probs
        banks["logit"][origin_idx, :] = logits
        
        if args.learn.add_gt_in_bank:
            banks['gt'][origin_idx] = labels
    
        
    else:
        start = banks["ptr"]
        end = start + len(idxs)
        idxs_replace = torch.arange(start, end).cuda() % len(banks["features"])
        banks["features"][idxs_replace, :] = features
        banks["probs"][idxs_replace, :] = probs
        banks["logit"][idxs_replace, :] = logits
        banks["ptr"] = end % len(banks["features"])

        if args.learn.add_gt_in_bank:
            banks['gt'][idxs_replace] = labels
        

@torch.no_grad()
def refine_predictions(
    model,
    features,
    probs,
    banks,
    args,
    gt_labels=None,
    return_index=False,
):
    if args.learn.refine_method == "nearest_neighbors":
        feature_bank = banks["features"]
        probs_bank = banks["probs"]
        pred_labels, probs, stack_index = soft_k_nearest_neighbors(
            features, feature_bank, probs_bank, args
        )
    elif args.learn.refine_method == "nearest_feat":
        feature_bank = banks["features"]
        probs_bank = banks["probs"]
        near_feat = selective_soft_k_nearest_neighbors(
            features, feature_bank, probs_bank, args
        )
        logits = model.src_model.classifier_q(near_feat.reshape(-1, 256))
        logits = logits.reshape(near_feat.shape[0], 10, probs[0].shape[0])
        probs = F.softmax(logits, dim=-1).mean(1)
        pred_labels = torch.argmax(probs, dim=1)
    elif args.learn.refine_method is None:
        pred_labels = probs.argmax(dim=1)
    else:
        raise NotImplementedError(
            f"{args.learn.refine_method} refine method is not implemented."
        )
    accuracy = None
    if gt_labels is not None:
        accuracy = (pred_labels == gt_labels).float().mean() * 100
   
    return pred_labels, probs, accuracy


def get_augmentation_versions(args):
    """
    Get a list of augmentations. "w" stands for weak, "s" stands for strong.

    E.g., "wss" stands for one weak, two strong.
    """
    transform_list = []
    for version in args.learn.aug_versions:
        if version == "s":
            transform_list.append(get_augmentation(args.data.aug_type))
        elif version == "w":
            transform_list.append(get_augmentation("plain"))
        else:
            raise NotImplementedError(f"{version} version not implemented.")
    transform = NCropsTransform(transform_list)

    return transform


def get_target_optimizer(model, args):
    if args.distributed:
        model = model.module
    backbone_params, extra_params = (
        model.src_model.get_params()
        if hasattr(model, "src_model")
        else model.get_params()
    )

    if args.optim.name == "sgd":
        optimizer = torch.optim.SGD(
            [
                {
                    "params": backbone_params,
                    "lr": args.optim.lr,
                    "momentum": args.optim.momentum,
                    "weight_decay": args.optim.weight_decay,
                    "nesterov": args.optim.nesterov,
                },
                {
                    "params": extra_params,
                    "lr": args.optim.lr * 10,
                    "momentum": args.optim.momentum,
                    "weight_decay": args.optim.weight_decay,
                    "nesterov": args.optim.nesterov,
                },
            ]
        )
    else:
        raise NotImplementedError(f"{args.optim.name} not implemented.")

    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]  # snapshot of the initial lr

    return optimizer


def train_target_domain(args):
    logging.info(
        f"Start target training on {args.data.src_domain}-{args.data.tgt_domain}..."
    )

    # if not specified, use the full length of dataset.
    if args.learn.queue_size == -1:
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
    src_model = Classifier(args, train_target, checkpoint_path)
    momentum_model = Classifier(args, train_target, checkpoint_path)
    if args.model_tta.type == "moco":
        model = AdaMoCo(
            src_model,
            momentum_model,
            K=args.model_tta.queue_size,
            m=args.model_tta.m,
            T_moco=args.model_tta.T_moco,
        ).cuda()
    elif args.model_tta.type == "mixco":
        model = AdaMixCo(
            src_model,
            momentum_model,
            K=args.model_tta.queue_size,
            m=args.model_tta.m,
            T_moco=args.model_tta.T_moco,
        ).cuda()
    elif args.model_tta.type == "sfda":
        model = hwc_MoCo(
            src_model,
            momentum_model,
            K=args.model_tta.queue_size,
            m=args.model_tta.m,
            T_moco=args.model_tta.T_moco,
        ).cuda()
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = CustomDistributedDataParallel(model, device_ids=[args.gpu])
    logging.info(f"1 - Created target model")

    val_transform = get_augmentation("test")
    label_file = os.path.join(args.data.image_root, f"{args.data.tgt_domain}_list.txt")
    val_dataset = ImageList(
        image_root=args.data.image_root,
        label_file=label_file,
        transform=val_transform,
    )
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=256, sampler=val_sampler, num_workers=2
    )
    pseudo_item_list, banks = eval_and_label_dataset(
        val_loader, model, banks=None, epoch=-1, args=args
    )
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
            
        # train for one epoch
        if args.model_tta.type == "moco":
            train_epoch_moco(train_loader, model, banks,
                        optimizer, epoch, args)
        elif args.model_tta.type == "mixco":
            train_epoch_mixco(train_loader, model, banks,
                        optimizer, epoch, args)
        elif args.model_tta.type == "sfda": 
            train_epoch_sfda(train_loader, model, banks,
                        optimizer, epoch, args)
        
        eval_and_label_dataset(val_loader, model, banks, epoch, args)

    if is_master(args):
        filename = f"checkpoint_{epoch:04d}_{args.data.src_domain}-{args.data.tgt_domain}-{args.sub_memo}_{args.seed}.pth.tar"
        save_path = os.path.join(args.log_dir, filename)
        save_checkpoint(model, optimizer, epoch, save_path=save_path)
        logging.info(f"Saved checkpoint {save_path}")
        
        
def train_epoch_moco(train_loader, model, banks,
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
        step = i + epoch * len(train_loader)
        adjust_learning_rate(optimizer, step, args)
        
        # weak aug model output
        feats_w, logits_w = model(images_w, cls_only=True)
        with torch.no_grad():
            probs_w = F.softmax(logits_w, dim=1)
            pseudo_labels_w, probs_w, _ = refine_predictions(
                feats_w, probs_w, banks, args=args, return_index=args.learn.return_index
            )
        # strong aug model output 
        feats_q, logits_q, logits_ins, feats_k, logits_k = model(images_q, images_k)
        
        # similarity btw prototype(mean) and feats_w
        prototypes, similarity_a = prototype_cluster(banks, feats_w)
        
        # similarity btw prototype(GMM) and feats_w
        # similarity_a = soft_gmm_clustering(banks, feats_w)
    
        # mixup
        alpha = 1.0
        inputs, targets_a, targets_b, lam = mixup_data(images_w, pseudo_labels_w,
                                                        alpha, use_cuda=True)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                    targets_a, targets_b))
        
        targets_mix = lam * torch.eye(args.num_clusters).cuda()[targets_a] + (1-lam) * torch.eye(args.num_clusters).cuda()[targets_b]
        
        feats_mix, target_mix_logit = model(inputs, cls_only=True)
        
        target_mix_hat = F.softmax(target_mix_logit, dim=1)
        
        # similarity btw prototype and mixup input
        prototypes, similarity_b = prototype_cluster(banks, feats_mix)
        
        # similarity btw prototype(GMM) and feats_w
        # similarity_b = soft_gmm_clustering(banks, feats_mix)
        
        loss_mix = mixup_criterion(target_mix_logit, targets_a, targets_b, lam=lam)
        loss_mix += KLLoss(similarity_a, probs_w, epsilon=1e-8)
        loss_mix += KLLoss(similarity_b, targets_mix, epsilon=1e-8)
        
        
            
        # Calculate reliable degree of pseudo labels
        if args.learn.use_ce_weight:
            with torch.no_grad():
                CE_weight = calculate_reliability(probs_w, feats_w, feats_q, feats_k)
        else:
            CE_weight = 1.
        
        # update key features and corresponding pseudo labels
        model.update_memory(feats_k, pseudo_labels_w, labels)


        # moco instance discrimination
        loss_ins, accuracy_ins = instance_loss(
            logits_ins=logits_ins,
            pseudo_labels=pseudo_labels_w,
            mem_labels=model.mem_labels,
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
            # args.learn.alpha * loss_cls
            args.learn.beta * loss_ins
            + args.learn.eta * loss_div
            + loss_mix 
        )
        
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
                # "loss_cls": args.learn.alpha * loss_cls.item(),
                "loss_ins": args.learn.beta * loss_ins.item(),
                "loss_div": args.learn.eta * loss_div.item(),
                "loss_mix": loss_mix.item(),
                "acc_ins": accuracy_ins.item(),
            }
            
            wandb.log(wandb_dict, commit=(i != len(train_loader) - 1))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.learn.print_freq == 0:
            progress.display(i)
        
                
def train_epoch_mixco(train_loader, model, banks,
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
        step = i + epoch * len(train_loader)
        adjust_learning_rate(optimizer, step, args)
        
        # weak aug model output
        feats_w, logits_w = model(images_w, cls_only=True)
        with torch.no_grad():
            probs_w = F.softmax(logits_w, dim=1)
            pseudo_labels_w, probs_w, _ = refine_predictions(
                feats_w, probs_w, banks, args=args, return_index=args.learn.return_index
            )
        
        # similarity btw prototype(mean) and feats_w
        prototypes, similarity_a = prototype_cluster(banks, feats_w)
        
        # similarity btw prototype(GMM) and feats_w
        # similarity_a = soft_gmm_clustering(banks, feats_w)
    
        # mixup
        alpha = 1.0
        inputs, targets_a, targets_b, lam = mixup_data(images_w, pseudo_labels_w,
                                                        alpha, use_cuda=True)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                    targets_a, targets_b))
        
        targets_mix = lam * torch.eye(args.num_clusters).cuda()[targets_a] + (1-lam) * torch.eye(args.num_clusters).cuda()[targets_b]
        
        feats_mix, target_mix_logit = model(inputs, cls_only=True)
        
        target_mix_hat = F.softmax(target_mix_logit, dim=1)
        
        # similarity btw prototype and mixup input
        prototypes, similarity_b = prototype_cluster(banks, feats_mix)
        
        # similarity btw prototype(GMM) and feats_w
        # similarity_b = soft_gmm_clustering(banks, feats_mix)
        
        loss_mix = mixup_criterion(target_mix_logit, targets_a, targets_b, lam=lam)
        loss_mix += KLLoss(similarity_a, probs_w, epsilon=1e-8)
        loss_mix += KLLoss(similarity_b, targets_mix, epsilon=1e-8)
        
        
        # strong aug model output 
        logits_mix, lbls_mix, logits_ins, feats_k, logits_k, logits_q, index = model(images_q, images_k)
        
            
        # Calculate reliable degree of pseudo labels
        CE_weight = 1.
        
        # update key features and corresponding pseudo labels
        model.update_memory(feats_k, pseudo_labels_w, labels)


        # moco instance discrimination
        loss_ins, accuracy_ins = instance_loss(
            logits_ins=logits_ins,
            pseudo_labels=pseudo_labels_w,
            mem_labels=model.mem_labels,
            contrast_type=args.learn.contrast_type,
        )
        # instance accuracy shown for only one process to give a rough idea
        top1_ins.update(accuracy_ins.item(), len(logits_ins))
        
        # mixco instance discrimination
        loss_mixco = mixco_loss(
            logits_ins=logits_mix,
            labels_ins=lbls_mix,
            pseudo_labels_a=pseudo_labels_w,
            pseudo_labels_b=pseudo_labels_w[index],
            mem_labels=model.mem_labels,
            contrast_type=args.learn.contrast_type,
        )

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
            # args.learn.alpha * loss_cls
            # args.learn.beta * loss_ins
            args.learn.eta * loss_div
            + loss_mix
            + loss_mixco
        )
        
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
                # "loss_cls": args.learn.alpha * loss_cls.item(),
                # "loss_ins": args.learn.beta * loss_ins.item(),
                "loss_div": args.learn.eta * loss_div.item(),
                "loss_mix": loss_mix.item(),
                "loss_mixco": loss_mixco.item(),
                "acc_ins": accuracy_ins.item(),
            }
            
            wandb.log(wandb_dict, commit=(i != len(train_loader) - 1))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.learn.print_freq == 0:
            progress.display(i)

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
        step = i + epoch * len(train_loader)
        adjust_learning_rate(optimizer, step, args)
        
        origin_idx = torch.where(idxs.reshape(-1,1)==banks['index'])[1]
        
        # weak aug model output
        feats_w, logits_w = model(images_w, banks, idxs, cls_only=True)
        with torch.no_grad():
            probs_w = F.softmax(logits_w, dim=1)
            pseudo_labels_w, probs_w, _ = refine_predictions(
                model, feats_w, probs_w, banks, args=args, return_index=args.learn.return_index
            )
        # strong aug model output 
        feats_q, logits_q, logits_ins, feats_k, logits_k, logits_neg_near = model(images_q, banks, idxs, images_k, pseudo_labels_w, epoch)
        
        
        
        # similarity btw prototype(mean) and feats_w
        prototypes, similarity_a = prototype_cluster(banks, feats_w)
        # prototypes, similarity_a = soft_gmm_clustering(banks, feats_w)
        
    
        # mixup
        alpha = 1.0
        inputs, targets_a, targets_b, lam, mix_idx = mixup_data(images_w, pseudo_labels_w,
                                                        alpha, use_cuda=True)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                    targets_a, targets_b))
        
        targets_mix = lam * torch.eye(args.num_clusters).cuda()[targets_a] + (1-lam) * torch.eye(args.num_clusters).cuda()[targets_b]
        
        feats_mix, target_mix_logit = model(inputs, banks, idxs, cls_only=True)
        
        target_mix_hat = F.softmax(target_mix_logit, dim=1)
        
        # similarity btw prototype and mixup input
        prototypes, similarity_b = prototype_cluster(banks, feats_mix)
        # prototypes, similarity_b = soft_gmm_clustering(banks, feats_mix)
        
        if args.learn.do_noise_detect:
        
            confidence = noise_detect_proto(prototypes, banks, args, temp=0.25)
        
    
        if args.learn.use_mixup_weight and args.learn.do_noise_detect:
            weight_a = torch.exp(banks["confidence"][origin_idx])
            weight_b = torch.exp(weight_a[mix_idx])
            loss_mix = mixup_criterion(target_mix_logit, targets_a, targets_b, lam=lam, weight_a=weight_a, weight_b=weight_b)
        elif args.learn.use_mixup_weight_2 and args.learn.do_noise_detect:
            weight_a = confidence[origin_idx]
            weight_b = weight_a[mix_idx]
            weight_mix = lam * weight_a + (1-lam) * weight_b
            loss_mix = KLLoss(target_mix_logit, targets_mix, epsilon=1e-8, weight_mix=weight_mix)
            # loss_mix = KLLoss(target_mix_logit, targets_mix, epsilon=1e-8)
        elif args.learn.use_mixup_regular:
            if args.learn.mixup_reg_type=='prototype':
                sim_cluster_a = F.normalize(feats_mix, dim=1) @ F.normalize(prototypes, dim=1).T
                sim_cluster_a = (sim_cluster_a*torch.eye(args.num_clusters).cuda()[targets_a]).sum(dim=1)
                sim_cluster_b = F.normalize(feats_mix, dim=1) @ F.normalize(prototypes, dim=1).T
                sim_cluster_b = (sim_cluster_b*torch.eye(args.num_clusters).cuda()[targets_b]).sum(dim=1)
                pred_ratio = sim_cluster_a/sim_cluster_b
            elif args.learn.mixup_reg_type=='sample':
                sim_cluster_a = F.normalize(feats_mix, dim=1) @ F.normalize(feats_w, dim=1).T
                sim_cluster_b = F.normalize(feats_mix, dim=1) @ F.normalize(feats_w[mix_idx], dim=1).T
                pred_ratio = torch.diag(sim_cluster_a)/torch.diag(sim_cluster_b)
            target_ratio = torch.tensor((lam/(1-lam))).cuda().repeat(pred_ratio.shape[0])
            loss_mix = torch.nn.L1Loss()(pred_ratio, target_ratio)
        else:
            loss_mix = mixup_criterion(target_mix_logit, targets_a, targets_b, lam=lam, weight_a=None, weight_b=None)
        loss_mix += KLLoss(similarity_a, probs_w, epsilon=1e-8)
        loss_mix += KLLoss(similarity_b, targets_mix, epsilon=1e-8)            
            
        # Calculate reliable degree of pseudo labels
        if args.learn.use_ce_weight and args.learn.do_noise_detect:
            CE_weight = torch.exp(confidence[origin_idx])
            # if epoch < 3:
            #     CE_weight = torch.exp(confidence[origin_idx])
            # else:
            #     CE_weight = 1.
        else:
            CE_weight = 1.
        
        # update key features and corresponding pseudo labels
        model.update_memory(feats_k, pseudo_labels_w, labels)
        
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
        if args.learn.ce_sup_type == "weak_strong_kl":
            loss_cls, accuracy_psd = classification_loss(
            logits_w, logits_q, probs_w, CE_weight, args)
        else:
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
            }
            
            wandb.log(wandb_dict, commit=(i != len(train_loader) - 1))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.learn.print_freq == 0:
            progress.display(i)
            
def noise_detect_cls(cluster_labels, labels, features, args, temp=0.25):
    labels = labels.long()
    centers = F.normalize(cluster_labels.T.mm(features), dim=1)
    context_assigments_logits = features.mm(centers.T) / temp
    context_assigments = F.softmax(context_assigments_logits, dim=1)
    losses = - context_assigments[torch.arange(labels.size(0)), labels]
    losses = losses.cpu().numpy()[:, np.newaxis]
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    losses = np.nan_to_num(losses)
    labels = labels.cpu().numpy()
    
    from sklearn.mixture import GaussianMixture
    confidence = np.zeros((losses.shape[0],))
    if args.learn.sep_gmm:
        for i in range(cluster_labels.size(1)):
            mask = labels == i
            c = losses[mask, :]
            gm = GaussianMixture(n_components=2, random_state=2).fit(c)
            pdf = gm.predict_proba(c)
            # label이 얼마나 clean한지 정량적으로 표현
            confidence[mask] = (pdf / pdf.sum(1)[:, np.newaxis])[:, np.argmin(gm.means_)]
    else:
        gm = GaussianMixture(n_components=2, random_state=0).fit(losses)
        pdf = gm.predict_proba(losses)
        confidence = (pdf / pdf.sum(1)[:, np.newaxis])[:, np.argmin(gm.means_)]
    confidence = torch.from_numpy(confidence).float().cuda()
    return confidence

def noise_detect_proto(prototypes, banks, args, temp=0.25):
    features = F.normalize(banks["features"], dim=1)
    labels = torch.argmax(banks["probs"], dim=1)
    labels = labels.long()
    centers = F.normalize(prototypes, dim=1)
    sim_w_center_logits = features @ centers.T / temp
    sim_w_center = F.softmax(sim_w_center_logits, dim=1)
    losses = - sim_w_center[torch.arange(labels.size(0)), labels]
    losses = losses.cpu().numpy()[:, np.newaxis]
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    # if np.isnan(losses).sum() > 0:
    #     print(1)
    losses = np.nan_to_num(losses)
    labels = labels.cpu().numpy()
    
    from sklearn.mixture import GaussianMixture
    confidence = np.zeros((losses.shape[0],))
    if args.learn.sep_gmm:
        for i in range(args.num_clusters):
            mask = labels == i
            c = losses[mask, :]
            gm = GaussianMixture(n_components=2, random_state=2).fit(c)
            pdf = gm.predict_proba(c)
            # label이 얼마나 clean한지 정량적으로 표현
            confidence[mask] = (pdf / pdf.sum(1)[:, np.newaxis])[:, np.argmin(gm.means_)]
    else:
        gm = GaussianMixture(n_components=2, random_state=0).fit(losses)
        pdf = gm.predict_proba(losses)
        confidence = (pdf / pdf.sum(1)[:, np.newaxis])[:, np.argmin(gm.means_)]
    confidence = torch.from_numpy(confidence).float().cuda()
    return confidence

@torch.no_grad()
def soft_gmm_clustering(banks, features):
    
    feature_bank = banks["features"]
    probs_bank = banks["probs"]
    
    clss_num = probs_bank.size(1)
    uniform = torch.ones(len(feature_bank),clss_num)/clss_num
    uniform = uniform.cuda()

    pi = probs_bank.sum(dim=0)
    mu = torch.matmul(probs_bank.t(),(feature_bank)) # matrix multiple (F,C) center??
    mu = mu / pi.unsqueeze(dim=-1).expand_as(mu) # normalize first 

    zz, gamma = gmm((feature_bank), pi, mu, uniform)
    pred_labels = gamma.argmax(dim=1)
    
    for round in range(1):
        pi = gamma.sum(dim=0)
        mu = torch.matmul(gamma.t(), (feature_bank))
        mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

        zz, gamma = gmm((feature_bank), pi, mu, gamma)
        pred_labels = gamma.argmax(axis=1)
        
    prototypes = mu
    
    similarity = F.normalize(features, dim=1) @ F.normalize(prototypes, dim=1).T
    similarity = F.softmax(similarity, dim=1)
    
    return prototypes, similarity

@torch.no_grad()
def gmm(all_fea, pi, mu, all_output):    
    epsilon = 1e-6
    Cov = []
    dist = []
    log_probs = []
    
    for i in range(len(mu)):
        temp = all_fea - mu[i]
        predi = all_output[:,i].unsqueeze(dim=-1)
        Covi = torch.matmul(temp.t(), temp * predi.expand_as(temp)) / (predi.sum()) + epsilon * torch.eye(temp.shape[1]).cuda()
        try:
            chol = torch.linalg.cholesky(Covi)
        except RuntimeError:
            Covi += epsilon * torch.eye(temp.shape[1]).cuda() * 100
            chol = torch.linalg.cholesky(Covi)
        chol_inv = torch.inverse(chol)
        Covi_inv = torch.matmul(chol_inv.t(), chol_inv)
        logdet = torch.logdet(Covi)
        mah_dist = (torch.matmul(temp, Covi_inv) * temp).sum(dim=1)
        log_prob = -0.5*(Covi.shape[0] * np.log(2*math.pi) + logdet + mah_dist) + torch.log(pi)[i]
        Cov.append(Covi)
        log_probs.append(log_prob)
        dist.append(mah_dist)
    Cov = torch.stack(Cov, dim=0)
    dist = torch.stack(dist, dim=0).t()
    log_probs = torch.stack(log_probs, dim=0).t()
    zz = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True).expand_as(log_probs)
    gamma = torch.exp(zz)
    
    return zz, gamma

def prototype(banks, features):
    feature_bank = banks["features"]
    probs_bank = banks["probs"]
    
    prototypes = probs_bank.T.mm(feature_bank)
    
    similarity = F.normalize(features, dim=1) @ F.normalize(prototypes, dim=1).T
    similarity = F.softmax(similarity, dim=1)
    
    return prototypes, similarity
   
def prototype_cluster(banks, features):
    feature_bank = banks["features"]
    probs_bank = banks["probs"]
    pseudo_label_bank = probs_bank.argmax(dim=1)
    
    n_cluster = probs_bank.shape[1]
    prototypes = []
    for i in range(n_cluster):
        idxs = torch.where(pseudo_label_bank==i)[0]
        if idxs.shape[0] > 0:
            prototype = feature_bank[idxs].mean(0) # average of samples with same labels
        else:
            prototype = torch.full((feature_bank[idxs].mean(0).shape[0],), 1e-8).cuda()
        prototypes.append(prototype.unsqueeze(0))
    prototypes = torch.cat(prototypes, dim=0)
    
    similarity = F.normalize(features, dim=1) @ F.normalize(prototypes, dim=1).T
    similarity = F.softmax(similarity, dim=1)
    
    return prototypes, similarity
 
def entropy(p, axis=1):
    return -torch.sum(p * torch.log2(p+1e-5), dim=axis)

def norm(x):
    return (x - x.min()) / (x.max()-x.min())

@torch.no_grad()
def calculate_reliability(probs_w, feats_w, feats_q, feats_k):
    # sim_w = norm(feats_w.mm(centers.T))
    # sim_q = norm(feats_q.mm(centers.T))
    # sim_k = norm(feats_k.mm(centers.T))
    # sim_avg = F.softmax((sim_w + sim_q + sim_k)/3, dim=1)
    max_entropy = torch.log2(torch.tensor(probs_w.size(1)))
    w = entropy(probs_w)
    # w = entropy(sim_avg)
    
    w = w / max_entropy
    # w = w * confidence
    w = torch.exp(-w)
    
    return w
    

@torch.no_grad()
def calculate_acc(logits, labels):
    preds = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean() * 100
    return accuracy

def cluster_loss(): 
    return ClusterLoss()

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
        mask[:, 1:] = pseudo_labels.reshape(-1, 1) != mem_labels  # (B, K)
        
        _, idx_near = torch.topk(logits_neg_near, k=5, dim=-1, largest=True)
        mask[:, 1:] *= torch.all(pseudo_labels.unsqueeze(1).repeat(1,5).unsqueeze(1) != mem_labels[idx_near].unsqueeze(0), dim=2) # (B, K)
        logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())

    loss = F.cross_entropy(logits_ins, labels_ins)

    accuracy = calculate_acc(logits_ins, labels_ins)

    return loss, accuracy

def mixco_loss(logits_ins, labels_ins, pseudo_labels_a, pseudo_labels_b, mem_labels, 
               contrast_type, epsilon=1e-8):
    N = pseudo_labels_a.shape[0]
    # in class_aware mode, do not contrast with same-class samples
    if contrast_type == "class_aware" and pseudo_labels_a is not None:
        mask = torch.ones_like(logits_ins, dtype=torch.bool)
        mask[:, 2*N:] = (pseudo_labels_a.reshape(-1, 1) != mem_labels) * (pseudo_labels_b.reshape(-1, 1) != mem_labels)
        logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())
    
    probs = F.softmax(logits_ins, dim=1)
    nll_loss = (-labels_ins * torch.log(probs + epsilon)).sum(dim=1).mean(0)

    return nll_loss


def classification_loss(logits_w, logits_s, target_labels, CE_weight, args):
    if not args.learn.do_noise_detect: CE_weight = 1.
    
    if args.learn.ce_sup_type == "weak_weak":
        loss_cls = (CE_weight * cross_entropy_loss(logits_w, target_labels, args)).mean()
        
    elif args.learn.ce_sup_type == "weak_strong":
        loss_cls = (CE_weight * cross_entropy_loss(logits_s, target_labels, args)).mean()
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


def smoothed_cross_entropy(logits, labels, num_classes, epsilon=0):
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        targets = (1 - epsilon) * targets + epsilon / num_classes
    loss = (-targets * log_probs).sum(dim=1).mean()

    return loss


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

def sfda_loss(banks, pseudo_labels_w, images_q, feats_q, logits_q, idxs):
    with torch.no_grad():
        output_f_norm = F.normalize(feats_q)
        output_f_ = output_f_norm
        softmax_out = torch.nn.Softmax(dim=1)(logits_q)

        origin_idx = torch.where(idxs.reshape(-1,1)==banks['index'])[1]
        banks['noram_features'][origin_idx] = output_f_
        banks['probs'][origin_idx] = softmax_out

        distance = output_f_ @ banks['noram_features'].T
        _, idx_near = torch.topk(distance, dim=-1, largest=True, k=5 + 1)
        idx_near = idx_near[:, 1:]  # batch x K
        score_near = banks['probs'][idx_near]  # batch x K x C

    # nn
    softmax_out_un = softmax_out.unsqueeze(1).expand(
        -1, 5, -1
    )  # batch x K x C

    loss_ins = torch.mean(
        (F.kl_div(softmax_out_un, score_near, reduction="none").sum(-1)).sum(1)
    ) # Equal to dot product

    mask = torch.ones((images_q.shape[0], images_q.shape[0])).cuda()
    diag_num = torch.diag(mask)
    mask_diag = torch.diag_embed(diag_num)
    mask = mask - mask_diag
    ## peu
    # pseudo_labels_w
    one_hot_vector = torch.eye(12).cuda()[pseudo_labels_w] ## 64,12
    filtering_mask = one_hot_vector @ one_hot_vector.T # 64,12 & 12,64 -> 64x64
    mask *= 1-filtering_mask

    copy = softmax_out.T  # .detach().clone()#

    dot_neg = softmax_out @ copy  # batch x batch

    dot_neg = (dot_neg * mask.cuda()).sum(-1)  # batch
    neg_pred = torch.mean(dot_neg)
    loss_ins += neg_pred * 1
    
    # accuracy_ins = calculate_acc(softmax_out.argmax(dim=1), pseudo_labels_w)
    accuracy_ins = (softmax_out.argmax(dim=1) == pseudo_labels_w).float().mean() * 100
    return loss_ins, accuracy_ins
    