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
from image_list import ImageList, mixup_data, fix_mixup_data
from moco.builder import CNA_MoCo
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
import math
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture  ## numpy version
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from sklearn.metrics import roc_auc_score, confusion_matrix

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
    mix_features, mix_labels, mix_logit, mix_index = [], [], [], []
    alpha = []
    logging.info("Eval and labeling...")
    iterator = tqdm(dataloader) if is_master(args) else dataloader
    for data in iterator:
        images, labels, idxs = data
        imgs = images[0].to("cuda")

        # (B, D) x (D, K) -> (B, K)
        feats, logits_cls = model(imgs, cls_only=True)

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
    
    if args.learn.do_noise_detect:
        logging.info(
            "Do Noise Detection"
        )
        noise_labels = pred_labels
        is_clean = gt_labels.cpu().numpy() == noise_labels.cpu().numpy()
        
        confidence, losses = noise_detect_cls(cluster_labels, noise_labels, features, banks, args)
        # confidence, = noise_detect_cls(cluster_labels, noise_labels, features, args, return_center=True)
        match_confi = confidence > 0.5 #
        match_label = gt_labels == noise_labels
        noise_accuracy = (match_confi == match_label).float().mean()
        only_clean_accuracy = ((match_confi == True) & (match_label == True)).float().mean()
        only_noise_accuracy = ((match_confi == False) & (match_label == True)).float().mean()
        pre, rec, f1, _ =  precision_recall_fscore_support(match_confi.cpu().numpy(), match_label.cpu().numpy(), average='macro')
        avg_pre_rec = average_precision_score(match_label.cpu().numpy(), match_confi.cpu().numpy())
        only_noise_accuracy = (pred_labels[~match_confi] == gt_labels[~match_confi]).float().mean()
        
        logging.info(
            f"noise_accuracy: {noise_accuracy}"
        )
        context_noise_auc = roc_auc_score(is_clean, confidence.cpu().numpy())
        logging.info(f"noise_accuracy: {noise_accuracy}")
        logging.info(f"only_clean_accuracy: {only_clean_accuracy}")
        logging.info(f"only_noise_accuracy: {only_noise_accuracy}")
        logging.info(f"roc_auc_score: {context_noise_auc}")
        logging.info(f"noise_precision: {pre}")
        logging.info(f"noise_recall: {rec}")
        logging.info(f"noise_f1score: {f1}")
        logging.info(f"noise_avg_recall_precision: {avg_pre_rec}")
        
        banks.update({"confidence": confidence[rand_idxs]})
        banks.update({"mix_confidence": confidence[mix_index][rand_idxs]})
        banks.update({"distance": losses[rand_idxs.cpu()]})


        wandb_dict['noise_accuracy']   = noise_accuracy
        wandb_dict['noise_precision']   = pre
        wandb_dict['noise_racall']      = rec
        wandb_dict['noise_f1score']     = f1
        wandb_dict['noise_avg_rec_pre'] = avg_pre_rec
        wandb_dict["only_clean_accuracy"]=only_clean_accuracy
        wandb_dict["only_noise_accuracy"]=only_noise_accuracy
        wandb_dict["context_noise_auc"]=context_noise_auc
        
    if False and use_wandb(args):
        import os
        save_dir = str(wandb.run.dir)
        logging.info(f"Saving Memory Bank : {save_dir}")
        with open(f'{save_dir}/mix_val_{epoch}.pickle','wb') as fw:
            pickle.dump(banks, fw)


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

    if epoch > -1 and args.learn.use_confidence_instance_loss: 
        if "confidence" in banks:    
            model.find_confidence(banks)

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
def soft_k_nearest_near_noise(features, features_bank, probs_bank, confidence_bank, args):
    pred_probs = []
    for feats in features.split(64):
        distances = get_distances(feats, features_bank, args.learn.dist_type)
        _, idxs = distances.sort()
        idxs = idxs[:, : args.learn.num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        mask = (confidence_bank[idxs] > 0.5).unsqueeze(-1)
        probs_ = (probs_bank[idxs, :]*mask)
        probs = (probs_.sum(dim=1)/(mask.sum(dim=1)+1e-8))
        pred_probs.append(probs)
    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)

    return pred_labels, pred_probs, None

@torch.no_grad()
def update_aug_labels(banks, idxs, features, logits, labels, args):
    # 1) avoid inconsistency among DDP processes, and
    # 2) have better estimate with more data points
    if args.distributed:
        idxs = concat_all_gather(idxs)
        features = concat_all_gather(features)
        logits = concat_all_gather(logits)
        
    probs = F.softmax(logits, dim=1)
    
    if args.learn.return_index:
        origin_idx = torch.where(idxs.reshape(-1,1)==banks['index'])[1]
        banks["aug_features"][origin_idx, :] = features
        banks["aug_probs"][origin_idx, :] = probs
        banks["aug_logit"][origin_idx, :] = logits
        
    else:
        start = banks["ptr"]
        end = start + len(idxs)
        idxs_replace = torch.arange(start, end).cuda() % len(banks["aug_features"])
        banks["aug_features"][idxs_replace, :] = features
        banks["aug_probs"][idxs_replace, :] = probs
        
        if args.learn.add_gt_in_bank:
            banks['gt'][idxs_replace] = labels

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

    elif args.learn.refine_method == "nearest_noise":
        confidence_bank = banks["confidence"]
        feature_bank = banks["features"]
        probs_bank = banks["probs"]
        pred_labels, probs, stack_index = soft_k_nearest_near_noise(
        features, feature_bank, probs_bank, confidence_bank, args
        )

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


def get_augmentation_versions(args, train=True):
    """
    Get a list of augmentations. "w" stands for weak, "s" stands for strong.

    E.g., "wss" stands for one weak, two strong.
    """
    augmentations = args.learn.aug_versions if train == True else args.learn.val_aug_versions
    transform_list = []
    for version in augmentations:
        if version == "s":
            transform_list.append(get_augmentation(args.data.aug_type))
        elif version == "w":
            transform_list.append(get_augmentation("plain"))
        elif version == "o":
            transform_list.append(get_augmentation("test"))
        elif version == "c":
            transform_list.append(get_augmentation("rand_crop"))
        elif version == "f":
            transform_list.append(get_augmentation("five_crop"))
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
                    "lr": args.optim.lr * args.optim.time,
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
        if args.data.dataset.lower() == 'pacs': 
            label_file = os.path.join(
                args.data.image_root, f"{args.data.tgt_domain}_test_kfold.txt"
            )
        elif args.data.dataset.lower() == 'domainnet': 
            label_file = os.path.join(
                args.data.image_root, f"{args.data.tgt_domain}_concat.txt"
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
    elif args.data.dataset.lower() == 'domainnet': 
        label_file = os.path.join(args.data.image_root, f"{args.data.tgt_domain}_concat.txt")
    else: 
        label_file = os.path.join(args.data.image_root, f"{args.data.tgt_domain}_list.txt")
    val_dataset = ImageList(
        image_root=args.data.image_root,
        label_file=label_file,
        transform=val_transform,
    )
    model = CNA_MoCo(
        src_model,
        momentum_model,
        K=args.model_tta.queue_size,
        m=args.model_tta.m,
        T_moco=args.model_tta.T_moco,
        dataset_legth=len(val_dataset),
        args=args
    ).cuda()
    
    if args.ckpt_path: 
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        print(ckpt.keys(),'11111')
        state_dict = dict()
        for name, param in ckpt["state_dict"].items():
            # get rid of 'module.' prefix brought by DDP
            if 'mem' in name : 
                state_dict[name] = []
            else: 
                name = name.replace("module.", "")
                state_dict[name] = param
        model.load_state_dict(state_dict, strict=False)
        logging.info(f"0 - Succes ckpt weight")

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
        gm = GaussianMixture(n_components=2, random_state=0)

    pseudo_item_list, banks = eval_and_label_dataset(
        val_loader, model, banks=None, epoch=-1, gm=gm, args=args
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
    if not args.do_inference: 
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
    top1_nos = AverageMeter("NOS_Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, loss_meter, top1_ins, top1_psd, top1_nos],
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
        confidence = banks['confidence']
        ignore_idx = confidence[origin_idx] < args.learn.conf_filter # select noise label

        # weak aug model output
        feats_w, logits_w = model(images_w, cls_only=True)
        with torch.no_grad():
            probs_w = F.softmax(logits_w, dim=1)
            pseudo_labels_w, probs_w, _ = refine_predictions(
                model, feats_w, probs_w, banks, args=args, return_index=args.learn.return_index
            )
        
        # similarity btw prototype(mean) and feats_w
        aug_prototypes = prototype_cluster(banks, use_aug_key=True)
        prototypes = get_center_proto(banks, use_confidence=False)
        # aug_prototypes= prototype_cluster(banks, use_aug_key=True)

        # strong aug model output 
        feats_q, logits_q, logits_ins, feats_k, logits_k, logits_neg_near, loss_proto = model(images_q, images_k, prototypes_q=prototypes, args=args)
        
        # mixup
        alpha = 1.0
        inputs_w, targets_w_a, targets_w_b, lam_w, mix_w_idx = mixup_data(images_w, pseudo_labels_w,
                                                        alpha, use_cuda=True)
        inputs_w, targets_w_a, targets_w_b = map(Variable, (inputs_w,
                                                    targets_w_a, targets_w_b))
        
        targets_w_mix = lam_w * torch.eye(args.num_clusters).cuda()[targets_w_a] + (1-lam_w) * torch.eye(args.num_clusters).cuda()[targets_w_b]
        
        feats_w_mix, target_w_mix_logit = model(inputs_w, cls_only=True)
        
        if args.learn.use_mixup_ws:
        
            inputs_q, targets_q_a, targets_q_b, lam_q, mix_q_idx = mixup_data(images_q, pseudo_labels_w,
                                                            alpha, use_cuda=True)
            inputs_q, targets_q_a, targets_q_b = map(Variable, (inputs_q,
                                                        targets_q_a, targets_q_b))
            
            targets_q_mix = lam_q * torch.eye(args.num_clusters).cuda()[targets_q_a] + (1-lam_q) * torch.eye(args.num_clusters).cuda()[targets_q_b]
            
            feats_q_mix, target_q_mix_logit = model(inputs_q, cls_only=True)
        
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
        
       
        loss = (
            args.learn.alpha * loss_cls
            + args.learn.beta * loss_ins
            + args.learn.eta * loss_div
            + loss_proto
            + loss_mix
        )
        
        loss_meter.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # use slow feature to update neighbor space
        with torch.no_grad():
            feats_w, logits_w = model.momentum_model(images_w, return_feats=True)
            feats_k, logits_k = model.momentum_model(images_k, return_feats=True)

        update_labels(banks, idxs, feats_w, logits_w, labels, args)
        
        if use_wandb(args):
            # matching_label = (banks['gt'] == banks['logit'].argmax(dim=1))
            # clean_idx = banks['confidence'] > 0.5
            # noise_accuracy = (clean_idx == matching_label).float().mean()
            # loss_meter.update(noise_accuracy)

            # tn, fp, fn, tp = confusion_matrix(matching_label.cpu().numpy(), clean_idx.cpu().numpy()).ravel()

            # pre = tp/(tp+fp)
            # rec    = tp/(tp+fn)
                
            wandb_dict = {
                "loss_cls": args.learn.alpha * loss_cls.item(),
                "loss_ins": args.learn.beta * loss_ins.item(),
                "loss_div": args.learn.eta * loss_div.item(),
                "loss_mix": loss_mix.item(),
                "acc_ins": accuracy_ins.item(),
                "loss_proto": loss_proto.item(),
                "lr": optimizer.param_groups[0]['lr'],
                # "rec_ins": rec,
                # "pre_ins": pre,
            }
            
            # wandb_dict.update({'noise_acc':noise_accuracy})
            
            wandb.log(wandb_dict, commit=(i != len(train_loader) - 1))

        
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.learn.print_freq == 0:
            progress.display(i)
            
def noise_detect_cls(cluster_labels, labels, features, banks, args, temp=0.25, return_center=False):
    
    labels = labels.long()
    centers = F.normalize(cluster_labels.T.mm(features), dim=1)
    context_assigments_logits = features.mm(centers.T) / temp # sim feature with center
    context_assigments = F.softmax(context_assigments_logits, dim=1)
    # labels model argmax
    # distance_label = F.argmax(context_assigments,axis=1)[1]
    losses = - context_assigments[torch.arange(labels.size(0)), labels] # select target cluster distance
    losses = losses.cpu().numpy()[:, np.newaxis]
    losses = (losses - losses.min()) / (losses.max() - losses.min()) # normalize (min,max)
    losses = np.nan_to_num(losses)
    labels = labels.cpu().numpy()
    
    from sklearn.mixture import GaussianMixture
    confidence = np.zeros((losses.shape[0],))
    banks['gm'].fit(losses)
    pdf = banks['gm'].predict_proba(losses)
    confidence = (pdf / pdf.sum(1)[:, np.newaxis])[:, np.argmin(banks['gm'].means_)]
    confidence = torch.from_numpy(confidence).float().cuda()
    if return_center : 
        # , losses, pdf / pdf.sum(1)[:, np.newaxis]
        return confidence, losses
    else: 
        return confidence, losses

def get_center_proto(banks, use_confidence):
    if use_confidence: 
        ignore_idx = banks['confidence'] > 0.5
        centers = F.normalize(banks['probs'][ignore_idx].T.mm(banks['features'][ignore_idx]), dim=1)
    else: 
        centers = F.normalize(banks['probs'].T.mm(banks['features']), dim=1)
    return centers


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
   
def prototype_cluster(banks, use_aug_key):
    if use_aug_key:  
        f_index = banks['confidence']> 0.5
        feature_bank = banks['features'][f_index]
        probs_bank = banks['probs'][f_index]
    else:
        # f_index = banks['confidence']
        feature_bank = banks['features']
        probs_bank = banks['probs']

        # f_index = banks['confidence']> 0.5
        # feature_bank = banks['aug_features'][f_index]
        # probs_bank = banks['aug_probs'][f_index]

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

    return prototypes

@torch.no_grad()
def calculate_acc(logits, labels):
    preds = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean() * 100
    return accuracy

def cluster_loss(): 
    return ClusterLoss()

def instance_loss(logits_ins, pseudo_labels, mem_labels, logits_neg_near, contrast_type, args):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()

    # in class_aware mode, do not contrast with same-class samples
    if contrast_type == "class_aware" and pseudo_labels is not None:
        mask = torch.ones_like(logits_ins, dtype=torch.bool)
        mask[:, 1:] = pseudo_labels.reshape(-1, 1) != mem_labels  # (B, K)
        logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())
    elif contrast_type == "nearest" and pseudo_labels is not None:
        mask = torch.ones_like(logits_ins, dtype=torch.bool)        
        _, idx_near = torch.topk(logits_neg_near, k=args.learn.near, dim=-1, largest=True)
        mask[:, 1:] = torch.all(pseudo_labels.unsqueeze(1).repeat(1,args.learn.near).unsqueeze(1) != mem_labels[idx_near].unsqueeze(0), dim=2) # (B, K)
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
