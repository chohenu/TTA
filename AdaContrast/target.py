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
import numpy as np
import wandb

from classifier import Classifier
from image_list import ImageList
from moco.builder import AdaMoCo
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
)

from losses import ClusterLoss

# TODO : install PyTorchGaussianMixture
# try: 
#     from torch_clustering import PyTorchGaussianMixture
# except: 
#     !pip install -e /opt/tta/AdaContrast/clustering/torch_clustering/
#     from torch_clustering import PyTorchGaussianMixture

from sklearn.mixture import GaussianMixture  ## numpy version

@torch.no_grad()
def eval_and_label_dataset(dataloader, model, banks, args):
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
        feats, logits_cls = model(imgs, cls_only=True)

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

    probs = F.softmax(logits, dim=1)
    rand_idxs = torch.randperm(len(features)).cuda()
    banks = {
        "features": features[rand_idxs][: args.learn.queue_size],
        "probs": probs[rand_idxs][: args.learn.queue_size],
        "ptr": 0,
    }
    
    confidence, context_assignments, centers = None, None, None
    # refine predicted labels
    if args.learn.do_noise_detect:
        logging.info(
            "Do Noise Detection"
        )
        noise_labels = pred_labels
        is_clean = gt_labels.cpu().numpy() == noise_labels.cpu().numpy()
        clean_ratio = np.average(is_clean)
        
        confidence, context_assignments, centers = noise_detect(cluster_labels, noise_labels, 
                                                                features, args)
        estimated_noise_ratio = (confidence > 0.5).float().mean().item()
        noise_accuracy = ((confidence > 0.5) == (gt_labels == noise_labels)).float().mean()
        from sklearn.metrics import roc_auc_score
        context_noise_auc = roc_auc_score(is_clean, confidence.cpu().numpy())
        logging.info(f"noise_accuracy: {noise_accuracy}")
        logging.info(f"noise_ratio: {1-clean_ratio}")
        logging.info(f"roc_auc_score: {context_noise_auc}")
    banks["confidence"] = confidence[rand_idxs][: args.learn.queue_size]
    banks["context_assignments"] = context_assignments[rand_idxs][: args.learn.queue_size]
    banks["centers"] = centers
    pred_labels, _, acc  = refine_predictions(
        features, probs, banks, args=args, gt_labels=gt_labels
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
        img_path, _, img_file = dataloader.dataset.item_list[idx]
        pseudo_item_list.append((img_path, int(pred_label), img_file))
    logging.info(f"Collected {len(pseudo_item_list)} pseudo labels.")

    if use_wandb(args):
        wandb.log(wandb_dict)

    return pseudo_item_list, banks, confidence, context_assignments, centers


@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank, confidence_bank,
                            context_assignments_bank, centers_bank,args):
    pred_probs = []
    for feats in features.split(64):
        distances = get_distances(feats, features_bank, args.learn.dist_type)
        _, idxs = distances.sort()
        idxs = idxs[:, : args.learn.num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :]
        confidences = confidence_bank[idxs].unsqueeze(2).repeat(1,1,probs.size(2))
        probs = (confidences * probs).mean(1)
        pred_probs.append(probs)
    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)

    return pred_labels, pred_probs


@torch.no_grad()
def GMM_clustering(features, features_bank, probs_bank, args):
    
    array_feature_bank = features_bank.cpu().numpy()
    n_features = features.size(1)
    n_components = probs_bank.size(1)
    logging.info(f"Making Gaussian mixture model")
    model = GaussianMixture(n_components=n_components, covariance_type="diag", random_state=0)
    model.fit(array_feature_bank)
    y_p = model.predict_proba(array_feature_bank)
    return y_p

def noise_detect(cluster_labels, labels, features, args, temp=0.25):
    labels = labels.long()
    centers = F.normalize(cluster_labels.T.mm(features), dim=1)
    context_assigments_logits = features.mm(centers.T) / temp
    context_assigments = F.softmax(context_assigments_logits, dim=1)
    losses = - context_assigments[torch.arange(labels.size(0)), labels]
    losses = losses.cpu().numpy()[:, np.newaxis]
    losses = (losses - losses.min()) / (losses.max() - losses.min())
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
    return confidence, context_assigments, centers

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
    end = start + len(idxs)
    idxs_replace = torch.arange(start, end).cuda() % len(banks["features"])
    banks["features"][idxs_replace, :] = features
    banks["probs"][idxs_replace, :] = probs
    banks["ptr"] = end % len(banks["features"])


@torch.no_grad()
def refine_predictions(
    features,
    probs,
    banks,
    args,
    gt_labels=None,
):
    if args.learn.refine_method == "nearest_neighbors":
        feature_bank = banks["features"]
        probs_bank = banks["probs"]
        confidence_bank = banks["confidence"]
        context_assignments_bank = banks["context_assignments"]
        centers_bank = banks["centers"]
        
        pred_labels, probs = soft_k_nearest_neighbors(
            features, feature_bank, probs_bank, confidence_bank,
            context_assignments_bank, centers_bank, args
        )
    elif args.learn.refine_method == "GMM":
        feature_bank = banks["features"]
        probs_bank = banks["probs"]
        ## TODO : Calculate distance using GMM model 
        ## using Attributes function (ex, predict(features) or predict_proba(features)  
        ## Link : https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
        gm = GMM_clustering(
            features, feature_bank, probs_bank, 
            args)        
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
    model = AdaMoCo(
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
    pseudo_item_list, banks, confidence, context_assignments, centers = eval_and_label_dataset(
        val_loader, model, banks=None, args=args
    )
    logging.info("2 - Computed initial pseudo labels")

    # Training data
    train_transform = get_augmentation_versions(args)
    train_dataset = ImageList(
        image_root=args.data.image_root,
        label_file=None,  # uses pseudo labels
        transform=train_transform,
        pseudo_item_list=pseudo_item_list,
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
        if args.learn.do_noise_detect:
            train_epoch_twin(train_loader, model, banks, confidence, 
                             context_assignments, centers, optimizer, epoch, args)
        else:
            train_epoch(train_loader, model, banks, optimizer, epoch, args)
        
        

        eval_and_label_dataset(val_loader, model, banks, args)

    if is_master(args):
        filename = f"checkpoint_{epoch:04d}_{args.data.src_domain}-{args.data.tgt_domain}-{args.sub_memo}_{args.seed}.pth.tar"
        save_path = os.path.join(args.log_dir, filename)
        save_checkpoint(model, optimizer, epoch, save_path=save_path)
        logging.info(f"Saved checkpoint {save_path}")
        
        
def train_epoch_twin(train_loader, model, banks, confidence, 
                     context_assignments, centers, optimizer, 
                     epoch, args):
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
        images, _, idxs = data
        idxs = idxs.to("cuda")
        images_w, images_q, images_k = (
            images[0].to("cuda"),
            images[1].to("cuda"),
            images[2].to("cuda"),
        )
        clear_confidence = confidence[idxs]
        
        # per-step scheduler
        step = i + epoch * len(train_loader)
        adjust_learning_rate(optimizer, step, args)
        
        # weak aug model output
        feats_w, logits_w = model(images_w, cls_only=True)
        with torch.no_grad():
            probs_w = F.softmax(logits_w, dim=1)
            pseudo_labels_w, probs_w, _ = refine_predictions(
                feats_w, probs_w, banks, args=args
            )

        # strong aug model output
        feats_q, logits_q, logits_ins, feats_k, logits_k = model(images_q, images_k)
        
        
        # Calculate reliable degree of pseudo labels
        if args.learn.use_ce_weight:
            with torch.no_grad():
                CE_weight = calculate_reliability(centers, clear_confidence, feats_w, feats_q, feats_k)
        else:
            CE_weight = 1.
        
        # update key features and corresponding pseudo labels
        model.update_memory(feats_k, pseudo_labels_w)


        # add cluster contrastive los s
        if args.learn.add_cluster_loss:
            clusterloss = cluster_loss()
            if args.learn.use_momory_bank: 
                bs = len(logits_q)
                ms = model.mem_feat.size(1)
                k = ms // bs
                feats_m = model.mem_feat[:, :bs*k].permute(1, 0)
                logtis_m = model.src_model.fc(feats_m)
                
                #repeat logit q
                logits_q_  = logits_q[None, :, :].repeat(k, 1, 1).flatten(0, 1)
                
                logits_cluster = torch.cat([F.softmax(logits_q_, dim=1), F.softmax(logtis_m, dim=1)], dim=0)
            else: 
                logits_cluster = torch.cat([F.softmax(logits_q, dim=1), F.softmax(logits_k, dim=1)], dim=0)
            loss_cluster = clusterloss(logits_cluster)

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
            args.learn.alpha * loss_cls
            + args.learn.beta * loss_ins
            + args.learn.eta * loss_div
        )
        loss_meter.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # use slow feature to update neighbor space
        with torch.no_grad():
            feats_w, logits_w = model.momentum_model(images_w, return_feats=True)

        update_labels(banks, idxs, feats_w, logits_w, args)

        if use_wandb(args):
            wandb_dict = {
                "loss_cls": args.learn.alpha * loss_cls.item(),
                "loss_ins": args.learn.beta * loss_ins.item(),
                "loss_div": args.learn.eta * loss_div.item(),
                "acc_ins": accuracy_ins.item(),
            }

            wandb.log(wandb_dict, commit=(i != len(train_loader) - 1))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.learn.print_freq == 0:
            progress.display(i)
            
def train_epoch(train_loader, model, banks, optimizer, epoch, args):
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

        feats_w, logits_w = model(images_w, cls_only=True)
        with torch.no_grad():
            probs_w = F.softmax(logits_w, dim=1)
            pseudo_labels_w, probs_w, _ = refine_predictions(
                feats_w, probs_w, banks, args=args
            )

        _, logits_q, logits_ins, keys = model(images_q, images_k)
        # update key features and corresponding pseudo labels
        model.update_memory(keys, pseudo_labels_w)

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
            logits_w, logits_q, pseudo_labels_w, args
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
        )
        loss_meter.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # use slow feature to update neighbor space
        with torch.no_grad():
            feats_w, logits_w = model.momentum_model(images_w, return_feats=True)

        update_labels(banks, idxs, feats_w, logits_w, args)

        if use_wandb(args):
            wandb_dict = {
                "loss_cls": args.learn.alpha * loss_cls.item(),
                "loss_ins": args.learn.beta * loss_ins.item(),
                "loss_div": args.learn.eta * loss_div.item(),
                "acc_ins": accuracy_ins.item(),
            }

            wandb.log(wandb_dict, commit=(i != len(train_loader) - 1))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.learn.print_freq == 0:
            progress.display(i)
            
def entropy(p, axis=1):
    return -torch.sum(p * torch.log2(p+1e-5), dim=axis)

def norm(x):
    return (x - x.min()) / (x.max()-x.min())

@torch.no_grad()
def calculate_reliability(centers, confidence, feats_w, feats_q, feats_k):
    sim_w = norm(feats_w.mm(centers.T))
    sim_q = norm(feats_q.mm(centers.T))
    sim_k = norm(feats_k.mm(centers.T))
    sim_avg = F.softmax((sim_w + sim_q + sim_k)/3, dim=1)
    max_entropy = torch.log2(torch.tensor(centers.size(0)))
    w = entropy(sim_avg)
    
    w = w / max_entropy
    w = w * confidence
    w = torch.exp(-w)
    
    return w
    

@torch.no_grad()
def calculate_acc(logits, labels):
    preds = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean() * 100
    return accuracy

def cluster_loss(): 
    return ClusterLoss()

def instance_loss(logits_ins, pseudo_labels, mem_labels, contrast_type):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()

    # in class_aware mode, do not contrast with same-class samples
    if contrast_type == "class_aware" and pseudo_labels is not None:
        mask = torch.ones_like(logits_ins, dtype=torch.bool)
        mask[:, 1:] = pseudo_labels.reshape(-1, 1) != mem_labels  # (B, K)
        logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())

    loss = F.cross_entropy(logits_ins, labels_ins)

    accuracy = calculate_acc(logits_ins, labels_ins)

    return loss, accuracy


def classification_loss(logits_w, logits_s, target_labels, CE_weight, args):
    if args.learn.ce_sup_type == "weak_weak":
        loss_cls = (CE_weight * cross_entropy_loss(logits_w, target_labels, args)).mean()
        accuracy = calculate_acc(logits_w, target_labels)
    elif args.learn.ce_sup_type == "weak_strong":
        loss_cls = (CE_weight * cross_entropy_loss(logits_s, target_labels, args)).mean()
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
