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

import shot_network as network
from shot_network import Classifier
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
    get_target_optimizer, 
    train_epoch_sfda
)
def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
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
        else: 
            label_file = os.path.join(
                args.data.image_root, f"{args.data.tgt_domain}_list.txt"
            )
        dummy_dataset = ImageList(args.data.image_root, label_file)
        data_length = len(dummy_dataset)
        args.learn.queue_size = data_length
        del dummy_dataset

    #### shotbase weigth load ####
    if args.data.dataset == 'OfficeHome':
        class_num = 65
        model_path = args.data.src_domain[0].upper()
    elif args.data.dataset == 'office':
        model_path = args.data.src_domain[0].upper()
        class_num = 31
    elif args.data.dataset == 'VISDA-C':
        model_path = 'T'
        class_num = 12
    elif args.data.dataset == 'office-caltech':
        class_num = 10
        
    logging.info('Starting Shot model to train')
    if args.model_src.arch[0:3] == 'res':
        netF = network.ResBase(res_name=args.model_src.arch)
    elif args.model_src.arch[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.model_src.arch)
    
    netB = network.feat_bottleneck(type="bn", feature_dim=netF.in_features, bottleneck_dim=args.model_src.bottleneck_dim)
    netC = network.feat_classifier(type="wn", class_num=class_num, bottleneck_dim=args.model_src.bottleneck_dim)

    modelpath = args.model_tta.src_log_dir +f'/{model_path}'+'/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath, map_location="cpu"))
    modelpath = args.model_tta.src_log_dir + f'/{model_path}'+'/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath, map_location="cpu"))
    modelpath = args.model_tta.src_log_dir + f'/{model_path}'+'/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath, map_location="cpu"))
    netC.eval()
    
    for k, v in netC.named_parameters():
        v.requires_grad = True
    for k, v in netF.named_parameters():
        v.requires_grad = True
    for k, v in netC.named_parameters():
        v.requires_grad = True
    
    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.optim.lr * 1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.optim.lr * 10}]
        
    logging.info('Add SHOT optimizer parameter')
    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    for param_group in optimizer.param_groups:
        param_group['weight_decay'] = args.optim.weight_decay
        param_group['momentum'] = args.optim.momentum
        param_group['nesterov'] = args.optim.nesterov

    train_target = (args.data.src_domain != args.data.tgt_domain)
    src_model = Classifier(args.model_src, netB, netF, netC, train_target)
    momentum_model = Classifier(args.model_src, netB, netF, netC, train_target)
    
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
    for k, v in model.named_parameters():
        v.requires_grad = True
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
    # optimizer = get_target_optimizer(model, args)
    
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
        