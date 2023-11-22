import logging
import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from classifier import Classifier
from image_list import ImageList
from moco.builder import CNA_MoCo
from utils import (
    is_master,
    save_checkpoint,
    CustomDistributedDataParallel,
)

from sklearn.mixture import GaussianMixture  ## numpy version
from target import (
    get_augmentation_versions, 
    eval_and_label_dataset, 
    get_target_optimizer, 
    train_epoch_sfda, 
)
from omegaconf import ListConfig
import wandb
from torchvision import transforms
from utils import TARGET_DICT
from image_list import load_image, ImageList
from torch.utils.data import Dataset
from itertools import chain

class MultiImageList(Dataset):
    def __init__(
        self,
        image_root: str,
        label_file: str,
        tgt_domain=None,
        transform=None,
        pseudo_item_list=None,
        dataset_name = None
    ):
        self.image_root = image_root
        self._label_file = label_file
        self.transform = transform
        self.target_dict = TARGET_DICT[dataset_name]

        assert (
            label_file or pseudo_item_list
        ), f"Must provide either label file or pseudo labels."

        item_lists = []
        tgt_names = []
        for i, tgt in zip(label_file, tgt_domain): 
            item_list, tgt_name = self.build_index(i, tgt) if i else pseudo_item_list
            item_lists.append(item_list)
            tgt_names.append(tgt_name)

        self.item_list = list(chain.from_iterable(item_lists))
        self.tgt_names = list(chain.from_iterable(tgt_names))

        # self.item_list = 
        self.resize_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    def build_index(self, label_file, tgt):
        """Build a list of <image path, class label> items.

        Args:
            label_file: path to the domain-net label file

        Returns:
            item_list: a list of <image path, class label> items.
        """
        # read in items; each item takes one line
        with open(label_file, "r") as fd:
            lines = fd.readlines()
        lines = [line.strip() for line in lines if line]

        item_list = []
        target_list = []
        for item in lines:
            img_file, label = item.split()
            img_path = os.path.join(self.image_root, img_file)
            label = int(label)
            item_list.append((img_path, label, img_file))
            target_list.append(self.target_dict[tgt])

        return item_list, target_list

    def __getitem__(self, idx):
        """Retrieve data for one item.

        Args:
            idx: index of the dataset item.
        Returns:
            img: <C, H, W> tensor of an image
            label: int or <C, > tensor, the corresponding class label. when using raw label
                file return int, when using pseudo label list return <C, > tensor.
        """
        img_path, label, _ = self.item_list[idx]
        img = load_image(img_path)
        if self.transform:
            img = self.transform(img)
            for crop_idx, crops in enumerate(img): 
                if not isinstance(crops, torch.Tensor): img[crop_idx] = torch.stack([self.resize_transform(crop) for crop in crops])

        return img, label, idx, self.tgt_names[idx]

    def __len__(self):
        return len(self.item_list)

def load_labels(args, tgt_domain, dataset_idx): 
    if args.learn.queue_size == -1:
        if args.data.dataset.lower() == 'pacs': 
            label_file = os.path.join(
                args.data.image_root, f"{tgt_domain}_test_kfold.txt"
            )
        else: 
            label_file = os.path.join(
                args.data.image_root, f"{tgt_domain}_list.txt"
            )
        dummy_dataset = ImageList(args.data.image_root, label_file)
        data_length = len(dummy_dataset)
        args.learn.queue_size = data_length
        del dummy_dataset
    
    if args.data.dataset.lower() == 'pacs': 
        label_file = os.path.join(args.data.image_root, f"{tgt_domain}_test_kfold.txt")
    else: 
        label_file = os.path.join(args.data.image_root, f"{tgt_domain}_list.txt")
    
    return label_file

# return val_dataset, train_dataset, len(val_dataset)

def train_target_domain(args):
    
    logging.info(
        f"Start target training on {args.data.src_domain}-{args.data.tgt_domain}..."
    )
    val_transform = get_augmentation_versions(args, train=False)
    train_transform = get_augmentation_versions(args, train=True)
    if isinstance(args.data.tgt_domain, ListConfig): 
        total_length = 0 
        label_files = []
        for idx, tgt_domain in enumerate(args.data.tgt_domain):
            logging.info(
            f"Start concatation target on {args.data.src_domain}-{tgt_domain}..."
            )
            label_files.append(load_labels(args,tgt_domain, dataset_idx=idx))
        
        val_dataset = MultiImageList(
            image_root=args.data.image_root,
            label_file=label_files,
            tgt_domain=args.data.tgt_domain,
            transform=val_transform,
            dataset_name=args.data.dataset,
        )

        train_dataset = MultiImageList(
            image_root=args.data.image_root,
            label_file=label_files,  # uses pseudo labels
            tgt_domain=args.data.tgt_domain,
            transform=train_transform,
            dataset_name=args.data.dataset,
            # pseudo_item_list=pseudo_item_list,
        )
        
        data_length = len(val_dataset)
        args.learn.queue_size = data_length

    else: 
        label_file = load_labels(args, args.data.tgt_domain, dataset_idx=0)
        val_dataset = ImageList(
                image_root=args.data.image_root,
                label_file=label_file,
                transform=val_transform,
            )    
        train_dataset = ImageList(
            image_root=args.data.image_root,
            label_file=label_file,  # uses pseudo labels
            transform=train_transform,
            # pseudo_item_list=pseudo_item_list,
        )

    checkpoint_path = os.path.join(
        args.model_tta.src_log_dir,
        f"best_{args.data.src_domain}_{args.seed}.pth.tar",
    )
    src_model = Classifier(args.model_src, False, checkpoint_path)
    momentum_model = Classifier(args.model_src, False, checkpoint_path)

    model = CNA_MoCo(
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
        gm = GaussianMixture(n_components=2, random_state=0)

    pseudo_item_list, banks = eval_and_label_dataset(
        val_loader, model, banks=None, epoch=-1, gm=gm, args=args
    )
    logging.info("2 - Computed initial pseudo labels")
    
    args.num_clusters = model.src_model.num_classes
    # Training data

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
        