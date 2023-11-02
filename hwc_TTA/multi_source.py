import os
import logging
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageNet
import wandb
from source import (
    get_source_optimizer,
    train_epoch,
    evaluate,


)
from classifier import Classifier
from image_list import ImageList
from utils import (
    adjust_learning_rate,
    concat_all_gather,
    get_augmentation,
    is_master,
    per_class_accuracy,
    remove_wrap_arounds,
    save_checkpoint,
    use_wandb,
    AverageMeter,
    ProgressMeter,
    get_class_dict
)
from image_list import load_image, ImageList
from torchvision import transforms
from itertools import chain
from torch.utils.data import Dataset

def multi_evaluate(val_loader, model, domain, args, wandb_commit=True):
    model.eval()

    logging.info(f"Evaluating...")
    gt_labels, all_preds, all_domain = [], [], []
    with torch.no_grad():
        iterator = tqdm(val_loader) if is_master(args) else val_loader
        for data in iterator:
            images = data[0].cuda(args.gpu, non_blocking=True)
            labels = data[1]
            src_domain = data[-1]

            logits = model(images)
            preds = logits.argmax(dim=1).cpu()

            gt_labels.append(labels)
            all_preds.append(preds)
            all_domain.append(src_domain)

    gt_labels = torch.cat(gt_labels)
    all_preds = torch.cat(all_preds)
    all_domain = torch.cat(all_domain)
    

    if args.distributed:
        gt_labels = concat_all_gather(gt_labels.cuda())
        all_preds = concat_all_gather(all_preds.cuda())
        all_domain = concat_all_gather(all_domain.cuda())

        ranks = len(val_loader.dataset) % dist.get_world_size()
        gt_labels = remove_wrap_arounds(gt_labels, ranks).cpu()
        all_preds = remove_wrap_arounds(all_preds, ranks).cpu()
        all_domain = remove_wrap_arounds(all_domain, ranks).cpu()

    accuracy = (all_preds == gt_labels).float().mean() * 100.0
    wandb_dict = {f"{domain} Acc": accuracy}
    for i in np.unique(all_domain.cpu().numpy()):
        src_domain_idx = torch.where(all_domain==i)
        src_domain_acc  = (gt_labels[src_domain_idx] == all_preds[src_domain_idx]).float().mean() * 100.0
        src_domain_name = get_class_dict(args)[1][i]
        wandb_dict.update({f"{src_domain_name} Acc": src_domain_acc})
        logging.info(f"{src_domain_name}_Accuracy: {src_domain_acc:.2f}")    

    logging.info(f"Accuracy: {accuracy:.2f}")
    if args.data.dataset == "VISDA-C":
        acc_per_class = per_class_accuracy(
            y_true=gt_labels.numpy(), y_pred=all_preds.numpy()
        )
        wandb_dict[f"{domain} Avg"] = acc_per_class.mean()
        wandb_dict[f"{domain} Per-class"] = acc_per_class

    if use_wandb(args):
        wandb.log(wandb_dict, commit=wandb_commit)

    return accuracy

class MultiImageList(Dataset):
    def __init__(
        self,
        image_root: str,
        label_file: str,
        domain=None,
        transform=None,
        pseudo_item_list=None,
        dataset_name = None,
        args=None
    ):
        self.image_root = image_root
        self._label_file = label_file
        self.transform = transform
        

        self.target_dict = get_class_dict(args)[0]
        assert (
            label_file or pseudo_item_list
        ), f"Must provide either label file or pseudo labels."

        item_lists = []
        src_names = []
        for i, src in zip(label_file, domain): 
            item_list, src_name = self.build_index(i, src) if i else pseudo_item_list
            item_lists.append(item_list)
            src_names.append(src_name)
        

        self.item_list = list(chain.from_iterable(item_lists))
        self.src_names = list(chain.from_iterable(src_names))

        # self.item_list = 
        self.resize_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    def build_index(self, label_file, src):
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
            target_list.append(self.target_dict[src])
        
        logging.info(f"Start concatation on {src}... length:{len(item_list)}")
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

        return img, label, idx, self.src_names[idx]

    def __len__(self):
        return len(self.item_list)


def load_labels(args, domain, dataset_idx): 
    if args.data.dataset.lower() == 'pacs': 
        label_file = os.path.join(args.data.image_root, f"{domain}_test_kfold.txt")
    elif args.data.dataset.lower() == 'domainnet': 
        label_file = os.path.join(args.data.image_root, f"{domain}_concat.txt")
    else: 
        label_file = os.path.join(args.data.image_root, f"{domain}_list.txt")
    
    return label_file

def train_source_domain(args):
    logging.info(f"Start source training on {args.data.src_domain}...")
    
    model = Classifier(args.model_src, train_target=False).to("cuda")
    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
    logging.info(f"1 - Created source model")

    # transforms
    train_transform = get_augmentation("plain")
    val_transform = get_augmentation("test")

    label_files = []
    for idx, src_domain in enumerate(args.data.src_domain):
        logging.info(f"Start concatation on {src_domain}...")
        label_files.append(load_labels(args,src_domain, dataset_idx=idx))
    
    val_dataset = MultiImageList(
        image_root=args.data.image_root,
        label_file=label_files,
        domain=args.data.src_domain,
        transform=val_transform,
        dataset_name=args.data.dataset,
        args=args
    )

    train_dataset = MultiImageList(
        image_root=args.data.image_root,
        label_file=label_files,  # uses pseudo labels
        domain=args.data.src_domain,
        transform=train_transform,
        dataset_name=args.data.dataset,
        args=args
        # pseudo_item_list=pseudo_item_list,
    )

    assert len(train_dataset) == len(val_dataset)

    # split the dataset with indices
    indices = np.random.permutation(len(train_dataset))
    num_train = int(len(train_dataset) * args.data.train_ratio)
    train_dataset = Subset(train_dataset, indices[:num_train])
    val_dataset = Subset(val_dataset, indices[num_train:])

    logging.info(
        f"Loaded {len(train_dataset)} samples for training "
        + f"and {len(val_dataset)} samples for validation",
    )

    # data loaders
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.data.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        num_workers=args.data.workers,
    )
    val_sampler = DistributedSampler(val_dataset) if args.distributed else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.data.batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=args.data.workers,
    )
    logging.info(f"2 - Created data loaders")

    optimizer = get_source_optimizer(model, args)
    args.learn.full_progress = args.learn.epochs * len(train_loader)
    logging.info(f"3 - Created optimizer")

    logging.info(f"Start training...")
    best_acc = 0.0
    domains_name = '_'.join(args.data.src_domain)

    for epoch in range(args.learn.start_epoch, args.learn.epochs):
        logging.info(f"Start eval...")
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_epoch(train_loader, model, optimizer, epoch, args)

        accuracy = multi_evaluate(val_loader, model, domain=domains_name, args=args)
        # evaluate
        if accuracy > best_acc and is_master(args):
            best_acc = accuracy
            filename = f"best_{domains_name}_{args.seed}.pth.tar"
            save_path = os.path.join(args.log_dir, filename)
            save_checkpoint(model, optimizer, epoch, save_path=save_path)

    # evaluate on target before any adaptation
    for t, src_domain in enumerate(args.data.target_domains):
        logging.info(f"testing...{src_domain}")
        if src_domain in args.data.src_domain:
            print(src_domain, args.data.src_domain)
            continue
        if args.data.dataset.lower() == 'pacs': 
            label_file = os.path.join(args.data.image_root, f"{src_domain}_test_kfold.txt")
        elif args.data.dataset.lower() == 'domainnet': 
            label_file = os.path.join(args.data.image_root, f"{src_domain}_concat.txt")
        else: 
            label_file = os.path.join(args.data.image_root, f"{src_domain}_list.txt")

        src_dataset = ImageList(args.data.image_root, label_file, val_transform)
        sampler = DistributedSampler(src_dataset) if args.distributed else None
        src_loader = DataLoader(
            src_dataset,
            batch_size=args.data.batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=args.data.workers,
        )

        logging.info(f"Evaluate {args.data.src_domain} model on {src_domain}")
        evaluate(
            src_loader,
            model,
            domain=f"{args.data.src_domain}-{src_domain}",
            args=args,
            wandb_commit=(t == len(args.data.target_domains) - 1),
        )