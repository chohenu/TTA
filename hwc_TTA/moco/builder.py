# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import concat_all_gather
from torch.autograd import Variable

import random
from sklearn.mixture import GaussianMixture
import torch
import torch.nn.functional as F

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, labels, ignore_idx):
        # CCE
        ce = self.cross_entropy(pred, labels)[ignore_idx].mean()

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred[ignore_idx] * torch.log(label_one_hot[ignore_idx]), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

class AdaMoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a memory bank
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
        self,
        src_model,
        momentum_model,
        K=16384,
        m=0.999,
        T_moco=0.07,
        checkpoint_path=None,
    ):
        """
        dim: feature dimension (default: 128)
        K: buffer size; number of keys
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(AdaMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T_moco = T_moco
        self.queue_ptr = 0

        # create the encoders
        self.src_model = src_model
        self.momentum_model = momentum_model

        # create the fc heads
        feature_dim = src_model.output_dim

        # freeze key model
        self.momentum_model.requires_grad_(False)

        # create the memory bank
        self.register_buffer("mem_feat", torch.randn(feature_dim, K))
        self.register_buffer(
            "mem_labels", torch.randint(0, src_model.num_classes, (K,))
        )
        self.register_buffer(
            "mem_gt", torch.randint(0, src_model.num_classes, (K,))
        )

        self.mem_feat = F.normalize(self.mem_feat, dim=0)

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        for name, param in checkpoint["state_dict"].items():
            # get rid of 'module.' prefix brought by DDP
            name = name[len("module.") :] if name.startswith("module.") else name
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        )

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        # encoder_q -> encoder_k
        for param_q, param_k in zip(
            self.src_model.parameters(), self.momentum_model.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def return_membank(self): 

        return {'mem_feature': self.mem_feat.cpu().numpy(),
                'mem_pseudo_labels': self.mem_labels.cpu().numpy()}

    @torch.no_grad()
    def update_memory(self, keys, pseudo_labels):
        """
        Update features and corresponding pseudo labels
        """
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        pseudo_labels = concat_all_gather(pseudo_labels)
        gt_labels = concat_all_gather(gt_labels.to('cuda'))
        

        start = self.queue_ptr
        end = start + len(keys)
        idxs_replace = torch.arange(start, end).cuda() % self.K
        self.mem_feat[:, idxs_replace] = keys.T
        self.mem_labels[idxs_replace] = pseudo_labels
        self.queue_ptr = end % self.K

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k=None, cls_only=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            feats_q: <B, D> query image features before normalization
            logits_q: <B, C> logits for class prediction from queries
            logits_ins: <B, K> logits for instance prediction
            k: <B, D> contrastive keys
        """

        # compute query features
        feats_q, logits_q = self.src_model(im_q, return_feats=True)

        if cls_only:
            return feats_q, logits_q

        q = F.normalize(feats_q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k, logits_k = self.momentum_model(im_k, return_feats=True)
            k = F.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.mem_feat.clone().detach()])

        # logits: Nx(1+K)
        logits_ins = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits_ins /= self.T_moco

        # dequeue and enqueue will happen outside
        return feats_q, logits_q, logits_ins, k, logits_k
    
class hwc_MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a memory bank
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
        self,
        src_model,
        momentum_model,
        K=16384,
        m=0.999,
        T_moco=0.07,
        dataset_legth=10000,
        checkpoint_path=None,
        args=None
    ):
        """
        dim: feature dimension (default: 128)
        K: buffer size; number of keys
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(hwc_MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T_moco = T_moco
        self.queue_ptr = 0

        # create the encoders
        self.src_model = src_model
        self.momentum_model = momentum_model

        # create the fc heads
        if args.data.dataset.lower() == 'cifar10': 
            feature_dim = src_model.fc.in_features
            self.num_classes = 10
        else: 
            feature_dim = src_model.output_dim
            self.num_classes = src_model.num_classes

        # freeze key model
        self.momentum_model.requires_grad_(False)

        # create the memory bank
        self.register_buffer("mem_feat", torch.randn(feature_dim, K))
        self.register_buffer(
            "mem_labels", torch.randint(0, self.num_classes, (K,))
        )

        self.register_buffer(
            "mem_probs", torch.rand(K, self.num_classes)
        )

        self.register_buffer(
            "mem_index", torch.randint(0, dataset_legth, (K,))
        )

        # self.gm = GaussianMixture(n_components=2, random_state=0)

        self.mem_feat = F.normalize(self.mem_feat, dim=0)

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

        self.args = args
        self.confidence = None 
        
    def find_confidence(self, banks): 
        origin_idx = torch.where(self.mem_index.reshape(-1,1)==banks['index'])[1]
        self.confidence = banks['confidence'][origin_idx]

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        for name, param in checkpoint["state_dict"].items():
            # get rid of 'module.' prefix brought by DDP
            name = name[len("module.") :] if name.startswith("module.") else name
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        )

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        # encoder_q -> encoder_k
        for param_q, param_k in zip(
            self.src_model.parameters(), self.momentum_model.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def return_membank(self): 

        return {'mem_feature': self.mem_feat.cpu().numpy(),
                'mem_pseudo_labels': self.mem_labels.cpu().numpy()}

    @torch.no_grad()
    def update_memory(self, keys, pseudo_labels, probs, index):
        """
        Update features and corresponding pseudo labels
        """
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        pseudo_labels = concat_all_gather(pseudo_labels)
        probs = concat_all_gather(probs)
        index = concat_all_gather(index)

        start = self.queue_ptr
        end = start + len(keys)
        idxs_replace = torch.arange(start, end).cuda() % self.K
        self.mem_feat[:, idxs_replace] = keys.T
        self.mem_labels[idxs_replace] = pseudo_labels
        self.queue_ptr = end % self.K
        self.mem_probs[idxs_replace, :] = probs
        self.mem_index[idxs_replace] = index

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def get_cluster_prob(self, embeddings, cluster_centers):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / 1))
        power = float(1 + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def forward(self, im_q, banks, idxs, im_k=None, pseudo_labels_w=None, epoch=None, cls_only=False, prototypes_q=None, prototypes_k=None, ignore_idx=None, args=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            feats_q: <B, D> query image features before normalization
            logits_q: <B, C> logits for class prediction from queries
            logits_ins: <B, K> logits for instance prediction
            k: <B, D> contrastive keys
        """

        # compute query features
        feats_q, logits_q = self.src_model(im_q, return_feats=True)

        if cls_only:
            return feats_q, logits_q

        q = F.normalize(feats_q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k, logits_k = self.momentum_model(im_k, return_feats=True)
            k = F.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            
        # nn
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        l_neg = torch.einsum("nc,ck->nk", [q, self.mem_feat.clone().detach()])
        # negative_nearest logits: MxM
        l_neg_near = torch.einsum("nc,ck->nk", [self.mem_feat.T.clone().detach(), self.mem_feat.clone().detach()])

        # logits: Nx(1+K)
        logits_ins = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits_ins /= self.T_moco

        if args.learn.use_proto_loss_v2: 
            psuedo_label = torch.argmax(logits_q, dim=1) # 64 = [0~ 12]
            norm_proto_q = F.normalize(prototypes_q, dim=1) # 12,256
            select_pos = norm_proto_q[psuedo_label] # 64,256
            l_pos_proto = torch.einsum('nc,nc->n', [q, select_pos]).unsqueeze(-1) # post pair 64,1
            # l_neg_proto = torch.einsum("nc,ck->nk", [select_pos, self.mem_feat.clone().detach()])
            l_neg_proto = torch.mm(q, select_pos.T) # neg pair (64,64)
            proto_logits_ins = torch.cat([l_pos_proto, l_neg_proto], dim=1)
            # apply temperature
            proto_logits_ins /= self.T_moco

            # labels: positive key indicators
            labels_ins = torch.zeros(proto_logits_ins.shape[0], dtype=torch.long).cuda()
            mask = torch.ones_like(proto_logits_ins, dtype=torch.bool)
            # mask[:, 1:] = psuedo_label.cuda().reshape(-1, 1) != self.mem_labels  # (B, K)
            mask[:, 1:] = psuedo_label.cuda().reshape(-1, 1) != psuedo_label  # (B, K)

            proto_logits_ins = torch.where(mask, proto_logits_ins, torch.tensor([float("-inf")]).cuda())
            loss_proto = F.cross_entropy(proto_logits_ins, labels_ins)
        else: 
            loss_proto = None
        
        return feats_q, logits_q, logits_ins, k, logits_k, l_neg_near, loss_proto
