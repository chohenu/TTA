# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import concat_all_gather
from torch.autograd import Variable


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
                'mem_pseudo_labels': self.mem_labels.cpu().numpy(),
                'mem_gt':self.mem_gt.cpu().numpy()}

    @torch.no_grad()
    def update_memory(self, keys, pseudo_labels, gt_labels):
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
        self.mem_gt[idxs_replace] = gt_labels
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
        checkpoint_path=None,
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
                'mem_pseudo_labels': self.mem_labels.cpu().numpy(),
                'mem_gt':self.mem_gt.cpu().numpy()}

    @torch.no_grad()
    def update_memory(self, keys, pseudo_labels, gt_labels):
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
        self.mem_gt[idxs_replace] = gt_labels
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

    def forward(self, im_q, banks, idxs, im_k=None, pseudo_labels_w=None, epoch=None, cls_only=False, prototypes=None):
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
            
        with torch.no_grad():
            softmax_out = torch.nn.Softmax(dim=1)(logits_q)

            origin_idx = torch.where(idxs.reshape(-1,1)==banks['index'])[1]
            banks['norm_features'][origin_idx] = q

            similarity = q @ banks['norm_features'].T
            _, idx_near = torch.topk(similarity, dim=-1, largest=True, k=5 + 1)
            idx_near = idx_near[:, 1:]  # batch x K
            feat_near = banks['norm_features'][idx_near]  # batch x K x F            
            # if epoch < 3:
            #     sim_ = feat_near @ banks['norm_features'].T
            #     _, idx_near_ = torch.topk(sim_, dim=-1, largest=True, k = 5 + 1)
            #     idx_near_ = idx_near_[:, :, 1:]
            #     bank_labels = torch.argmax(banks["probs"], dim=1)
            #     mask = pseudo_labels_w.unsqueeze(1).unsqueeze(1).repeat(1,5,5)==bank_labels[idx_near_]
            #     mask= mask.sum(dim=-1) > 0
            #     feat_near = feat_near*mask.unsqueeze(-1)
            
            # curriculum nearest neighbor based epoch
            # sim_ = feat_near @ banks['norm_features'].T
            # _, idx_near_ = torch.topk(sim_, dim=-1, largest=True, k = 5 + 1)
            # idx_near_ = idx_near_[:, :, 1:]
            # bank_labels = torch.argmax(banks["probs"], dim=1)
            # mask = pseudo_labels_w.unsqueeze(1).unsqueeze(1).repeat(1,5,5)==bank_labels[idx_near_]
            # mask= mask.sum(dim=-1) > (4-epoch)
            # feat_near = feat_near*mask.unsqueeze(-1)
        # nn
        q_ = q.unsqueeze(1).expand(-1, 5, -1) # batch x K x F 
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nkc,nkc->n", [q_, feat_near]).unsqueeze(-1)
        # l_pos = torch.einsum("nkc,nkc->nk", [q_, feat_near]).unsqueeze(-1)
        # l_pos = l_pos.mean(axis=1)
        # l_pos = l_pos / q_.shape[1]
        # adacontrast
        # l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.mem_feat.clone().detach()])
        # negative_nearest logits: MxM
        l_neg_near = torch.einsum("nc,ck->nk", [self.mem_feat.T.clone().detach(), self.mem_feat.clone().detach()])

        # logits: Nx(1+K)
        logits_ins = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits_ins /= self.T_moco

        # prototype wise contrastive learning
        with torch.no_grad():
            proto_sim = q @ F.normalize(prototypes, dim=1).T ## (B x feature)x (Features class)= B, class
            proto_label = torch.argmax(proto_sim, axis=1) 
            proto_loss = F.cross_entropy(proto_sim, proto_label) 
            # + F.cross_entropy(proto_sim, torch.argmax(softmax_out,1))


        # dequeue and enqueue will happen outside
        return feats_q, logits_q, logits_ins, k, logits_k, l_neg_near, proto_loss

class AdaMixCo(nn.Module):
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
        super(AdaMixCo, self).__init__()

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
                'mem_pseudo_labels': self.mem_labels.cpu().numpy(),
                'mem_gt':self.mem_gt.cpu().numpy()}

    @torch.no_grad()
    def update_memory(self, keys, pseudo_labels, gt_labels):
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
        self.mem_gt[idxs_replace] = gt_labels
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
    
    @torch.no_grad()
    def img_mixer(self, x, use_cuda=True):
        
        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        im_q1, im_q2 = x, x[index]

        # each image get different lambda
        lam = torch.from_numpy(np.random.uniform(0, 1, size=(batch_size,1,1,1))).float().to(x.device)
        imgs_mix = lam * im_q1 + (1-lam) * im_q2
        lbls_mix = torch.cat((torch.diag(lam.squeeze()), torch.diag((1-lam).squeeze())), dim=1)

        return imgs_mix, lbls_mix, index
    
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

        if cls_only:
            # compute query features
            feats_q, logits_q = self.src_model(im_q, return_feats=True)
            q = F.normalize(feats_q, dim=1)
            return feats_q, logits_q

        else: 
            imgs_mix, lbls_mix, index = self.img_mixer(im_q)
            imgs_mix, lbls_mix, index = map(Variable, (imgs_mix, lbls_mix, index))

            # compute query features
            feats_q, logits_q  = self.src_model(im_q, return_feats=True)
            feats_mix_q, logits_mix_q = self.src_model(imgs_mix, return_feats=True)

        q = F.normalize(feats_q, dim=1)
        q_mix = F.normalize(feats_mix_q, dim=1)
        
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

        # mixed logits: N x 2N
        logits_mix_pos = torch.mm(q_mix, torch.cat((feats_q, feats_q[index]), dim=0).T) 
        # mixed negative logits: N x K
        logits_mix_neg = torch.mm(q_mix, self.mem_feat.clone().detach())
        logits_mix = torch.cat([logits_mix_pos, logits_mix_neg], dim=1) # N x (2N+K)
        lbls_mix = torch.cat([lbls_mix, torch.zeros_like(logits_mix_neg)], dim=1) # N x (2N+K)

        # apply temperature
        logits_ins /= self.T_moco
        logits_mix /= 1
    

        return logits_mix, lbls_mix, logits_ins, k, logits_k, logits_q, index
