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
        dataset_legth=None,
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

        self.register_buffer(
            "mem_probs", torch.rand(K, src_model.num_classes)
        )

        self.register_buffer(
            "mem_index", torch.randint(0, dataset_legth, (K,))
        )

        # self.gm = GaussianMixture(n_components=2, random_state=0)

        self.mem_feat = F.normalize(self.mem_feat, dim=0)

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

        self.cluster_loss = nn.KLDivLoss(size_average=False)

        self.sce_loss = SCELoss(1, 1, src_model.num_classes)
        self.args = args
    def fit_gmm(self, banks): 
        labels = self.mem_labels.long()
        centers = F.normalize(self.mem_probs.T.mm(self.mem_feat.T), dim=1)
        context_assigments_logits = self.mem_feat.T.mm(centers.T) / 0.25 # sim feature with center
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
        self.confidence = confidence
        self.losses = losses
        self.distance_pseudo = torch.argmax(context_assigments, axis=1)

    def check_accuracy(self,): 
        match_confi = self.confidence > 0.5
        match_label = self.mem_labels == self.mem_gt
        noise_accuracy = (match_confi == match_label).float().mean()

        # only_clean_accuracy = ((match_confi == True) & (match_label == True)).float().mean()
        only_clean_accuracy = (self.mem_gt[match_confi] == self.mem_labels[match_confi]).float().mean()
        only_noise_accuracy = (self.mem_gt[~match_confi] == self.mem_labels[~match_confi]).float().mean()

        logging.info(f"model_noise_accuracy: {noise_accuracy}")
        logging.info(f"model_only_clean_accuracy: {only_clean_accuracy}")
        logging.info(f"model_only_noise_accuracy: {only_noise_accuracy}")

        log_dict = {"model_noise_accuracy" : noise_accuracy}
        log_dict.update({"model_only_clean_accuracy" : only_clean_accuracy})
        log_dict.update({"model_only_noise_accuracy" : only_noise_accuracy})
        return log_dict

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
                'mem_pseudo_labels': self.mem_labels.cpu().numpy(),
                'mem_gt':self.mem_gt.cpu().numpy()}

    @torch.no_grad()
    def update_memory(self, keys, pseudo_labels, gt_labels, probs, index):
        """
        Update features and corresponding pseudo labels
        """
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        pseudo_labels = concat_all_gather(pseudo_labels)
        gt_labels = concat_all_gather(gt_labels.to('cuda'))
        probs = concat_all_gather(probs)
        index = concat_all_gather(index)

        start = self.queue_ptr
        end = start + len(keys)
        idxs_replace = torch.arange(start, end).cuda() % self.K
        self.mem_feat[:, idxs_replace] = keys.T
        self.mem_labels[idxs_replace] = pseudo_labels
        self.mem_gt[idxs_replace] = gt_labels
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

    def forward(self, im_q, banks, idxs, im_k=None, pseudo_labels_w=None, epoch=None, cls_only=False, prototypes_q=None, prototypes_k=None, ignore_idx=None):
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
            
        # with torch.no_grad():
        #     softmax_out = torch.nn.Softmax(dim=1)(logits_q)

        #     origin_idx = torch.where(idxs.reshape(-1,1)==banks['index'])[1]
        #     banks['norm_features'][origin_idx] = q

        #     similarity = q @ banks['norm_features'].T
        #     _, idx_near = torch.topk(similarity, dim=-1, largest=True, k=5 + 1)
        #     idx_near = idx_near[:, 1:]  # batch x K
        #     feat_near = banks['norm_features'][idx_near]  # batch x K x F            
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
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        q_ = q.unsqueeze(1).expand(-1, 5, -1) # batch x K x F 
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # l_pos = torch.einsum("nkc,nkc->n", [q_, feat_near]).unsqueeze(-1)
        # l_pos = torch.einsum("nkc,nkc->nk", [q_, feat_near]).unsqueeze(-1) # B, K, 1
        # l_pos = l_pos.mean(axis=1) # ) # B, 1
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
        # proto_loss = None
        
        use_nce_type = False
        diff_nce_type = True
        if use_nce_type: 
            norm_proto_q = F.normalize(prototypes_q, dim=1)
            norm_proto_k = F.normalize(prototypes_k, dim=1)
            l_pos_proto = torch.einsum('nc,nc->n', [norm_proto_q, norm_proto_k]).unsqueeze(-1)
            l_neg_proto = torch.einsum("nc,ck->nk", [norm_proto_q, self.mem_feat.clone().detach()])
            proto_logits_ins = torch.cat([l_pos_proto, l_neg_proto], dim=1)
            # apply temperature
            proto_logits_ins /= self.T_moco

            # labels: positive key indicators
            labels_ins = torch.zeros(proto_logits_ins.shape[0], dtype=torch.long).cuda()
            mask = torch.ones_like(proto_logits_ins, dtype=torch.bool)
            mask[:, 1:] = torch.range(0,l_pos_proto.size(0)-1).cuda().reshape(-1, 1) != self.mem_labels  # (B, K)
            clean_confi = self.confidence > 0.5
            clean_confi = clean_confi.unsqueeze(0).repeat(mask.size(0),1)
            # mask[:,1:] = mask[:,1:] * clean_confi # (B, K) 

            proto_logits_ins = torch.where(mask, proto_logits_ins, torch.tensor([float("-inf")]).cuda())
            proto_loss = F.cross_entropy(proto_logits_ins, labels_ins)

        elif diff_nce_type: 
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
            origin_idx = torch.where(idxs.reshape(-1,1)==banks['index'])[1]
            batch_confidence = banks['confidence'][origin_idx]
            if self.args.learn.use_conf_filter: 
                clean_confi = batch_confidence > 0.5
            else: 
                clean_confi = batch_confidence < 0.5

            clean_confi = clean_confi.unsqueeze(0).repeat(mask.size(0),1)
            # mask[:,1:] = mask[:,1:] * clean_confi # (B, K) 

            proto_logits_ins = torch.where(mask, proto_logits_ins, torch.tensor([float("-inf")]).cuda())
            proto_loss = F.cross_entropy(proto_logits_ins, labels_ins)
        else: 
            proto_prob = F.softmax(self.src_model.classifier_q(F.normalize(prototypes_q, dim=1)), dim=1)
            proto_sim = q @ F.normalize(prototypes_q, dim=1).T ## (B x feature)x (Features class)= B, class
            proto_label = torch.argmax(proto_sim, axis=1) 
            proto_sim = F.softmax(proto_sim,dim=1)
            if ignore_idx.sum() > 0:
                proto_loss = torch.nn.CrossEntropyLoss(reduction='none')(proto_sim, proto_label)
                proto_loss = proto_loss[ignore_idx].mean()  
                # proto_loss = F.kl_div(proto_sim[ignore_idx], proto_prob[proto_label][ignore_idx], reduction="none").sum(-1).mean()
            else:
                proto_loss = (torch.tensor([0.0]).to("cuda")*torch.nn.CrossEntropyLoss(reduction='none')(proto_sim, proto_label)).mean()
        
        
        # proto_label = torch.argmax(proto_sim, axis=1)
        # weight_sim = proto_sim/self.T_moco
        # if ignore_idx.sum() > 0:
            # proto_loss = torch.nn.CrossEntropyLoss(reduction='none')(proto_sim, proto_label)
            # proto_loss = proto_loss[ignore_idx].mean()
            # F.kl_div(proto_sim,proto_label)
            # proto_loss = self.sce_loss(proto_sim,proto_label, ignore_idx)
        # else:
        #     proto_loss = (torch.tensor([0.0]).to("cuda")*torch.nn.CrossEntropyLoss(reduction='none')(proto_sim, proto_label)).mean()
        
        # with torch.no_grad():
            # proto_sim = q @ F.normalize(prototypes, dim=1).T ## (B x feature)x (Features class)= B, class
            # proto_sim /= self.T_moco
            # proto_sim = proto_sim**2 # sharping
            # proto_label = torch.argmax(proto_sim, axis=1) 
            # proto_loss = F.cross_entropy(proto_sim, proto_label) 
            # weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9)
            # proto_loss = self.cluster_loss(proto_sim, softmax_out)/softmax_out.shape[0]

        #     # + F.cross_entropy(proto_sim, torch.argmax(softmax_out,1))

        # cosin_distance = (1 - similarity)
        # (cosin_distance**0.5).mean(axis=1)/torch.log(torch.tensor(cosin_distance.size(1)+10))
        # # if cluster_result is not None:  
        # index = torch.where(idxs.reshape(-1,1)==banks['index'])[1]
        # proto_sim = q @ F.normalize(prototypes, dim=1).T ## (B x feature)x (Features class)= B, class
        # proto_label = torch.argmax(proto_sim, axis=1) 
        # pos_prototypes = prototypes[proto_label]

        # all_proto_id = list(range(im2cluster.max()+1))
        # neg_proto_id = set(proto_label.max()+1)-set(pos_proto_id.tolist())

        # proto_labels = []
        # proto_logits = []
        
        # # for n, (im2cluster,prototypes,density) in enumerate(zip(1 - similarity, prototypes , cluster_result['density'])):
        #     # get positive prototypes
        # # pos_proto_id = im2cluster[index]
        # # pos_prototypes = prototypes[pos_proto_id]    
        
        # # sample negative prototypes
        # all_proto_id = [i for i in range(im2cluster.max()+1)]       
        # neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist())
        # neg_proto_id = random.sample(neg_proto_id,self.r) #sample r negative prototypes 
        # neg_prototypes = prototypes[neg_proto_id]    

        # proto_selected = torch.cat([pos_prototypes,neg_prototypes],dim=0)
        
        # # compute prototypical logits
        # logits_proto = torch.mm(q,proto_selected.t())
        
        # # targets for prototype assignment
        # labels_proto = torch.linspace(0, q.size(0)-1, steps=q.size(0)).long().cuda()
        
        # # scaling temperatures for the selected prototypes
        # temp_proto = density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id).cuda()],dim=0)]  
        # logits_proto /= temp_proto
        
        # proto_labels.append(labels_proto)
        # proto_logits.append(logits_proto)
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
