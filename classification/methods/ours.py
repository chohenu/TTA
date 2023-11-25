"""
Builds upon: https://github.com/DianCh/AdaContrast
Corresponding paper: https://arxiv.org/abs/2204.10377
"""

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from sklearn.mixture import GaussianMixture

from utils import concat_all_gather

from methods.base import TTAMethod
from models.model import BaseModel

from utils import CustomDistributedDataParallel

NUM_CLASSES = {"domainnet-126": 126, "VISDA-C": 12, "OfficeHome":65, "pacs":7 , "office":31 }

class hwc_AdaMoCo(nn.Module):
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
        device=None,
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
        super(hwc_AdaMoCo, self).__init__()
        
        self.device=device
        
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
            "mem_probs", torch.rand(K, src_model.num_classes)
        )


        # self.gm = GaussianMixture(n_components=2, random_state=0)

        self.mem_feat = F.normalize(self.mem_feat, dim=0)

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

        self.args = args
        self.confidence = None 
        

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
        # keys = concat_all_gather(keys)
        # pseudo_labels = concat_all_gather(pseudo_labels)
        # probs = concat_all_gather(probs)
        # index = concat_all_gather(index)

        start = self.queue_ptr
        end = start + len(keys)
        idxs_replace = torch.arange(start, end).to(self.device) % self.K
        self.mem_feat[:, idxs_replace] = keys.T
        self.mem_labels[idxs_replace] = pseudo_labels
        self.queue_ptr = end % self.K
        self.mem_probs[idxs_replace, :] = probs
        

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

    def forward(self, im_q, im_k=None, pseudo_labels_w=None, cls_only=False, prototypes_q=None, use_proto_loss_v2=None):
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

        if use_proto_loss_v2: 
            psuedo_label = torch.argmax(logits_q, dim=1) # 64 = [0~ 12]
            norm_proto_q = F.normalize(prototypes_q, dim=1) # 12,256
            select_pos = norm_proto_q[psuedo_label] # 64,256
            l_pos_proto = torch.einsum('nc,nc->n', [q, select_pos]).unsqueeze(-1) # post pair 64,1
            l_neg_proto = torch.mm(q, select_pos.T) # neg pair (64,64)
            proto_logits_ins = torch.cat([l_pos_proto, l_neg_proto], dim=1)
            # apply temperature
            proto_logits_ins /= self.T_moco

            # labels: positive key indicators
            labels_ins = torch.zeros(proto_logits_ins.shape[0], dtype=torch.long).to(self.device)
            mask = torch.ones_like(proto_logits_ins, dtype=torch.bool)
            mask[:, 1:] = psuedo_label.to(self.device).reshape(-1, 1) != psuedo_label  # (B, K)

            proto_logits_ins = torch.where(mask, proto_logits_ins, torch.tensor([float("-inf")]).to(self.device))
            loss_proto = F.cross_entropy(proto_logits_ins, labels_ins)
        else: 
            loss_proto = None
        
        return feats_q, logits_q, logits_ins, k, logits_k, l_neg_near, loss_proto


class hwc_AdaContrast(TTAMethod):
    def __init__(self, num_classes, base_model, momentum_model, optimizer, cfg, steps, episodic, dataset_name, arch_name, queue_size, momentum, temperature, contrast_type, ce_type, alpha, beta, eta,
                 dist_type, ce_sup_type, refine_method, num_neighbors, device):
        super().__init__(base_model.to(device), optimizer, steps, episodic, device)

        self.device = device
        # Hyperparameters
        self.queue_size = queue_size
        self.m = momentum
        self.T_moco = temperature

        self.contrast_type = contrast_type
        self.ce_type = ce_type
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

        self.dist_type = dist_type
        self.ce_sup_type = ce_sup_type
        self.refine_method = refine_method
        self.num_neighbors = num_neighbors

        self.first_X_samples = 0

        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic

        # if dataset_name != "domainnet126":
        #     self.src_model = BaseModel(model, arch_name, dataset_name)
        #     self.momentum_model = BaseModel(momentum_model, arch_name, dataset_name)
        #     # Setup EMA model
        # else:
        self.src_model = base_model
        self.momentum_model = momentum_model

        self.model = hwc_AdaMoCo(
                        src_model=self.src_model,
                        momentum_model=self.momentum_model,
                        K=self.queue_size,
                        m=self.m,
                        T_moco=self.T_moco,
                        device=self.device
                        ).to(self.device)
        
        self.num_classes = num_classes
        self.cfg = cfg
        
        gm = GaussianMixture(n_components=2, random_state=0,max_iter=50)

        self.banks = {
        "features": torch.tensor([], device='cuda'),
        "probs": torch.tensor([], device='cuda'),
        "logit": torch.tensor([], device='cuda'),
        "index": torch.tensor([], device='cuda'),
        "ptr": 0,
        'gm':gm
        }

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_states, self.optimizer_state = \
            self.copy_model_and_optimizer()

    def forward(self, x):
       
        images_test, images_w, images_q, images_k = x
        
        del images_w
        del images_q
        del images_k

        # Train model
        self.model.train()
        super().forward(x)

        # Create the final output prediction
        self.model.eval()
        _, outputs = self.model(images_test, None, None, cls_only=True)
        return outputs

    @torch.no_grad()
    def forward_sliding_window(self, x):
        """
        :param x: The buffered data created with a sliding window
        :return: Dummy output. Has no effect
        """
        imgs_test = x[0]
        return torch.zeros_like(imgs_test)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        idxs, images_w, images_q, images_k = x

        self.model.train()
        # weak aug model output
        feats_w, logits_w = self.model(images_w, cls_only=True)
        with torch.no_grad():
            probs_w = F.softmax(logits_w, dim=1)
            if use_proto := self.first_X_samples >= 1024:
                self.refine_method = "nearest_neighbors"
            else:
                self.refine_method = None
                self.first_X_samples += len(feats_w)
            pseudo_labels_w, probs_w, _ = refine_predictions(
                feats_w, probs_w, self.banks, self.refine_method, self.dist_type, self.num_neighbors)
        
        # similarity btw prototype(mean) and feats_w
        if use_proto: 
            # use confidence center
            cur_probs   = torch.cat([self.banks['probs'], probs_w], dim=0)
            cur_feats   = torch.cat([self.banks['features'], feats_w.detach()], dim=0) ## current 
            cur_pseudo  = torch.cat([self.banks['logit'].argmax(dim=1), pseudo_labels_w], dim=0) ## current 
            
            confidence, _ = self.noise_detect_cls(self.device, cur_probs, cur_pseudo, cur_feats, self.banks)
            origin_idx = torch.arange(self.banks['probs'].size(0), self.banks['probs'].size(0)+probs_w.size(0))
            ignore_idx = confidence[origin_idx] < 0.5 # select noise label
            
            do_noise_detect = True
            use_ce_weight = True
            use_proto_loss_v2 = True
            prototypes = self.get_center_proto(self.banks, use_confidence=False)
        else: 
            do_noise_detect = False
            use_ce_weight = False
            prototypes = None 
            ignore_idx = None
            use_proto_loss_v2 = False
    
        # strong aug model output 
        feats_q, logits_q, logits_ins, feats_k, logits_k, logits_neg_near, loss_proto = self.model(images_q, im_k=images_k, pseudo_labels_w=pseudo_labels_w, 
                                                                                            prototypes_q=prototypes, use_proto_loss_v2=use_proto_loss_v2)
 
        
        # mixup
        alpha = 1.0
        inputs_w, targets_w_a, targets_w_b, lam_w, mix_w_idx = mixup_data(images_w, pseudo_labels_w,
                                                        alpha, self.device, use_cuda=True)
        inputs_w, targets_w_a, targets_w_b = map(Variable, (inputs_w,
                                                    targets_w_a, targets_w_b))
        
        targets_w_mix = lam_w * torch.eye(self.num_classes).to(self.device)[targets_w_a] + (1-lam_w) * torch.eye(self.num_classes).to(self.device)[targets_w_b]
        
        feats_w_mix, target_w_mix_logit = self.model(inputs_w, cls_only=True)
        
  
        
        inputs_q, targets_q_a, targets_q_b, lam_q, mix_q_idx = mixup_data(images_q, pseudo_labels_w,
                                                        alpha, self.device, use_cuda=True)
        inputs_q, targets_q_a, targets_q_b = map(Variable, (inputs_q,
                                                    targets_q_a, targets_q_b))
        
        targets_q_mix = lam_q * torch.eye(self.num_classes).to(self.device)[targets_q_a] + (1-lam_q) * torch.eye(self.num_classes).to(self.device)[targets_q_b]
        
        feats_q_mix, target_q_mix_logit = self.model(inputs_q, cls_only=True)
        
      
        
        # similarity btw prototype and mixup input
        if do_noise_detect:
            weight_a = confidence[origin_idx]
            mix_w_idx = mix_w_idx.to(self.device)
            weight_b = weight_a[mix_w_idx]
            weight_mix_w = lam_w * weight_a + (1-lam_w) * weight_b
            loss_mix = KLLoss(target_w_mix_logit, targets_w_mix, epsilon=1e-8, weight_mix=weight_mix_w)
            
            weight_mix_q = lam_q * weight_a + (1-lam_q) * weight_b
            loss_mix += KLLoss(target_q_mix_logit, targets_q_mix, epsilon=1e-8, weight_mix=weight_mix_q)
            # loss_mix = KLLoss(target_w_mix_logit, targets_w_mix, epsilon=1e-8)
        else:
            loss_mix = mixup_criterion(target_w_mix_logit, targets_w_a, targets_w_b, lam=lam_w, weight_a=None, weight_b=None)
            
        # Calculate reliable degree of pseudo labels
        if use_ce_weight and do_noise_detect:
            CE_weight = confidence[origin_idx]
            CE_weight[ignore_idx] = 0 
        else:
            CE_weight = 1.
            
        # update key features and corresponding pseudo labels
        self.model.update_memory(feats_k, pseudo_labels_w, probs_w, idxs)
        
        
        # sfda instance loss
        loss_ins, _ = instance_loss(
            logits_ins=logits_ins,
            pseudo_labels=pseudo_labels_w,
            mem_labels=self.model.mem_labels,
            logits_neg_near=logits_neg_near,
            contrast_type=self.contrast_type,
            device=self.device
        )
        
        # classification
        loss_cls, _ = classification_loss(
            logits_w, logits_q, pseudo_labels_w, do_noise_detect, CE_weight, self.ce_sup_type
        )

        # diversification
        loss_div = (
            diversification_loss(logits_w, logits_q, self.ce_sup_type)
            if self.eta > 0
            else torch.tensor([0.0]).to(self.device)
        )

        loss = (
            self.alpha * loss_cls
            + self.beta * loss_ins
            + self.eta * loss_div
            + loss_mix
        )
        if loss_proto: loss += loss_proto
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # use slow feature to update neighbor space
        with torch.no_grad():
            feats_w, logits_w = self.model.momentum_model(images_w, return_feats=True)

        self.update_labels(idxs, feats_w, logits_w)

        return logits_q

    def reset(self):
        super().reset()
        self.model = hwc_AdaMoCo(
                        src_model=self.src_model,
                        momentum_model=self.momentum_model,
                        K=self.queue_size,
                        m=self.m,
                        T_moco=self.T_moco,
                        device=self.device
                        ).to(self.device)
        self.first_X_samples = 0
        gm = GaussianMixture(n_components=2, random_state=0,max_iter=50)

        self.banks = {
        "features": torch.tensor([], device=self.device),
        "probs": torch.tensor([], device=self.device),
        "logit": torch.tensor([], device=self.device),
        "index": torch.tensor([], device=self.device),
        "ptr": 0,
        'gm':gm
        }

    @torch.no_grad()
    def update_labels(self, idxs, features, logits):
        # 1) avoid inconsistency among DDP processes, and
        # 2) have better estimate with more data points
        if self.cfg.DISTRIBUTED:
            idxs = concat_all_gather(idxs)
            features = concat_all_gather(features)
            logits = concat_all_gather(logits)
            
        probs = F.softmax(logits, dim=1)
        start = self.banks["ptr"]
        end = start + len(features)
        self.banks["features"] = torch.cat([self.banks["features"], features], dim=0)
        self.banks["probs"] = torch.cat([self.banks["probs"], probs], dim=0)
        self.banks["logit"] = torch.cat([self.banks["logit"], logits], dim=0)
        self.banks["index"] = torch.cat([self.banks["index"], idxs], dim=0)
        self.banks["ptr"] = end % len(self.banks["features"])


    @staticmethod
    def configure_model(model):
        """Configure model"""
        model.train()
        # disable grad, to (re-)enable only what we update
        model.requires_grad_(False)
        # enable all trainable
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
            else:
                m.requires_grad_(True)
        return model
    
    def noise_detect_cls(self, device, cluster_labels, labels, features, banks, temp=0.25, return_center=False):
    
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
        confidence = torch.from_numpy(confidence).float().to(device)
        if return_center : 
            # , losses, pdf / pdf.sum(1)[:, np.newaxis]
            return confidence, losses
        else: 
            return confidence, losses
        
    def get_center_proto(self, banks, use_confidence):
        if use_confidence: 
            ignore_idx = banks['confidence'] > 0.5
            centers = F.normalize(banks['probs'][ignore_idx].T.mm(banks['features'][ignore_idx]), dim=1)
        else: 
            centers = F.normalize(banks['probs'].T.mm(banks['features']), dim=1)
        return centers
    
    

def mixup_data(x, y, alpha=1.0, device=None, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    lam = torch.tensor(lam).to(device)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)
   
    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index

@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank, dist_type, num_neighbors):
    pred_probs = []
    for feats in features.split(64):
        distances = get_distances(feats, features_bank, dist_type)
        _, idxs = distances.sort()
        idxs = idxs[:, : num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)

    return pred_labels, pred_probs



def instance_loss(logits_ins, pseudo_labels, mem_labels, logits_neg_near, contrast_type, device):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).to(device)

    # in class_aware mode, do not contrast with same-class samples
    if contrast_type == "class_aware" and pseudo_labels is not None:
        mask = torch.ones_like(logits_ins, dtype=torch.bool)
        mask[:, 1:] = pseudo_labels.reshape(-1, 1) != mem_labels  # (B, K)
        logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).to(device))
    elif contrast_type == "nearest" and pseudo_labels is not None:
        mask = torch.ones_like(logits_ins, dtype=torch.bool)        
        _, idx_near = torch.topk(logits_neg_near, k=10, dim=-1, largest=True)
        mask[:, 1:] = torch.all(pseudo_labels.unsqueeze(1).repeat(1,10).unsqueeze(1) != mem_labels[idx_near].unsqueeze(0), dim=2) # (B, K)
        logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).to(device))

    loss = F.cross_entropy(logits_ins, labels_ins)

    accuracy = None

    return loss, accuracy


def classification_loss(logits_w, logits_s, target_labels, do_noise_detect, CE_weight, ce_sup_type):
    if not do_noise_detect: CE_weight = 1.
    
    if ce_sup_type == "weak_weak":
        loss_cls = cross_entropy_loss(logits_w, target_labels)
        accuracy = None
    elif ce_sup_type == "weak_strong":
        loss_cls = (CE_weight * cross_entropy_loss(logits_s, target_labels))
        loss_cls = loss_cls[loss_cls!=0].mean() # ignore zero
        accuracy = None
    else:
        raise NotImplementedError(
            f"{ce_sup_type} CE supervision type not implemented."
        )
    return loss_cls, accuracy


def div(logits, epsilon=1e-8):
    probs = F.softmax(logits, dim=1)
    probs_mean = probs.mean(dim=0)
    loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))

    return loss_div


def diversification_loss(logits_w, logits_s, ce_sup_type):
    if ce_sup_type == "weak_weak":
        loss_div = div(logits_w)
    elif ce_sup_type == "weak_strong":
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


def cross_entropy_loss(logits, labels):
    return torch.nn.CrossEntropyLoss(reduction='none')(logits, labels)
    


def entropy_minimization(logits, device):
    if len(logits) == 0:
        return torch.tensor([0.0]).to(device)
    probs = F.softmax(logits, dim=1)
    ents = -(probs * probs.log()).sum(dim=1)

    loss = ents.mean()
    return loss


def get_distances(X, Y, dist_type="euclidean"):
    """
    Args:
        X: (N, D) tensor
        Y: (M, D) tensor
    """
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances

def KLLoss(input, target, epsilon=1e-8, weight_mix=None):
    softmax = F.softmax(input, dim=1)
    kl_loss = (- target * torch.log(softmax + epsilon)).sum(dim=1)
    if weight_mix is not None:
        kl_loss *= torch.exp(weight_mix)
    return kl_loss.mean(dim=0)
def mixup_criterion(pred, y_a, y_b, lam, weight_a, weight_b):
    if weight_a is not None and weight_b is not None:
        return (weight_a * lam * torch.nn.CrossEntropyLoss(reduction='none')(pred, y_a) 
                + weight_b * (1 - lam) * torch.nn.CrossEntropyLoss(reduction='none')(pred, y_b)).mean()
    else:
        return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)
    
    
    
@torch.no_grad()
def refine_predictions(
    features,
    probs,
    banks,
    refine_method,
    dist_type,
    num_neighbors,
    gt_labels=None):
    if refine_method == "nearest_neighbors":
        feature_bank = banks["features"]
        probs_bank = banks["probs"]
        pred_labels, probs = soft_k_nearest_neighbors(
            features, feature_bank, probs_bank, dist_type, num_neighbors
        )
    elif refine_method is None:
        pred_labels = probs.argmax(dim=1)
    else:
        raise NotImplementedError(
            f"{refine_method} refine method is not implemented."
        )
    accuracy = None
    if gt_labels is not None:
        accuracy = (pred_labels == gt_labels).float().mean() * 100

    return pred_labels, probs, accuracy