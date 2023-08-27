
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class ClusterLoss(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(self, tau=1.0, multiplier=2):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier

    def forward(self, c, get_map=False):
        n = c.shape[0]
        assert n % self.multiplier == 0

        # c = c / np.sqrt(self.tau)
        c_list = [x for x in c.chunk(self.multiplier)]
        c_aug0 = c_list[0]
        c_aug1 = c_list[1]
        p_i = c_aug0.sum(0).view(-1)
        p_i /= p_i.sum()
        en_i = np.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_aug1.sum(0).view(-1)
        p_j /= p_j.sum()
        en_j = np.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        en_loss = en_i + en_j

        c = torch.cat((c_aug0.t(), c_aug1.t()), dim=0)
        n = c.shape[0]

        c = F.normalize(c, p=2, dim=1) / np.sqrt(self.tau)

        logits = c @ c.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.log_softmax(logits, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1)

        return loss + en_loss


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

def make_mem_neg(logits_ins, pseudo_labels, mem_labels, contrast_type):
    # in class_aware mode, do not contrast with same-class samples
    mask = torch.ones_like(logits_ins, dtype=torch.bool)
    mask[:, 1:] = pseudo_labels.reshape(-1, 1) != mem_labels  # (B, K)
    logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())
    return mask[:,1:]


def classification_loss(logits_w, logits_s, target_labels, CE_weight, args):
    if not args.learn.do_noise_detect: CE_weight = 1.
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

def mixup_criterion(pred, y_a, y_b, lam):
    return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)

def mixcoloss(logit_mix, y_mix):
    probs = F.softmax(logit_mix, 1) 
    nll_loss = (- y_mix * torch.log(probs)).sum(1).mean()
    # accuracy = calculate_acc(logit_mix, y_mix)
    return nll_loss

import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

