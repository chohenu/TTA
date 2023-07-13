
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
