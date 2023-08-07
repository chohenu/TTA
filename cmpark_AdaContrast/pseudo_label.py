import torch
import numpy as np
from utils import (
    adjust_learning_rate,
    concat_all_gather,
    get_augmentation,
    get_distances)

import math

@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank, args):
    pred_probs = []
    for feats in features.split(64):
        distances = get_distances(feats, features_bank, args.learn.dist_type)
        _, idxs = distances.sort()
        idxs = idxs[:, : args.learn.num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)

    return pred_labels, pred_probs

@torch.no_grad()
def soft_k_nearest_neighbors_select(features, features_bank, probs_bank, logit_bank,args):
    pred_probs, keep_index, remove_index = [], [], []
    for feats in features.split(64):
        distances = get_distances(feats, features_bank, args.learn.dist_type)
        _, idxs = distances.sort()
        idxs = idxs[:, : args.learn.num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)

        ###############################
        #### select psuedo labeling####
        ###############################

        pred_start = logit_bank[idxs, :] # select topk similar index 
        pred_start = pred_start.permute((1,0,2)) # change (num_nbrs, 64, num_classes)        
        pred_start     = torch.nn.functional.softmax(torch.squeeze(pred_start), dim=2).max(2)[0] 
        ## Confidence Based Selection
        pred_con       = pred_start                                                                  
        conf_thres     = pred_con.mean()
        confidence_sel = pred_con.mean(0) > conf_thres

        ## Uncertainty Based Selection
        pred_std              = pred_start.std(0)                                                                               
        uncertainty_threshold = pred_std.mean(0)    
        uncertainty_sel       = pred_std<uncertainty_threshold

        ## Confidence and Uncertainty Based Selection
        truth_array = torch.logical_and(uncertainty_sel, confidence_sel)
        ind_keep   = truth_array.nonzero()
        ind_remove = (~truth_array).nonzero()

        keep_index.append(ind_keep)
        remove_index.append(ind_remove)

        ################################
        ##### done psuedo labeling #####
        ################################
        

    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)
    keep_index = torch.cat(keep_index)
    remove_index = torch.cat(remove_index)
    return pred_labels, pred_probs, [keep_index,remove_index]

@torch.no_grad()
def soft_k_nearest_neighbors_fixmatch(features, features_bank, probs_bank, args):
    filter_type = 'None' # ['mix','sim']

    pred_probs, stack_num_neighbors = [], []
    for feats in features.split(64):
        distances = get_distances(feats, features_bank, args.learn.dist_type)
        _, idxs = distances.sort()
        idxs = idxs[:, : args.learn.num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :]
        if filter_type == 'prob': 
            f_probs = []
            for b in range(probs.size(0)): # filtering nbr
                max_probs , max_idx = torch.max(probs[b], dim=-1)
                # same_idx = max_idx == max_idx[0] # max prob
                filter_prob = max_probs > 0.5 # max prob
                ind_keep = filter_prob
                f_probs.append(probs[b,ind_keep].mean(0))
            pred_probs.append(torch.stack(f_probs))

        elif filter_type == 'mix': 
            f_probs = []
            for b in range(probs.size(0)): # filtering nbr
                max_probs , max_idx = torch.max(probs[b], dim=-1)
                ind_keep = max_probs > 0.7 
                f_probs.append(probs[b,ind_keep].mean(0))
            pred_probs.append(torch.stack(f_probs))

        else: 
            pred_probs.append(probs.mean(1))
        ## filtering prob
        # 
        stack_num_neighbors.append(idxs.cpu().numpy())

    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)
    stack_num_neighbors = np.concatenate(stack_num_neighbors)

    return pred_labels, pred_probs, stack_num_neighbors


@torch.no_grad()
def soft_k_nearest_neighbors_pos_neg(features, features_bank, probs_bank, args):
    pred_probs, stack_num_neighbors = [], []
    for feats in features.split(64):
        distances = get_distances(feats, features_bank, args.learn.dist_type)
        _, idxs = distances.sort()
        pos_idxs = idxs[:, : args.learn.num_neighbors]
        neg_idxs = idxs[:, : args.learn.num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[pos_idxs, :].mean(1)
        pred_probs.append(probs)
        stack_num_neighbors.append(pos_idxs.cpu().numpy())

    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)
    stack_num_neighbors = np.concatenate(stack_num_neighbors)

    return pred_labels, pred_probs, stack_num_neighbors

@torch.no_grad()
def soft_k_nearest_neighbors_based_context(features, features_bank, probs_bank, confidence_bank,
                                            context_assignments_bank, centers_bank,args):
    pred_probs = []
    for feats in features.split(64):
        distances = get_distances(feats, features_bank, args.learn.dist_type)
        _, idxs = distances.sort()
        idxs = idxs[:, : args.learn.num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)

    return pred_labels, pred_probs

@torch.no_grad()
def soft_gmm_clustering(features, features_bank, probs_bank, args):
    # features # all current stack features (D, F)
    # features_bank # shuffling all features (D, F)
    # probs_bank # shuffling all pro b(D, C)
    
    clss_num = probs_bank.size(1)
    uniform = torch.ones(len(features),clss_num)/clss_num
    uniform = uniform.cuda()

    pi = probs_bank.sum(dim=0)
    mu = torch.matmul(probs_bank.t(),(features_bank)) # matrix multiple (F,C) center??
    mu = mu / pi.unsqueeze(dim=-1).expand_as(mu) # normalize first 

    zz, gamma = gmm((features_bank), pi, mu, uniform)
    pred_labels = gamma.argmax(dim=1)
    
    for round in range(1):
        pi = gamma.sum(dim=0)
        mu = torch.matmul(gamma.t(), (features_bank))
        mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

        zz, gamma = gmm((features_bank), pi, mu, gamma)
        pred_labels = gamma.argmax(axis=1)
            
    return pred_labels, None, None

@torch.no_grad()
def torch_gmm_clustering(features, features_bank, probs_bank, args):
    # features # all current stack features (D, F)
    # features_bank # shuffling all features (D, F)
    # probs_bank # shuffling all pro b(D, C)
    
    
    clss_num = probs_bank.size(1)
    uniform = torch.ones(len(features_bank),clss_num)/clss_num
    uniform = uniform.cuda()

    pi = probs_bank.sum(dim=0)
    mu = torch.matmul(probs_bank.t(),(features_bank)) ## matrix multiple (F,C)
    mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

    zz, gamma = gmm((features_bank), pi, mu, uniform)
    pred_labels = gamma.argmax(dim=1)
    
    for round in range(1):
        pi = gamma.sum(dim=0)
        mu = torch.matmul(gamma.t(), (features_bank))
        mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

        zz, gamma = gmm((features), pi, mu, gamma)
        pred_labels = gamma.argmax(axis=1)
            
    return pred_labels, None, None
    # array
    # array_feature_bank = features_bank.cpu().numpy()
    # n_features = features.size(1)
    # n_components = probs_bank.size(1)
    # logging.info(f"Making Gaussian mixture model")
    # model = GaussianMixture(n_components=n_components, covariance_type="diag", random_state=0)
    # model.fit(array_feature_bank)
    # y_p = model.predict_proba(array_feature_bank)
    # return y_p

@torch.no_grad()
def center_nearest_neighbors(features, features_bank, probs_bank, confidence_bank,
                            context_assignments_bank, centers_bank,args):
    pred_probs = []
    for feats in features.split(64):
        distances = get_distances(feats, features_bank, args.learn.dist_type)
        _, idxs = distances.sort()
        idxs = idxs[:, : args.learn.num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)

    return pred_labels, pred_probs

@torch.no_grad()
def gmm(all_fea, pi, mu, all_output):    
    epsilon = 1e-6
    Cov = []
    dist = []
    log_probs = []
    
    for i in range(len(mu)):
        temp = all_fea - mu[i]
        predi = all_output[:,i].unsqueeze(dim=-1)
        Covi = torch.matmul(temp.t(), temp * predi.expand_as(temp)) / (predi.sum()) + epsilon * torch.eye(temp.shape[1]).cuda()
        try:
            chol = torch.linalg.cholesky(Covi)
        except RuntimeError:
            Covi += epsilon * torch.eye(temp.shape[1]).cuda() * 100
            chol = torch.linalg.cholesky(Covi)
        chol_inv = torch.inverse(chol)
        Covi_inv = torch.matmul(chol_inv.t(), chol_inv)
        logdet = torch.logdet(Covi)
        mah_dist = (torch.matmul(temp, Covi_inv) * temp).sum(dim=1)
        log_prob = -0.5*(Covi.shape[0] * np.log(2*math.pi) + logdet + mah_dist) + torch.log(pi)[i]
        Cov.append(Covi)
        log_probs.append(log_prob)
        dist.append(mah_dist)
    Cov = torch.stack(Cov, dim=0)
    dist = torch.stack(dist, dim=0).t()
    log_probs = torch.stack(log_probs, dim=0).t()
    zz = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True).expand_as(log_probs)
    gamma = torch.exp(zz)
    
    return zz, gamma

class CenterGMM:
    def __init__(self, features_bank, probs_bank, bank, args): 

        # cacluate uniform 
        clss_num = probs_bank.size(1)
        uniform = torch.ones(len(features_bank),clss_num)/clss_num
        uniform = uniform.cuda()
    
        pi = probs_bank.sum(dim=0)
        mu = torch.matmul(probs_bank.t(),(features_bank)) ## matrix multiple (F,C)
        mu = mu / pi.unsqueeze(dim=-1).expand_as(mu) # normalize first 
        zz, gamma = gmm((features_bank), pi, mu, uniform)
        self.gamma = gamma
        self.bank = bank
        
    def __call__(self, features_bank, index_bank, batch_idxs):
        pi = self.gamma.sum(dim=0)
        mu = torch.matmul(self.gamma.t(), (features_bank))
        mu = mu / pi.unsqueeze(dim=-1).expand_as(mu) # normalize first 
        zz, self.gamma = gmm((features_bank), pi, mu, self.gamma)
        origin_gamm = self.gamma[index_bank]
        pseudo_label = origin_gamm[batch_idxs] 
        return pseudo_label, mu

@torch.no_grad()
def refine_predictions(
    features,
    probs,
    banks,
    args,
    gt_labels=None,
    return_index=False,
):
    if args.learn.refine_method == "nearest_neighbors":
        feature_bank = banks["features"]
        probs_bank = banks["probs"]
        pred_labels, probs, stack_index = soft_k_nearest_neighbors(
            features, feature_bank, probs_bank, args
        )
        

    elif args.learn.refine_method == "nearest_neighbors_fixmatch":
        feature_bank = banks["features"]
        probs_bank = banks["probs"]
        pred_labels, probs, stack_index = soft_k_nearest_neighbors_fixmatch(
            features, feature_bank, probs_bank, args
        )

    elif args.learn.refine_method == "nearest_neighbors_pos_neg":
        feature_bank = banks["features"]
        probs_bank = banks["probs"]
        pred_labels, probs, stack_index = soft_k_nearest_neighbors_pos_neg(
            features, feature_bank, probs_bank, args
        )

    elif args.learn.refine_method == "nearest_neighbors_select":
        feature_bank = banks["features"]
        probs_bank = banks["probs"]
        logit_bank = banks["logit"]
        pred_labels, probs, stack_index = soft_k_nearest_neighbors_select(
            features, feature_bank, probs_bank, logit_bank, args
        )


    elif args.learn.refine_method == "gmm":
        feature_bank = banks["features"]
        probs_bank = banks["probs"]

        ## TODO : Calculate distance using GMM model 
        ## using Attributes function (ex, predict(features) or predict_proba(features)  
        ## Link : https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
        pred_labels,_,stack_index= soft_gmm_clustering(
            features, feature_bank, probs_bank, args
        )
    elif args.learn.refine_method is None:
        pred_labels = probs.argmax(dim=1)
    else:
        raise NotImplementedError(
            f"{args.learn.refine_method} refine method is not implemented."
        )
    accuracy = None
    if gt_labels is not None:
        accuracy = (pred_labels == gt_labels).float().mean() * 100
    if return_index: 
        return pred_labels, probs, accuracy, stack_index
    else:
        return pred_labels, probs, accuracy
