"""
Builds upon: https://github.com/DianCh/AdaContrast
Corresponding paper: https://arxiv.org/abs/2204.10377
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import TTAMethod
from models.model import BaseModel

import numpy as np

from math import pi
from scipy.special import logsumexp


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
		device='cuda',
	):
		"""
		dim: feature dimension (default: 128)
		K: buffer size; number of keys
		m: moco momentum of updating key encoder (default: 0.999)
		T: softmax temperature (default: 0.07)
		"""
		super(AdaMoCo, self).__init__()
		self.device = device
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
		self.register_buffer("mem_labels", torch.randint(0, src_model.num_classes, (K,)))
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
	def update_memory(self, keys, pseudo_labels):
		"""
		Update features and corresponding pseudo labels
		"""

		start = self.queue_ptr
		end = start + len(keys)
		idxs_replace = torch.arange(start, end).to(self.device) % self.K
		self.mem_feat[:, idxs_replace] = keys.T
		self.mem_labels[idxs_replace] = pseudo_labels
		self.queue_ptr = end % self.K

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

			k, _ = self.momentum_model(im_k, return_feats=True)
			k = F.normalize(k, dim=1)

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
		return feats_q, logits_q, logits_ins, k


class AdaContrast_v2(TTAMethod):
	def __init__(self, model, optimizer, steps, episodic, dataset_name, arch_name, queue_size, momentum, temperature, contrast_type, ce_type, alpha, beta, eta,
				 dist_type, ce_sup_type, refine_method, num_neighbors, device):
		super().__init__(model.to(device), optimizer, steps, episodic, device)

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

		if dataset_name != "domainnet126":
			self.src_model = BaseModel(model, arch_name, dataset_name)
		else:
			self.src_model = model

		# Setup EMA model
		self.momentum_model = self.copy_model(self.src_model)

		self.model = AdaMoCo(
						src_model=self.src_model,
						momentum_model=self.momentum_model,
						K=self.queue_size,
						m=self.m,
						T_moco=self.T_moco,
						device=self.device).to(self.device)

		self.banks = {
			"features": torch.tensor([], device=device),
			"probs": torch.tensor([], device=device),
			"ptr": 0
		}

		# note: if the self.model is never reset, like for continual adaptation,
		# then skipping the state copy would save memory
		self.models = [self.src_model, self.momentum_model]
		self.model_states, self.optimizer_state = \
			self.copy_model_and_optimizer()

	def forward(self, x):
		images_test, images_w, images_q, images_k = x

		# Train model
		self.model.train()
		super().forward(x)
		
		# Create the final output prediction
		self.model.eval()
		_, outputs = self.model(images_test, cls_only=True)
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
		_, images_w, images_q, images_k = x

		self.model.train()
		feats_w, logits_w = self.model(images_w, cls_only=True)
		with torch.no_grad():
			probs_w = F.softmax(logits_w, dim=1)
			if self.first_X_samples >= 1024:
				self.refine_method = "nearest_neighbors"
			else:
				self.refine_method = None
				self.first_X_samples += len(feats_w)

			pseudo_labels_w, probs_w, _ = refine_predictions(
				feats_w, probs_w, self.banks, self.refine_method, self.dist_type, self.num_neighbors, self.device,
			)

		_, logits_q, logits_ins, keys = self.model(images_q, images_k)
		# update key features and corresponding pseudo labels
		self.model.update_memory(keys, pseudo_labels_w)

		# moco instance discrimination
		loss_ins, _ = instance_loss(
			logits_ins=logits_ins,
			pseudo_labels=pseudo_labels_w,
			mem_labels=self.model.mem_labels,
			contrast_type=self.contrast_type,
			device=self.device,
		)

		# classification
		loss_cls, _ = classification_loss(
			logits_w, logits_q, pseudo_labels_w, self.ce_sup_type
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
		)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# use slow feature to update neighbor space
		with torch.no_grad():
			feats_w, logits_w = self.model.momentum_model(images_w, return_feats=True)

		self.update_labels(feats_w, logits_w)

		return logits_q

	def reset(self):
		super().reset()
		self.model = AdaMoCo(
						src_model=self.src_model,
						momentum_model=self.momentum_model,
						K=self.queue_size,
						m=self.m,
						T_moco=self.T_moco,
						).to(self.device)
		self.first_X_samples = 0
		self.banks = {
			"features": torch.tensor([], device=self.device),
			"probs": torch.tensor([], device=self.device),
			"ptr": 0
		}

	@torch.no_grad()
	def update_labels(self, features, logits):
		# 1) avoid inconsistency among DDP processes, and
		# 2) have better estimate with more data points

		probs = F.softmax(logits, dim=1)

		start = self.banks["ptr"]
		end = start + len(features)
		if self.banks["features"].shape[0] < self.queue_size:
			self.banks["features"] = torch.cat([self.banks["features"], features], dim=0)
			self.banks["probs"] = torch.cat([self.banks["probs"], probs], dim=0)
			self.banks["ptr"] = end % len(self.banks["features"])
		else:
			idxs_replace = torch.arange(start, end).to(self.device) % len(self.banks["features"])
			self.banks["features"][idxs_replace, :] = features
			self.banks["probs"][idxs_replace, :] = probs
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

@torch.no_grad()
def GMM_clustering(features, features_bank, probs_bank, device):
	n_features = features.size(1)
	n_components = 126
	
	model = GaussianMixture(n_components, n_features, covariance_type='full').to(device)
	model.fit(features_bank)
	
	y_p = model.predict(features, probs=True)
	
	return y_p
	
@torch.no_grad()
def refine_predictions(
	features,
	probs,
	banks,
	refine_method,
	dist_type,
	num_neighbors,
	device,
	gt_labels=None):
	
	if refine_method == "nearest_neighbors":
		feature_bank = banks["features"]
		probs_bank = banks["probs"]
		pred_labels, probs = soft_k_nearest_neighbors(
			features, feature_bank, probs_bank, dist_type, num_neighbors
		)
	elif refine_method == "GMM":
		feature_bank = banks["features"]
		probs_bank = banks["probs"]
		pred_labes, probs = GMM_clustering(
			features, feature_bank, probs_bank, device
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


def instance_loss(logits_ins, pseudo_labels, mem_labels, contrast_type, device):
	# labels: positive key indicators
	labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).to(device)

	# in class_aware mode, do not contrast with same-class samples
	if contrast_type == "class_aware" and pseudo_labels is not None:
		mask = torch.ones_like(logits_ins, dtype=torch.bool)
		mask[:, 1:] = pseudo_labels.reshape(-1, 1) != mem_labels  # (B, K)
		logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).to(device))

	loss = F.cross_entropy(logits_ins, labels_ins)

	accuracy = None

	return loss, accuracy


def classification_loss(logits_w, logits_s, target_labels, ce_sup_type):
	if ce_sup_type == "weak_weak":
		loss_cls = cross_entropy_loss(logits_w, target_labels)
		accuracy = None
	elif ce_sup_type == "weak_strong":
		loss_cls = cross_entropy_loss(logits_s, target_labels)
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
	return F.cross_entropy(logits, labels)


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

def calculate_matmul_n_times(n_components, mat_a, mat_b):
	"""
	Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
	Bypasses torch.matmul to reduce memory footprint.
	args:
		mat_a:      torch.Tensor (n, k, 1, d)
		mat_b:      torch.Tensor (1, k, d, d)
	"""
	res = torch.zeros(mat_a.shape).to(mat_a.device)
	
	for i in range(n_components):
		mat_a_i = mat_a[:, i, :, :].squeeze(-2)
		mat_b_i = mat_b[0, i, :, :].squeeze()
		res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)
	
	return res


def calculate_matmul(mat_a, mat_b):
	"""
	Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
	Bypasses torch.matmul to reduce memory footprint.
	args:
		mat_a:      torch.Tensor (n, k, 1, d)
		mat_b:      torch.Tensor (n, k, d, 1)
	"""
	assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
	return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)


class GaussianMixture(torch.nn.Module):
	"""
	Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components).
	Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
	The model then extends them to (n, 1, d).
	The model parametrization (mu, sigma) is stored as (1, k, d),
	probabilities are shaped (n, k, 1) if they relate to an individual sample,
	or (1, k, 1) if they assign membership probabilities to one of the mixture components.
	"""
	def __init__(self, n_components, n_features, covariance_type="full", eps=1.e-6, init_params="kmeans", mu_init=None, var_init=None):
		"""
		Initializes the model and brings all tensors into their required shape.
		The class expects data to be fed as a flat tensor in (n, d).
		The class owns:
			x:               torch.Tensor (n, 1, d)
			mu:              torch.Tensor (1, k, d)
			var:             torch.Tensor (1, k, d) or (1, k, d, d)
			pi:              torch.Tensor (1, k, 1)
			covariance_type: str
			eps:             float
			init_params:     str
			log_likelihood:  float
			n_components:    int
			n_features:      int
		args:
			n_components:    int
			n_features:      int
		options:
			mu_init:         torch.Tensor (1, k, d)
			var_init:        torch.Tensor (1, k, d) or (1, k, d, d)
			covariance_type: str
			eps:             float
			init_params:     str
		"""
		super(GaussianMixture, self).__init__()

		self.n_components = n_components
		self.n_features = n_features

		self.mu_init = mu_init
		self.var_init = var_init
		self.eps = eps

		self.log_likelihood = -np.inf

		self.covariance_type = covariance_type
		self.init_params = init_params

		assert self.covariance_type in ["full", "diag"]
		assert self.init_params in ["kmeans", "random"]

		self._init_params()


	def _init_params(self):
		if self.mu_init is not None:
			assert self.mu_init.size() == (1, self.n_components, self.n_features), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
			# (1, k, d)
			self.mu = torch.nn.Parameter(self.mu_init, requires_grad=False)
		else:
			self.mu = torch.nn.Parameter(torch.randn(1, self.n_components, self.n_features), requires_grad=False)

		if self.covariance_type == "diag":
			if self.var_init is not None:
				# (1, k, d)
				assert self.var_init.size() == (1, self.n_components, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
				self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
			else:
				self.var = torch.nn.Parameter(torch.ones(1, self.n_components, self.n_features), requires_grad=False)
		elif self.covariance_type == "full":
			if self.var_init is not None:
				# (1, k, d, d)
				assert self.var_init.size() == (1, self.n_components, self.n_features, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i, %i)" % (self.n_components, self.n_features, self.n_features)
				self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
			else:
				self.var = torch.nn.Parameter(
					torch.eye(self.n_features).reshape(1, 1, self.n_features, self.n_features).repeat(1, self.n_components, 1, 1),
					requires_grad=False
				)

		# (1, k, 1)
		self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_components, 1), requires_grad=False).fill_(1. / self.n_components)
		self.params_fitted = False


	def check_size(self, x):
		if len(x.size()) == 2:
			# (n, d) --> (n, 1, d)
			x = x.unsqueeze(1)

		return x


	def bic(self, x):
		"""
		Bayesian information criterion for a batch of samples.
		args:
			x:      torch.Tensor (n, d) or (n, 1, d)
		returns:
			bic:    float
		"""
		x = self.check_size(x)
		n = x.shape[0]

		# Free parameters for covariance, means and mixture components
		free_params = self.n_features * self.n_components + self.n_features + self.n_components - 1

		bic = -2. * self.__score(x, as_average=False).mean() * n + free_params * np.log(n)

		return bic


	def fit(self, x, delta=1e-3, n_iter=100, warm_start=False):
		"""
		Fits model to the data.
		args:
			x:          torch.Tensor (n, d) or (n, k, d)
		options:
			delta:      float
			n_iter:     int
			warm_start: bool
		"""
		if not warm_start and self.params_fitted:
			self._init_params()

		x = self.check_size(x)

		if self.init_params == "kmeans" and self.mu_init is None:
			mu = self.get_kmeans_mu(x, n_centers=self.n_components)
			self.mu.data = mu

		i = 0
		j = np.inf

		while (i <= n_iter) and (j >= delta):

			log_likelihood_old = self.log_likelihood
			mu_old = self.mu
			var_old = self.var

			self.__em(x)
			self.log_likelihood = self.__score(x)

			if torch.isinf(self.log_likelihood.abs()) or torch.isnan(self.log_likelihood):
				device = self.mu.device
				# When the log-likelihood assumes unbound values, reinitialize model
				self.__init__(self.n_components,
					self.n_features,
					covariance_type=self.covariance_type,
					mu_init=self.mu_init,
					var_init=self.var_init,
					eps=self.eps)
				for p in self.parameters():
					p.data = p.data.to(device)
				if self.init_params == "kmeans":
					self.mu.data, = self.get_kmeans_mu(x, n_centers=self.n_components)

			i += 1
			j = self.log_likelihood - log_likelihood_old

			if j <= delta:
				# When score decreases, revert to old parameters
				self.__update_mu(mu_old)
				self.__update_var(var_old)

		self.params_fitted = True


	def predict(self, x, probs=False):
		"""
		Assigns input data to one of the mixture components by evaluating the likelihood under each.
		If probs=True returns normalized probabilities of class membership.
		args:
			x:          torch.Tensor (n, d) or (n, 1, d)
			probs:      bool
		returns:
			p_k:        torch.Tensor (n, k)
			(or)
			y:          torch.LongTensor (n)
		"""
		x = self.check_size(x)

		weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

		if probs:
			p_k = torch.exp(weighted_log_prob)
			return torch.squeeze(p_k / (p_k.sum(1, keepdim=True)))
		else:
			return torch.squeeze(torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor))


	def predict_proba(self, x):
		"""
		Returns normalized probabilities of class membership.
		args:
			x:          torch.Tensor (n, d) or (n, 1, d)
		returns:
			y:          torch.LongTensor (n)
		"""
		return self.predict(x, probs=True)


	def sample(self, n):
		"""
		Samples from the model.
		args:
			n:          int
		returns:
			x:          torch.Tensor (n, d)
			y:          torch.Tensor (n)
		"""
		counts = torch.distributions.multinomial.Multinomial(total_count=n, probs=self.pi.squeeze()).sample()
		x = torch.empty(0, device=counts.device)
		y = torch.cat([torch.full([int(sample)], j, device=counts.device) for j, sample in enumerate(counts)])

		# Only iterate over components with non-zero counts
		for k in np.arange(self.n_components)[counts > 0]: 
			if self.covariance_type == "diag":
				x_k = self.mu[0, k] + torch.randn(int(counts[k]), self.n_features, device=x.device) * torch.sqrt(self.var[0, k])
			elif self.covariance_type == "full":
				d_k = torch.distributions.multivariate_normal.MultivariateNormal(self.mu[0, k], self.var[0, k])
				x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])

			x = torch.cat((x, x_k), dim=0)

		return x, y


	def score_samples(self, x):
		"""
		Computes log-likelihood of samples under the current model.
		args:
			x:          torch.Tensor (n, d) or (n, 1, d)
		returns:
			score:      torch.LongTensor (n)
		"""
		x = self.check_size(x)

		score = self.__score(x, as_average=False)
		return score


	def _estimate_log_prob(self, x):
		"""
		Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
		args:
			x:            torch.Tensor (n, d) or (n, 1, d)
		returns:
			log_prob:     torch.Tensor (n, k, 1)
		"""
		x = self.check_size(x)

		if self.covariance_type == "full":
			mu = self.mu
			var = self.var

			precision = torch.inverse(var)
			d = x.shape[-1]

			log_2pi = d * np.log(2. * pi)

			log_det = self._calculate_log_det(precision)

			x_mu_T = (x - mu).unsqueeze(-2)
			x_mu = (x - mu).unsqueeze(-1)

			x_mu_T_precision = calculate_matmul_n_times(self.n_components, x_mu_T, precision)
			x_mu_T_precision_x_mu = calculate_matmul(x_mu_T_precision, x_mu)

			return -.5 * (log_2pi - log_det + x_mu_T_precision_x_mu)

		elif self.covariance_type == "diag":
			mu = self.mu
			prec = torch.rsqrt(self.var)

			log_p = torch.sum((mu * mu + x * x - 2 * x * mu) * prec, dim=2, keepdim=True)
			log_det = torch.sum(torch.log(prec), dim=2, keepdim=True)

			return -.5 * (self.n_features * np.log(2. * pi) + log_p - log_det)


	def _calculate_log_det(self, var):
		"""
		Calculate log determinant in log space, to prevent overflow errors.
		args:
			var:            torch.Tensor (1, k, d, d)
		"""
		log_det = torch.empty(size=(self.n_components,)).to(var.device)
		
		for k in range(self.n_components):
			log_det[k] = 2 * torch.log(torch.diagonal(torch.linalg.cholesky(var[0,k]))).sum()

		return log_det.unsqueeze(-1)


	def _e_step(self, x):
		"""
		Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities) that a data point was generated by one of the k mixture components.
		Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
		This is the so-called expectation step of the EM-algorithm.
		args:
			x:              torch.Tensor (n, d) or (n, 1, d)
		returns:
			log_prob_norm:  torch.Tensor (1)
			log_resp:       torch.Tensor (n, k, 1)
		"""
		x = self.check_size(x)

		weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

		log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
		log_resp = weighted_log_prob - log_prob_norm

		return torch.mean(log_prob_norm), log_resp


	def _m_step(self, x, log_resp):
		"""
		From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
		args:
			x:          torch.Tensor (n, d) or (n, 1, d)
			log_resp:   torch.Tensor (n, k, 1)
		returns:
			pi:         torch.Tensor (1, k, 1)
			mu:         torch.Tensor (1, k, d)
			var:        torch.Tensor (1, k, d)
		"""
		x = self.check_size(x)

		resp = torch.exp(log_resp)

		pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
		mu = torch.sum(resp * x, dim=0, keepdim=True) / pi

		if self.covariance_type == "full":
			eps = (torch.eye(self.n_features) * self.eps).to(x.device)
			var = torch.sum((x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2)) * resp.unsqueeze(-1), dim=0,
							keepdim=True) / torch.sum(resp, dim=0, keepdim=True).unsqueeze(-1) + eps

		elif self.covariance_type == "diag":
			x2 = (resp * x * x).sum(0, keepdim=True) / pi
			mu2 = mu * mu
			xmu = (resp * mu * x).sum(0, keepdim=True) / pi
			var = x2 - 2 * xmu + mu2 + self.eps

		pi = pi / x.shape[0]

		return pi, mu, var


	def __em(self, x):
		"""
		Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
		args:
			x:          torch.Tensor (n, 1, d)
		"""
		_, log_resp = self._e_step(x)
		pi, mu, var = self._m_step(x, log_resp)

		self.__update_pi(pi)
		self.__update_mu(mu)
		self.__update_var(var)


	def __score(self, x, as_average=True):
		"""
		Computes the log-likelihood of the data under the model.
		args:
			x:                  torch.Tensor (n, 1, d)
			sum_data:           bool
		returns:
			score:              torch.Tensor (1)
			(or)
			per_sample_score:   torch.Tensor (n)

		"""
		weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
		per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

		if as_average:
			return per_sample_score.mean()
		else:
			return torch.squeeze(per_sample_score)


	def __update_mu(self, mu):
		"""
		Updates mean to the provided value.
		args:
			mu:         torch.FloatTensor
		"""
		assert mu.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

		if mu.size() == (self.n_components, self.n_features):
			self.mu = mu.unsqueeze(0)
		elif mu.size() == (1, self.n_components, self.n_features):
			self.mu.data = mu


	def __update_var(self, var):
		"""
		Updates variance to the provided value.
		args:
			var:        torch.FloatTensor
		"""
		if self.covariance_type == "full":
			assert var.size() in [(self.n_components, self.n_features, self.n_features), (1, self.n_components, self.n_features, self.n_features)], "Input var does not have required tensor dimensions (%i, %i, %i) or (1, %i, %i, %i)" % (self.n_components, self.n_features, self.n_features, self.n_components, self.n_features, self.n_features)

			if var.size() == (self.n_components, self.n_features, self.n_features):
				self.var = var.unsqueeze(0)
			elif var.size() == (1, self.n_components, self.n_features, self.n_features):
				self.var.data = var

		elif self.covariance_type == "diag":
			assert var.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

			if var.size() == (self.n_components, self.n_features):
				self.var = var.unsqueeze(0)
			elif var.size() == (1, self.n_components, self.n_features):
				self.var.data = var


	def __update_pi(self, pi):
		"""
		Updates pi to the provided value.
		args:
			pi:         torch.FloatTensor
		"""
		assert pi.size() in [(1, self.n_components, 1)], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (1, self.n_components, 1)

		self.pi.data = pi


	def get_kmeans_mu(self, x, n_centers, init_times=50, min_delta=1e-3):
		"""
		Find an initial value for the mean. Requires a threshold min_delta for the k-means algorithm to stop iterating.
		The algorithm is repeated init_times often, after which the best centerpoint is returned.
		args:
			x:            torch.FloatTensor (n, d) or (n, 1, d)
			init_times:   init
			min_delta:    int
		"""
		if len(x.size()) == 3:
			x = x.squeeze(1)
		x_min, x_max = x.min(), x.max()
		x = (x - x_min) / (x_max - x_min)
		
		min_cost = np.inf

		for i in range(init_times):
			tmp_center = x[np.random.choice(np.arange(x.shape[0]), size=n_centers, replace=False), ...]
			l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - tmp_center), p=2, dim=2)
			l2_cls = torch.argmin(l2_dis, dim=1)

			cost = 0
			for c in range(n_centers):
				cost += torch.norm(x[l2_cls == c] - tmp_center[c], p=2, dim=1).mean()

			if cost < min_cost:
				min_cost = cost
				center = tmp_center

		delta = np.inf

		while delta > min_delta:
			l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - center), p=2, dim=2)
			l2_cls = torch.argmin(l2_dis, dim=1)
			center_old = center.clone()

			for c in range(n_centers):
				center[c] = x[l2_cls == c].mean(dim=0)

			delta = torch.norm((center_old - center), dim=1).max()

		return (center.unsqueeze(0)*(x_max - x_min) + x_min)