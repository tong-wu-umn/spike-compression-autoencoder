import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_layers, get_padding, L2_dist, SELayer
from .reslib import ResNeXtBottleNeck, BasicResBlock
from .groupnorm import GroupNorm

class spk_vq_vae_resnet(nn.Module):
	def __init__(self, param):
		super(spk_vq_vae_resnet, self).__init__()

		org_dim     = param[0]
		conv1_ch    = param[1]
		conv2_ch    = param[2]
		conv0_ker   = param[3]
		conv1_ker   = param[4]
		conv2_ker   = param[5]
		self.vq_dim = param[6]
		self.vq_num = param[7]
		cardinality = param[8]
		norm_group  = param[9]
		self.vqc    = param[10]
		dropRate    = param[11]

		# ResNeXt encoder
		self.res0 = make_layers(org_dim, conv1_ch, conv0_ker, n_layers=1, cardinality=1, norm_group=norm_group, dropRate=0)
		self.resx = ResNeXtBottleNeck(conv1_ch, conv1_ker, cardinality=cardinality, norm_group=norm_group, dropRate=dropRate)
		self.res2 = nn.Sequential(
			nn.Conv1d(conv1_ch, conv2_ch, conv2_ker, groups=1, padding=get_padding(conv2_ker), bias=False),
			nn.BatchNorm1d(conv2_ch),
			#nn.GroupNorm(num_channels=conv2_ch, num_groups=norm_group),
			nn.Dropout(p=dropRate)
		)
		
		self.deres2 = make_layers(conv2_ch, conv1_ch, conv2_ker, n_layers=1, norm_group=norm_group, decode=True, dropRate=dropRate)
		self.deres1 = BasicResBlock(conv1_ch, conv1_ker, n_layers=2, norm_group=norm_group, decode=True, dropRate=dropRate)
		self.deres0 = nn.ConvTranspose1d(conv1_ch, org_dim, conv0_ker, padding=get_padding(conv0_ker))

		self.ds = nn.MaxPool1d(2)
		self.us = nn.Upsample(scale_factor=2)

		self.embed = nn.Embedding(self.vq_num, self.vq_dim)
		self.init_embed()
		self.embed_freq = np.zeros(self.vq_num, dtype=int)

		# parameter initialization
		# for m in self.modules():
		# 	if isinstance(m, nn.Conv1d):
		# 		nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
		# 	elif isinstance(m, nn.ConvTranspose1d):
		# 		nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
		# 	elif isinstance(m, nn.BatchNorm1d):
		# 		m.weight.data.fill_(1)
		# 		m.bias.data.zero_()
		# 	elif isinstance(m, nn.Embedding):
		# 		initrange = 1.0 / self.vq_dim
		# 		self.embed.weight.data.uniform_(-initrange, initrange)

	def init_embed(self):
		initrange = 1.0 / self.vq_dim
		self.embed.weight.data.uniform_(-initrange, initrange)

	def encoder(self, x):
		out = self.res0(x)
		out = self.resx(out)
		out = self.ds(out)
		out = self.resx(out)
		out = self.ds(out)
		out = self.res2(out)
		return out

	def decoder(self, x):
		out = self.deres2(x)
		out = self.us(out)
		out = self.deres1(out)
		out = self.us(out)
		out = self.deres1(out)
		return self.deres0(out)

	def decomposed_vq(self, Z):
		W1 = self.embed1.weight
		W2 = self.embed2.weight

		z_seg = Z.view(-1, 2, 8)
		j1 = L2_dist(z_seg[:,0:1],W1[None,:]).sum(2).min(1)[1]
		j2 = L2_dist(z_seg[:,1:2],W2[None,:]).sum(2).min(1)[1]

		return torch.cat((W1[j1], W2[j2]), dim=1)

	def forward(self, x):
		h = self.encoder(x)
		org_h = h

		# reshape
		sz = h.size()
		if self.vqc:
			h = h.permute(0, 2, 1)
			h = h.contiguous()
		Z = h.view(-1, self.vq_dim)

		# Sample nearest embedding
		W = self.embed.weight
		j = L2_dist(Z[:,None],W[None,:]).sum(2).min(1)[1]
		W_j = W[j]
		# W_j = self.decomposed_vq(Z)

		if not self.training:
			self.embed_stat(j)

		# Stop gradients
		ze_sg = Z.detach()
		W_j_sg = W_j.detach()

		# inverse reshape
		if self.vqc:
			h = W_j.view(sz[0], sz[2], sz[1])
			h = h.permute(0, 2, 1)
		else:
			h = W_j.view(sz[0], sz[1], sz[2])

		def hook(grad):
			nonlocal org_h
			self.saved_grad = grad
			self.saved_h = org_h
			return grad

		h.requires_grad_()
		h.register_hook(hook)

		commit_loss = L2_dist(Z, W_j_sg).sum(1).mean()
		vq_loss = L2_dist(ze_sg, W_j).sum(1).mean()

		return self.decoder(h), commit_loss, vq_loss

	def bwd(self):
		self.saved_h.backward(self.saved_grad)

	def embed_stat(self, x):
		uni_idx, idx_cnts = np.unique(x.data.cpu().numpy(), return_counts=True)
		self.embed_freq[uni_idx] += idx_cnts

	def embed_reset(self):
		self.embed_freq = np.zeros(self.vq_num, dtype=int)

