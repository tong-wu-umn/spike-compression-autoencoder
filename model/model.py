import numpy as np
import math
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.datasets as datasets
import densenet as ds

from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# %% VAE for spike compression
def make_layers(in_channels, out_channels, kernel_size, n_layers, decode):
	layers = []
	for i in range(n_layers):
		if decode:
			layers.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, padding=get_padding(kernel_size)))
		else:
			layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=get_padding(kernel_size)))
		layers.append(nn.BatchNorm1d(out_channels))
		if i < (n_layers-1):
			layers.append(nn.ReLU(inplace=True))
		in_channels = out_channels

	return nn.Sequential(*layers)

class ResBlock(nn.Module):
	def __init__(self, inplanes, planes, ker, layers, decode=False):
		super(ResBlock, self).__init__()
		
		self.main_conv = make_layers(inplanes, planes, ker, layers, decode=decode)
		self.skip_conv = nn.Conv1d(inplanes, planes, 1, bias=False)

		self.relu = nn.ReLU(inplace=True)		
		self.inplaces = inplanes
		self.planes = planes

	def forward(self, x):
		out = self.main_conv(x)
		if self.inplaces != self.planes:
			residual = self.skip_conv(x)
		else:
			residual = x

		out += residual
		out = self.relu(out)
		return out

def get_padding(ker):
	return int((ker - 1) / 2)

def L2_dist(a, b):
	return ((a - b) ** 2)

def unique_cuda(x):
	"""x is indices of dict corresponding to input batch during forward"""
	uniqe_idx = np.unique(x.cpu().numpy())
	return torch.from_numpy(uniqe_idx).cuda()

# %% VQ function
class vq(torch.autograd.Function):
	def forward(self, x, w):
		i = ((x[:,None] - w[None,:]) ** 2).sum(2).min(1)[1]
		unx_i = unique_cuda(i)

		w_copy = w.clone()
		for _, idx in enumerate(unx_i):
			w_copy[idx] = x[torch.nonzero(i==idx).squeeze()].mean(0)

		return w[i], -2 * (w_copy - w)

	def backward(self, grad_out, grad_w):
		grad_x = grad_out.clone()
		return grad_x, None

class naive_decoder(nn.Module):
	def __init__(self, n_planes):
		super(naive_decoder, self).__init__()
		self.n_planes = n_planes
		self.deconv3 = nn.ConvTranspose1d(n_planes[3], n_planes[2], kernel_size=3, padding=1)
		self.deconv2 = nn.ConvTranspose1d(n_planes[2], n_planes[1], kernel_size=3, padding=1)
		self.deconv1 = nn.ConvTranspose1d(n_planes[1], n_planes[0], kernel_size=3, padding=1)
		self.deconv0 = nn.ConvTranspose1d(n_planes[0], 1, kernel_size=3, padding=1)

		self.bn3 = nn.BatchNorm1d(n_planes[2])
		self.bn2 = nn.BatchNorm1d(n_planes[1])
		self.bn1 = nn.BatchNorm1d(n_planes[0])

	def forward(self, z):
		x = z.view(-1, self.n_planes[3], self.n_planes[4])
		x = F.upsample(x, scale_factor=2, mode='nearest')
		x = F.relu(self.bn3(self.deconv3(x)))
		x = F.upsample(x, scale_factor=2, mode='nearest')
		x = F.relu(self.bn2(self.deconv2(x)))
		x = F.upsample(x, scale_factor=2, mode='nearest')
		x = F.relu(self.bn1(self.deconv1(x)))
		return self.deconv0(x)

# %% spike compression VQ-VAE
class spk_vq_vae_3L(nn.Module):
	def __init__(self, param):
		super(spk_vq_vae_3L, self).__init__()

		org_dim     = param[0]
		conv1_ch    = param[1]
		conv2_ch    = param[2]
		conv3_ch    = param[3]
		conv1_ker   = param[4]
		conv2_ker   = param[5]
		conv3_ker   = param[6]
		conv1_dil   = param[7]
		conv2_dil   = param[8]
		conv3_dil   = param[9]
		self.vq_dim = param[10]
		self.vq_num = param[11]

		self.param = param

		self.conv1 = make_layers(org_dim,  conv1_ch, conv1_ker, get_padding(conv1_ker, conv1_dil), 2, conv1_dil, True, False)
		self.conv2 = make_layers(conv1_ch, conv2_ch, conv2_ker, get_padding(conv2_ker, conv2_dil), 3, conv2_dil, True, False)
		self.conv3 = make_layers(conv2_ch, conv3_ch, conv3_ker, get_padding(conv3_ker, conv3_dil), 3, conv3_dil, True, False)

		self.deconv3 = nn.ConvTranspose1d(conv3_ch, conv2_ch, conv3_ker, padding=get_padding(conv3_ker, conv3_dil), dilation=conv3_dil)
		self.deconv2 = nn.ConvTranspose1d(conv2_ch, conv1_ch, conv2_ker, padding=get_padding(conv2_ker, conv2_dil), dilation=conv2_dil)
		self.deconv1 = nn.ConvTranspose1d(conv1_ch, org_dim,  conv1_ker, padding=get_padding(conv1_ker, conv1_dil), dilation=conv1_dil)

		# self.fc1 = nn.Linear(conv3_ch, self.vq_dim)
		# self.fc2 = nn.Linear(self.vq_dim, conv3_ch)
		self.fc1 = nn.Linear(conv3_ch*8, self.vq_dim)
		self.fc2 = nn.Linear(self.vq_dim, conv3_ch*8)

		self.bn4 = nn.BatchNorm1d(conv3_ch)
		self.bn3 = nn.BatchNorm1d(conv2_ch)
		self.bn2 = nn.BatchNorm1d(conv1_ch)

		self.embed = nn.Embedding(self.vq_num, self.vq_dim)
		self.init_weights()

	def init_weights(self):
		initrange = 1.0 / self.vq_num
		self.embed.weight.data.uniform_(-initrange, initrange)

	def encoder(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		self.l = x.size()[2]
		#return x

		#return x.view(-1, self.vq_dim)

		#NCL -> NLC
		# x = x.permute(0, 2, 1)
		# x = x.contiguous()
		# return F.relu(self.fc1(x.view(-1, self.param[3])))

		# NCL -> N(C*L) -> N * vq_dim
		return F.tanh(self.fc1(x.view(len(x), -1)))

	def vq_func(self, x):
		return vq()(x, self.embed.weight)

	def decoder(self, z):
		# x = F.relu(self.fc2(z))
		# x = x.view(-1, self.l, self.param[3])
		# x = x.permute(0, 2, 1)

		x = F.tanh(self.fc2(z))
		x = x.view(-1, self.param[3], self.l)

		x = F.upsample(x, scale_factor=2, mode='nearest')
		x = F.relu(self.bn3(self.deconv3(x)))
		x = F.upsample(x, scale_factor=2, mode='nearest')
		x = F.relu(self.bn2(self.deconv2(x)))
		x = F.upsample(x, scale_factor=2, mode='nearest')
		
		return self.deconv1(x)

	def forward(self, x):
		z_e = self.encoder(x)

		# Sample nearest embedding
		W = self.embed.weight
		j = L2_dist(z_e[:,None],W[None,:]).sum(2).min(1)[1]
		W_j = W[j]
		if not self.training:
			self.embed_stat(j)

		# Stop gradients
		ze_sg = z_e.detach()
		W_j_sg = W_j.detach()

		recon_x = self.decoder(W_j)

		#return recon_x, L2_dist(z_e, zq_sg).sum(1).mean()
		return recon_x, L2_dist(z_e, W_j_sg).sum(1).mean(), L2_dist(ze_sg, W_j).sum(1).mean()
		#return recon_x

	def embed_stat(self, x):
		uni_idx, idx_cnts = np.unique(x.data.cpu().numpy(), return_counts=True)
		self.embed_freq[uni_idx] += idx_cnts

	def embed_reset(self):
		self.embed_freq = np.zeros(self.vq_num, dtype=int)

class spk_vq_vae_resnet(nn.Module):
	def __init__(self, param):
		super(spk_vq_vae_resnet, self).__init__()

		org_dim     = param[0]
		conv1_ch    = param[1]
		conv2_ch    = param[2]
		conv3_ch    = param[3]
		conv0_ker   = param[4]
		conv1_ker   = param[5]
		conv2_ker   = param[6]
		conv3_ker   = param[7]
		self.vq_dim = param[8]
		self.vq_num = param[9]

		self.conv3ch = param[3]

		self.conv = make_layers(org_dim, conv1_ch, conv0_ker, get_padding(conv0_ker, 1), 1, 1)
		self.res1 = residual_layers(conv1_ch, conv1_ch, conv1_ker, get_padding(conv1_ker, 1), 2)
		self.res2 = residual_layers(conv1_ch, conv2_ch, conv2_ker, get_padding(conv2_ker, 1), 3)
		self.res3 = residual_layers(conv2_ch, conv3_ch, conv3_ker, get_padding(conv3_ker, 1), 3)

		self.deres3 = inverse_residual_layers(conv3_ch, conv2_ch, conv2_ker, get_padding(conv2_ker, 1), 4)
		self.deres2 = inverse_residual_layers(conv2_ch, conv1_ch, conv1_ker, get_padding(conv1_ker, 1), 4)
		self.deres1 = inverse_residual_layers(conv1_ch, conv1_ch, conv0_ker, get_padding(conv0_ker, 1), 3)
		self.deconv = nn.ConvTranspose1d(conv1_ch, org_dim, conv0_ker, padding=get_padding(conv0_ker, 1))

		# 1-dim conv layers to match channels (dimensions)
		self.conv_ch_1to2 = nn.Conv1d(conv1_ch, conv2_ch, kernel_size=1)
		self.conv_ch_2to3 = nn.Conv1d(conv2_ch, conv3_ch, kernel_size=1)
		self.conv_ch_3to2 = nn.Conv1d(conv3_ch, conv2_ch, kernel_size=1)
		self.conv_ch_2to1 = nn.Conv1d(conv2_ch, conv1_ch, kernel_size=1)

		# 1-dim conv layers to match size
		self.conv_ds1 = nn.Conv1d(conv1_ch, conv1_ch, kernel_size=2, stride=2)
		self.conv_ds2 = nn.Conv1d(conv2_ch, conv2_ch, kernel_size=2, stride=2)
		self.conv_ds3 = nn.Conv1d(conv3_ch, conv3_ch, kernel_size=2, stride=2)

		self.conv_us3 = nn.ConvTranspose1d(conv3_ch, conv3_ch, kernel_size=2, stride=2, output_padding=0)
		self.conv_us2 = nn.ConvTranspose1d(conv2_ch, conv2_ch, kernel_size=2, stride=2, output_padding=0)
		self.conv_us1 = nn.ConvTranspose1d(conv1_ch, conv1_ch, kernel_size=2, stride=2, output_padding=0)

		self.embed = nn.Embedding(self.vq_num, self.vq_dim)
		self.init_embed()
		self.embed_freq = np.zeros(self.vq_num, dtype=int)

	def init_embed(self):
		initrange = 1.0 / self.vq_num
		self.embed.weight.data.uniform_(-initrange, initrange)

	def encoder(self, x):
		x = self.conv(x)
		res1_out = self.res1(x)
		res1_out = torch.add(x, 1, res1_out)
		
		res2_inp = self.conv_ds1(res1_out)
		res2_out = self.res2(res2_inp)
		res2_out = torch.add(self.conv_ch_1to2(res2_inp), 1, res2_out)

		res3_inp = self.conv_ds2(res2_out)
		res3_out = self.res3(res3_inp)
		res3_out = torch.add(self.conv_ch_2to3(res3_inp), 1, res3_out)
		x = self.conv_ds3(res3_out)
		self.l = x.size()[2]
		
		return x.view(-1, self.vq_dim)

	def vq_func(self, x):
		return vq()(x, self.embed.weight)

	def decoder(self, z):
		x = z.view(-1, self.conv3ch, self.vq_dim)
		x = self.conv_us3(x)

		res3_out = self.deres3(x)
		res3_out = torch.add(self.conv_ch_3to2(x), 1, res3_out)

		res2_inp = self.conv_us2(res3_out)
		res2_out = self.deres2(res2_inp)
		res2_out = torch.add(self.conv_ch_2to1(res2_inp), 1, res2_out)

		res1_inp = self.conv_us1(res2_out)
		res1_out = self.deres1(res1_inp)
		res1_out = torch.add(res1_inp, 1, res1_out)
		return self.deconv(res1_out)

	def forward(self, x):
		z_e = self.encoder(x)

		# Sample nearest embedding
		W = self.embed.weight
		j = L2_dist(z_e[:,None],W[None,:]).sum(2).min(1)[1]
		W_j = W[j]
		if not self.training:
			self.embed_stat(j)

		# Stop gradients
		ze_sg = z_e.detach()
		W_j_sg = W_j.detach()
		
		recon_x = self.decoder(W_j)
		return recon_x, L2_dist(z_e, W_j_sg).sum(1).mean(), L2_dist(ze_sg, W_j).sum(1).mean()

	def embed_stat(self, x):
		uni_idx, idx_cnts = np.unique(x.data.cpu().numpy(), return_counts=True)
		self.embed_freq[uni_idx] += idx_cnts

	def embed_reset(self):
		self.embed_freq = np.zeros(self.vq_num, dtype=int)

class spk_vq_vae_2L_resnet(nn.Module):
	def __init__(self, param):
		super(spk_vq_vae_2L_resnet, self).__init__()

		org_dim     = param[0]
		conv1_ch    = param[1]
		conv2_ch    = param[2]
		conv0_ker   = param[3]
		conv1_ker   = param[4]
		conv2_ker   = param[5]
		self.vq_dim = param[6]
		self.vq_num = param[7]
		self.param = param

		self.conv = make_layers(org_dim, conv1_ch, conv0_ker, 1)
		self.res1 = make_layers(conv1_ch, conv1_ch, conv1_ker, 3)
		self.res2 = make_layers(conv1_ch, conv2_ch, conv2_ker, 1)

		self.deres2 = make_layers(conv2_ch, conv1_ch, conv2_ker, 1, decode=True)
		self.deres1 = make_layers(conv1_ch, conv1_ch, conv1_ker, 3, decode=True)
		self.deconv = nn.ConvTranspose1d(conv1_ch, org_dim, conv0_ker, padding=get_padding(conv0_ker))

		# 1-dim conv layers to match channels (dimensions)
		self.conv_ch_1to2 = nn.Conv1d(conv1_ch, conv2_ch, kernel_size=1, bias=False)
		self.conv_ch_2to1 = nn.Conv1d(conv2_ch, conv1_ch, kernel_size=1, bias=False)

		# 1-dim conv layers to match size
		self.conv_ds1 = nn.Conv1d(conv1_ch, conv1_ch, kernel_size=2, stride=2, padding=0)
		self.conv_ds2 = nn.Conv1d(conv2_ch, conv2_ch, kernel_size=2, stride=2, padding=0)

		self.conv_us2 = nn.ConvTranspose1d(conv2_ch, conv2_ch, kernel_size=2, stride=2, padding=0, output_padding=0)
		self.conv_us1 = nn.ConvTranspose1d(conv1_ch, conv1_ch, kernel_size=2, stride=2, padding=0, output_padding=0)

		self.embed = nn.Embedding(self.vq_num, self.vq_dim)
		self.init_weights()
		self.embed_freq = np.zeros(self.vq_num, dtype=int)
	
	def init_weights(self):
		initrange = 1.0 / self.vq_num
		self.embed.weight.data.uniform_(-initrange, initrange)

	def encoder(self, x):
		x = self.conv(x)
		res1_out = self.res1(x)
		res1_out = torch.add(x, 1, res1_out)
		#res1_out = F.relu(res1_out, inplace=True)
		
		res2_inp = self.conv_ds1(res1_out)
		res2_out = self.res2(res2_inp)
		res2_out = torch.add(self.conv_ch_1to2(res2_inp), 1, res2_out)
		#res2_out = F.relu(res2_out, inplace=True)

		x = self.conv_ds2(res2_out)
		return x

	# def vq_func(self, x):
	# 	return vq()(x, self.embed.weight)

	def decoder(self, z):
		res2_inp = self.conv_us2(z)
		res2_out = self.deres2(res2_inp)
		res2_out = torch.add(self.conv_ch_2to1(res2_inp), 1, res2_out)
		#res2_out = F.relu(res2_out, inplace=True)

		res1_inp = self.conv_us1(res2_out)
		res1_out = self.deres1(res1_inp)
		res1_out = torch.add(res1_inp, 1, res1_out)
		#res1_out = F.relu(res1_out, inplace=True)

		x = res1_out
		return self.deconv(x)

	def forward(self, x):
		h = self.encoder(x)
		org_h = h

		# reshape
		sz = h.size()
		h = h.contiguous()
		Z = h.view(-1, self.vq_dim)

		# Sample nearest embedding
		W = self.embed.weight
		j = L2_dist(Z[:,None],W[None,:]).sum(2).min(1)[1]
		W_j = W[j]
		if not self.training:
			self.embed_stat(j)

		# Stop gradients
		ze_sg = Z.detach()
		W_j_sg = W_j.detach()

		# inverse reshape
		h = W_j.view(sz[0], sz[1], sz[2])

		def hook(grad):
			nonlocal org_h
			self.saved_grad = grad
			self.saved_h = org_h
			return grad

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

class spk_vq_vae_2L_lstm_resnet(nn.Module):
	def __init__(self, param):
		super(spk_vq_vae_2L_lstm_resnet, self).__init__()

		org_dim     = param[0]
		conv1_ch    = param[1]
		conv2_ch    = param[2]
		conv0_ker   = param[3]
		conv1_ker   = param[4]
		conv2_ker   = param[5]
		self.vq_dim = param[6]
		self.vq_num = param[7]

		# self.conv2ch = param[2]
		self.param = param

		self.conv = make_layers(org_dim, conv1_ch, conv0_ker, get_padding(conv0_ker, 1), 1, dropout=False)
		self.res1 = residual_layers(conv1_ch, conv1_ch, conv1_ker, get_padding(conv1_ker, 1), 2)
		self.res2 = residual_layers(conv1_ch, conv2_ch, conv2_ker, get_padding(conv2_ker, 1), 3)

		self.deres2 = inverse_residual_layers(conv2_ch, conv1_ch, conv1_ker, get_padding(conv1_ker, 1), 3)
		self.deres1 = inverse_residual_layers(conv1_ch, conv1_ch, conv0_ker, get_padding(conv0_ker, 1), 2)
		self.deconv = nn.ConvTranspose1d(conv1_ch, org_dim, conv0_ker, padding=get_padding(conv0_ker, 1))

		# 1-dim conv layers to match channels (dimensions)
		self.conv_ch_1to2 = nn.Conv1d(conv1_ch, conv2_ch, kernel_size=1, bias=True)
		self.conv_ch_2to1 = nn.Conv1d(conv2_ch, conv1_ch, kernel_size=1, bias=True)

		# 1-dim conv layers to match size
		self.conv_ds1 = nn.Conv1d(conv1_ch, conv1_ch, kernel_size=2, stride=2, padding=0)
		self.conv_ds2 = nn.Conv1d(conv2_ch, conv2_ch, kernel_size=2, stride=2, padding=0)

		self.conv_us2 = nn.ConvTranspose1d(conv2_ch, conv2_ch, kernel_size=2, stride=2, padding=0, output_padding=0)
		self.conv_us1 = nn.ConvTranspose1d(conv1_ch, conv1_ch, kernel_size=2, stride=2, padding=0, output_padding=0)

		self.embed = nn.Embedding(self.vq_num, self.vq_dim)
		self.init_weights()
		self.embed_freq = np.zeros(self.vq_num, dtype=int)

		#self.naive_decoder = self._make_layer(naive_decoder, [conv1_ch, conv1_ch, conv2_ch, conv3_ch])
		#self.naive_decoder = naive_decoder([conv1_ch, conv1_ch, conv2_ch, conv3_ch, self.vq_dim])

		self.lstm1 = nn.LSTM(conv2_ch, conv2_ch, 1)
		self.lstm2 = nn.LSTM(conv2_ch, conv2_ch, 1)
	
	def init_weights(self):
		initrange = 1.0 / self.vq_num
		self.embed.weight.data.uniform_(-initrange, initrange)

	def encoder(self, x, hc0):
		x = self.conv(x)
		res1_out = self.res1(x)
		res1_out = torch.add(x, 1, res1_out)
		
		res2_inp = self.conv_ds1(res1_out)
		res2_out = self.res2(res2_inp)
		res2_out = torch.add(self.conv_ch_1to2(res2_inp), 1, res2_out)
		x = self.conv_ds2(res2_out)

		# in LSTM, NCL -> LNC
		x, _ = self.lstm1(x.permute(2, 0, 1), hc0)
		x = x.permute(1, 2, 0)
		x = x.contiguous()

		self.l = x.size()[2]
		return x.view(-1, self.vq_dim)

	def vq_func(self, x):
		return vq()(x, self.embed.weight)

	def decoder(self, z, hc0):
		x = z.view(-1, self.param[2], self.vq_dim)
		x, _ = self.lstm2(x.permute(2, 0, 1), hc0)
		x = x.permute(1, 2, 0)

		res2_inp = self.conv_us2(x)
		res2_out = self.deres2(res2_inp)
		res2_out = torch.add(self.conv_ch_2to1(res2_inp), 1, res2_out)

		res1_inp = self.conv_us1(res2_out)
		res1_out = self.deres1(res1_inp)
		res1_out = torch.add(res1_inp, 1, res1_out)
		return self.deconv(res1_out)

	def forward(self, x):
		# init hidden variables for LSTM
		h0 = Variable(torch.zeros(1, len(x), self.param[2])).cuda()
		c0 = Variable(torch.zeros(1, len(x), self.param[2])).cuda()
		hidden_forward, hidden_backward = (h0, c0), (h0, c0)

		# network
		z_e = self.encoder(x, hidden_forward)

		# Sample nearest embedding
		W = self.embed.weight
		j = L2_dist(z_e[:,None],W[None,:]).sum(2).min(1)[1]
		W_j = W[j]
		if not self.training:
			self.embed_stat(j)

		# Stop gradients
		ze_sg = z_e.detach()
		W_j_sg = W_j.detach()

		recon_x = self.decoder(W_j, hidden_backward)
		return recon_x, L2_dist(z_e, W_j_sg).sum(1).mean(), L2_dist(ze_sg, W_j).sum(1).mean()

	def embed_stat(self, x):
		uni_idx, idx_cnts = np.unique(x.data.cpu().numpy(), return_counts=True)
		self.embed_freq[uni_idx] += idx_cnts

	def embed_reset(self):
		self.embed_freq = np.zeros(self.vq_num, dtype=int)

class spk_vq_vae_densenet(nn.Module):
	def __init__(self, growth_rate=12, reduction=0.5, bottleneck=True, dropRate=0.0):
		super(spk_vq_vae_densenet, self).__init__()
		in_planes = 2 * growth_rate
		n = 10
		if bottleneck == True:
			block = ds.BottleneckBlock
		else:
			block = ds.BasicBlock

		# 1st encoder conv before any dense block
		self.conv1 = nn.Conv1d(1, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
		blk0_planes = in_planes

		# 1st encoder block
		self.block1 = ds.DenseBlock(n, in_planes, growth_rate, block, dropRate)
		in_planes = int(in_planes+n*growth_rate)
		self.trans1 = ds.TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), downsample=True, dropRate=dropRate)
		in_planes = int(math.floor(in_planes*reduction))
		blk1_planes = in_planes

		# 2nd encoder block
		self.block2 = ds.DenseBlock(n, in_planes, growth_rate, block, dropRate)
		in_planes = int(in_planes+n*growth_rate)
		self.trans2 = ds.TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), downsample=True, dropRate=dropRate)
		in_planes = int(math.floor(in_planes*reduction))
		blk2_planes = in_planes

		# 3rd encoder block
		self.block3 = ds.DenseBlock(n, in_planes, growth_rate, block, dropRate)
		in_planes = int(in_planes+n*growth_rate)
		self.trans3 = ds.TransitionBlock(in_planes, int(math.floor(in_planes)), downsample=True, dropRate=dropRate)
		#in_planes = int(math.floor(in_planes*reduction))
		blk3_planes = in_planes
		self.blk3_planes = blk3_planes

		# parameters for embedding
		self.vq_num = 128
		self.vq_dim = 8
		self.embed = nn.Embedding(self.vq_num, self.vq_dim)
		self.init_weights()
		self.embed_freq = np.zeros(self.vq_num, dtype=int)

		# naive decoder
		self.naive_decoder = naive_decoder([blk0_planes, blk1_planes, blk2_planes, blk3_planes, self.vq_dim])

		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				n = m.kernel_size[0] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()

	def init_weights(self):
		initrange = 1.0 / self.vq_num
		self.embed.weight.data.uniform_(-initrange, initrange)
		#self.embed.weight.data.normal_()

	def encoder(self, x):
		out = self.conv1(x)
		out = self.trans1(self.block1(out))
		out = self.trans2(self.block2(out))
		out = self.trans3(self.block3(out))
		return out.view(-1, self.vq_dim)

	def forward(self, x):
		z_e = self.encoder(x)

		# Sample nearest embedding
		W = self.embed.weight
		j = L2_dist(z_e[:,None],W[None,:]).sum(2).min(1)[1]
		W_j = W[j]
		if not self.training:
			self.embed_stat(j)

		# Stop gradients
		ze_sg = z_e.detach()
		W_j_sg = W_j.detach()
		
		recon_x = self.naive_decoder(W_j)
		return recon_x, L2_dist(z_e, W_j_sg).sum(1).mean(), L2_dist(ze_sg, W_j).sum(1).mean()

	def embed_stat(self, x):
		uni_idx, idx_cnts = np.unique(x.data.cpu().numpy(), return_counts=True)
		self.embed_freq[uni_idx] += idx_cnts

	def embed_reset(self):
		self.embed_freq = np.zeros(self.vq_num, dtype=int)

class spk_vq_vae_lstm(nn.Module):
	def __init__(self, param):
		super(spk_vq_vae_lstm, self).__init__()

		org_dim     = param[0]
		conv1_ch    = param[1]
		conv2_ch    = param[2]
		conv3_ch    = param[3]
		conv1_ker   = param[4]
		conv2_ker   = param[5]
		conv3_ker   = param[6]
		conv1_dil   = param[7]
		conv2_dil   = param[8]
		conv3_dil   = param[9]
		lstm_hidden = param[10]
		self.vq_dim = param[11]
		self.vq_num = param[12]

		self.param = param

		self.conv1 = make_layers(org_dim,  conv1_ch, conv1_ker, get_padding(conv1_ker, conv1_dil), 2, conv1_dil, True, False)
		self.conv2 = make_layers(conv1_ch, conv2_ch, conv2_ker, get_padding(conv2_ker, conv2_dil), 3, conv2_dil, True, False)
		#self.conv3 = make_layers(conv2_ch, conv3_ch, conv3_ker, get_padding(conv3_ker, conv3_dil), 2, conv3_dil, True, True)

		self.lstm1 = nn.LSTM(conv2_ch, lstm_hidden, 1)
		self.lstm2 = nn.LSTM(lstm_hidden, conv2_ch, 1)

		#self.deconv3 = nn.ConvTranspose1d(conv3_ch, conv2_ch, conv3_ker, padding=get_padding(conv3_ker, conv3_dil), dilation=conv3_dil)
		self.deconv2 = nn.ConvTranspose1d(conv2_ch, conv1_ch, conv2_ker, padding=get_padding(conv2_ker, conv2_dil), dilation=conv2_dil)
		self.deconv1 = nn.ConvTranspose1d(conv1_ch, org_dim,  conv1_ker, padding=get_padding(conv1_ker, conv1_dil), dilation=conv1_dil)

		# self.fc1 = nn.Linear(conv3_ch, self.vq_dim)
		# self.fc2 = nn.Linear(self.vq_dim, conv3_ch)

		self.bn4 = nn.BatchNorm1d(conv3_ch)
		self.bn3 = nn.BatchNorm1d(conv2_ch)
		self.bn2 = nn.BatchNorm1d(conv1_ch)
		#self.bn1 = nn.BatchNorm1d(conv1_ch)

		self.embed = nn.Embedding(self.vq_num, self.vq_dim)
		self.init_weights()

	def init_weights(self):
		initrange = 1.0 / self.vq_num
		self.embed.weight.data.uniform_(-initrange, initrange)

	def encoder(self, x, hc0):
		x = self.conv1(x)
		x = self.conv2(x)

		# in LSTM, NCL -> LNC
		x, _ = self.lstm1(x.permute(2, 0, 1), hc0)
		# LNC -> NCL -> (N*C)L
		x = x.permute(1, 2, 0)
		x = x.contiguous()
		return x.view(-1, self.vq_dim)

	def vq_func(self, x):
		return vq()(x, self.embed.weight)

	def decoder(self, z, hc0):
		x = z.view(-1, self.param[10], self.vq_dim)
		x, _ = self.lstm2(x.permute(2, 0, 1), hc0)
		x = x.permute(1, 2, 0)
		x = F.upsample(x, scale_factor=2, mode='nearest')
		x = F.relu(self.bn2(self.deconv2(x)))
		x = F.upsample(x, scale_factor=2, mode='nearest')
		return self.deconv1(x)

	def forward(self, x):
		h0 = Variable(torch.zeros(1, len(x), self.param[10])).cuda()
		c0 = Variable(torch.zeros(1, len(x), self.param[10])).cuda()
		hidden_forward = (h0, c0)

		h1 = Variable(torch.zeros(1, len(x), self.param[2])).cuda()
		c1 = Variable(torch.zeros(1, len(x), self.param[2])).cuda()
		hidden_backward = (h1, c1)

		z_e = self.encoder(x, hidden_forward)
		z_q, self.dict_grad = self.vq_func(z_e)
		zq_sg = z_q.detach()
		recon_x = self.decoder(z_q, hidden_backward)
		return recon_x, L2_dist(z_e, zq_sg).sum(1).mean()

	def update_dict(self, lr):
		self.embed.weight.data -= lr * self.dict_grad.data

class spk_vq_vae_4L(nn.Module):
	def __init__(self, param):
		super(spk_vq_vae_4L, self).__init__()

		org_dim     = param[0]
		conv1_ch    = param[1]
		conv2_ch    = param[2]
		conv3_ch    = param[3]
		conv4_ch	= param[4]
		conv1_ker   = param[5]
		conv2_ker   = param[6]
		conv3_ker   = param[7]
		conv4_ker   = param[8]
		conv1_dil   = param[9]
		conv2_dil   = param[10]
		conv3_dil   = param[11]
		conv4_dil   = param[12]
		self.vq_dim = param[13]
		self.vq_num = param[14]

		self.conv4ch = param[4]

		self.conv1 = make_layers(org_dim,  conv1_ch, conv1_ker, get_padding(conv1_ker, conv1_dil), 2, conv1_dil, True, False)
		self.conv2 = make_layers(conv1_ch, conv2_ch, conv2_ker, get_padding(conv2_ker, conv2_dil), 3, conv2_dil, True, False)
		self.conv3 = make_layers(conv2_ch, conv3_ch, conv3_ker, get_padding(conv3_ker, conv3_dil), 3, conv3_dil, True, False)
		self.conv4 = make_layers(conv3_ch, conv4_ch, conv4_ker, get_padding(conv4_ker, conv4_dil), 3, conv4_dil, True, False)

		self.deconv4 = nn.ConvTranspose1d(conv4_ch, conv3_ch, conv4_ker, padding=get_padding(conv4_ker, conv4_dil), dilation=conv4_dil)
		self.deconv3 = nn.ConvTranspose1d(conv3_ch, conv2_ch, conv3_ker, padding=get_padding(conv3_ker, conv3_dil), dilation=conv3_dil)
		self.deconv2 = nn.ConvTranspose1d(conv2_ch, conv1_ch, conv2_ker, padding=get_padding(conv2_ker, conv2_dil), dilation=conv2_dil)
		self.deconv1 = nn.ConvTranspose1d(conv1_ch, org_dim,  conv1_ker, padding=get_padding(conv1_ker, conv1_dil), dilation=conv1_dil)

		self.fc1 = nn.Linear(conv4_ch, self.vq_dim)
		self.fc2 = nn.Linear(self.vq_dim, conv4_ch)

		self.bn4 = nn.BatchNorm1d(conv4_ch)
		self.bn3 = nn.BatchNorm1d(conv3_ch)
		self.bn2 = nn.BatchNorm1d(conv2_ch)
		self.bn1 = nn.BatchNorm1d(conv1_ch)

		self.embed = nn.Embedding(self.vq_num, self.vq_dim)
		self.init_weights()

	def init_weights(self):
		initrange = 1.0 / self.vq_num
		self.embed.weight.data.uniform_(-initrange, initrange)

	def encoder(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		self.l = x.size()[2]
		return x
		# return x.view(-1, self.vq_dim)

		# NCL -> NLC
		# x = x.permute(0, 2, 1)
		# x = x.contiguous()
		# return F.tanh(self.fc1(x.view(-1, self.conv4ch)))

	def vq_func(self, x):
		return vq()(x, self.embed.weight)

	def decoder(self, z):
		# x = F.relu(self.fc2(z))
		# x = x.view(-1, self.l, self.conv4ch)
		# x = x.permute(0, 2, 1)

		#x = z.view(-1, self.conv4ch, self.vq_dim)

		x = z
		x = F.upsample(x, scale_factor=2)
		x = F.relu(self.bn3(self.deconv4(x)))
		x = F.upsample(x, scale_factor=2)
		x = F.relu(self.bn2(self.deconv3(x)))
		x = F.upsample(x, scale_factor=2)
		x = F.relu(self.bn1(self.deconv2(x)))
		x = F.upsample(x, scale_factor=2)
		return self.deconv1(x)

	def forward(self, x):
		z_e = self.encoder(x)
		# z_q, self.dict_grad = self.vq_func(z_e)
		# zq_sg = z_q.detach()
		recon_x = self.decoder(z_e)
		return recon_x, L2_dist(z_e, z_e).sum(1).mean()

	def update_dict(self, lr):
		self.embed.weight.data -= lr * self.dict_grad.data

class spk_vae(nn.Module):
	def __init__(self, param):
		super(spk_vae, self).__init__()

		org_dim     = param[0]
		conv1_ch    = param[1]
		conv2_ch    = param[2]
		conv3_ch    = param[3]
		conv1_ker   = param[4]
		conv2_ker   = param[5]
		conv3_ker   = param[6]
		self.vq_dim = param[7]
		self.vq_num = param[8]

		self.conv1 = make_layers(org_dim,  conv1_ch, conv1_ker, get_padding(conv1_ker), 2)
		self.conv2 = make_layers(conv1_ch, conv2_ch, conv2_ker, get_padding(conv2_ker), 2)
		self.conv3 = make_layers(conv2_ch, conv3_ch, conv3_ker, get_padding(conv3_ker), 2)

		self.deconv3 = nn.ConvTranspose1d(conv3_ch, conv2_ch, conv3_ker, padding=get_padding(conv3_ker))
		self.deconv2 = nn.ConvTranspose1d(conv2_ch, conv1_ch, conv2_ker, padding=get_padding(conv2_ker))
		self.deconv1 = nn.ConvTranspose1d(conv1_ch, org_dim,  conv1_ker, padding=get_padding(conv1_ker))

		self.fc21 = nn.Linear(conv3_ch*8, self.vq_dim)
		self.fc22 = nn.Linear(conv3_ch*8, self.vq_dim)
		self.fc3  = nn.Linear(self.vq_dim, conv3_ch*8)

		self.bn4 = nn.BatchNorm1d(conv3_ch * 8)
		self.bn3 = nn.BatchNorm1d(conv3_ch)
		self.bn2 = nn.BatchNorm1d(conv2_ch)
		self.bn1 = nn.BatchNorm1d(conv1_ch)

	def encoder(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = x.view(x.size(0), -1)
		return self.fc21(x), self.fc22(x)

	def reparameterize(self, mu, logvar):
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = Variable(std.data.new(std.size()).normal_())
			return eps.mul(std).add_(mu)
		else:
			return mu

	def decoder(self, z):
		x = F.relu(self.bn4(self.fc3(z)))
		x = F.upsample(x = x.view(x.size(0), conv3_ch, -1), size=[x.size(0), x.size(1), x.size(2)*2])
		x = F.relu(self.bn3(self.deconv3(x)))
		x = F.upsample(x, size=[x.size(0), x.size(1), x.size(2)*2])
		x = F.relu(self.bn2(self.deconv2(x)))
		x = F.upsample(x, size=[x.size(0), x.size(1), x.size(2)*2])
		return self.bn1(self.deconv1(x))

	def forward(self, x):
		mu, logvar = self.encoder(x)
		z = self.reparameterize(mu, logvar)
		return self.decoder(z), mu, logvar

class SpikeDataset(Dataset):
	"""Dataset wrapping of spikes."""
	def __init__(self, args, is_train=True):
		self.args = args
		trainmat = sio.loadmat(self.args.train_data_path)
		testmat  = sio.loadmat(self.args.test_data_path)

		traindata = trainmat['all_spks'].astype(np.float32)
		testdata = testmat['all_spks'].astype(np.float32)

		if traindata.shape[0] < traindata.shape[1]:
			traindata = np.transpose(traindata, (1, 0))
		if testdata.shape[0] < testdata.shape[1]:
			testdata = np.transpose(testdata, (1, 0))

		np.random.seed(self.args.seed)
		train_rand_idx = np.random.permutation(len(traindata))
		test_rand_idx = np.random.permutation(len(testdata))

		self.train_data = traindata[train_rand_idx[:round(len(traindata) * self.args.train_ratio)]]
		self.train_mean = self.train_data.mean(0)
		self.train_std = self.train_data.std(0)
		self.train_norm = (self.train_data - self.train_mean) / self.train_std
		
		self.test_data  = testdata[test_rand_idx[round(len(testdata) * (1-self.args.test_ratio)):]]
		self.test_mean = self.test_data.mean(0)
		self.test_std = self.test_data.std(0)

		if self.args.val_norm:
			self.test_norm = (self.test_data - self.test_mean) / self.test_std
		else:
			self.test_norm = (self.test_data - self.train_mean) / self.train_std

		if is_train:
			self.spikes = self.train_norm[:,None]
		else:
			self.spikes = self.test_norm[:,None]

	def __getitem__(self, index):
		spk = self.spikes[index]
		return torch.from_numpy(spk)

	def __len__(self):
		return len(self.spikes)

class Helper():
	def __init__(self, args):
		self.args = args

	def create_data_loaders(self):
		self.train_set = SpikeDataset(self.args, is_train=True)
		self.test_set = SpikeDataset(self.args, is_train=False)

		# self.train_mean = train_set.train_mean
		# self.train_std = train_set.train_std
		# self.test_data = test_set.spikes

		train_loader = DataLoader(
			dataset=self.train_set,
			batch_size=self.args.batch_size,
			shuffle=True,
			num_workers=4
		)

		test_loader = DataLoader(
			dataset=self.test_set,
			batch_size=self.args.test_batch_size,
			shuffle=True,
			num_workers=4
		)
		
		return train_loader, test_loader

	def param_for_recon(self):
		if self.args.val_norm:
			return self.train_set.test_mean, self.train_set.test_std, self.test_set.spikes
		else:
			return self.train_set.train_mean, self.train_set.train_std, self.test_set.spikes

	def init_embedding(self, vq_num, vq_dim):
		pca = PCA(n_components = vq_dim)
		pca.fit(self.train_set.train_norm)
		train_data_pca = pca.transform(self.train_set.train_norm)
		kmeans = KMeans(n_clusters=vq_num).fit(train_data_pca)
		
		return torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).cuda()