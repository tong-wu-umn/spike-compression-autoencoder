import numpy as np
import math
import scipy.io as sio
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

def get_padding(ker):
	return int((ker - 1) / 2)

def L2_dist(a, b):
	return ((a - b) ** 2)

def make_layers(in_channels, out_channels, kernel_size, n_layers=1, cardinality=1, dropRate=0, decode=False):
	layers = []
	for i in range(n_layers):
		if decode:
			layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, groups=cardinality, padding=get_padding(kernel_size), bias=False))
		else:
			layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, groups=cardinality, padding=get_padding(kernel_size), bias=False))
		layers.append(nn.BatchNorm2d(out_channels))
		layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Dropout2d(p=dropRate))
		in_channels = out_channels
	
	return nn.Sequential(*layers)

class SpikeDataset(Dataset):
	"""Dataset wrapping of spikes."""
	def __init__(self, args, is_train=True):
		self.args = args
		trainmat = sio.loadmat(self.args.train_data_path)
		testmat  = sio.loadmat(self.args.test_data_path)

		traindata = trainmat['total_spks'].astype(np.float32)
		testdata = testmat['total_spks'].astype(np.float32)

		np.random.seed(self.args.seed)
		train_rand_idx = np.random.permutation(len(traindata))
		np.random.seed(self.args.seed)
		test_rand_idx = np.random.permutation(len(testdata))

		self.train_data = traindata[train_rand_idx[:round(len(traindata) * self.args.train_ratio)]]
		self.train_mean, self.train_std = self.train_data.mean(), self.train_data.std()
		self.train_norm = (self.train_data - self.train_mean) / self.train_std
		
		self.test_data  = testdata[test_rand_idx[round(len(testdata) * (1-self.args.test_ratio)):]]
		self.test_mean, self.test_std = self.test_data.mean(), self.test_data.std()

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

class SpikeDataset_3d(Dataset):
	"""Dataset wrapping of spikes."""
	def __init__(self, args, is_train=True):
		self.args = args
		trainmat = sio.loadmat(self.args.train_data_path)
		testmat  = sio.loadmat(self.args.test_data_path)

		traindata = trainmat['total_spks'].astype(np.float32)
		testdata = testmat['total_spks'].astype(np.float32)

		# traindata = traindata[:, :self.args.spk_ch]
		# testdata = testdata[:, :self.args.spk_ch]

		np.random.seed(self.args.seed)
		train_rand_idx = np.random.permutation(len(traindata))
		np.random.seed(self.args.seed)
		test_rand_idx = np.random.permutation(len(testdata))

		self.train_data = traindata[train_rand_idx[:round(len(traindata) * self.args.train_ratio)]]
		sz = self.train_data.shape
		tmp = np.reshape(self.train_data, (sz[0]*sz[1], sz[2]))
		self.train_mean, self.train_std = tmp.mean(0, keepdims=True)[None,:], tmp.std(0, keepdims=True)[None,:]
		self.train_norm = (self.train_data - self.train_mean) / self.train_std
		
		self.test_data  = testdata[test_rand_idx[round(len(testdata) * (1-self.args.test_ratio)):]]
		sz = self.test_data.shape
		tmp = np.reshape(self.test_data, (sz[0]*sz[1], sz[2]))
		self.test_mean, self.test_std = tmp.mean(0, keepdims=True)[None,:], tmp.std(0, keepdims=True)[None,:]

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
		self.train_set = SpikeDataset_3d(self.args, is_train=True)
		self.test_set = SpikeDataset_3d(self.args, is_train=False)

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
