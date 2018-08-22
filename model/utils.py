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
			layers.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, groups=cardinality, padding=get_padding(kernel_size), bias=False))
		else:
			layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, groups=cardinality, padding=get_padding(kernel_size), bias=False))
		layers.append(nn.BatchNorm1d(out_channels))
		layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Dropout(p=dropRate))
		in_channels = out_channels
	
	return nn.Sequential(*layers)

class SpikeDataset(Dataset):
	"""Dataset wrapping of spikes."""
	def __init__(self, args):
		self.args = args
		datamat = sio.loadmat(self.args.data_path)
		total_spks = datamat['total_spks'].astype(np.float32)

		if self.args.train_mode:
			self.data = total_spks[:round(len(total_spks) * self.args.train_portion)]
		else:
			self.data = total_spks[round(len(total_spks) * self.args.train_portion):]

	def get_normalizer(self):
		assert not self.args.train_mode, "Non-training data cannot be normalizer."
		sz = self.data.shape
		tmp = np.reshape(self.data, (sz[0]*sz[1], sz[2]))
		train_mean, train_std = tmp.mean(0, keepdims=True)[None,:], tmp.std(0, keepdims=True)[None,:]
		return train_mean, train_std

	def apply_norm(self, d_mean, d_std):
		self.data = (self.data - d_mean) / d_std
		
	def __getitem__(self, index):
		return torch.from_numpy(self.data[index])

	def __len__(self):
		return len(self.data)


class Helper():
	def __init__(self, args):
		self.args = args

	def create_data_loaders(self):
		self.train_set = SpikeDataset(self.args, is_train=True)
		self.test_set = SpikeDataset(self.args, is_train=False)

		train_loader = DataLoader(
			dataset=self.train_set,
			batch_size=self.args.batch_size,
			shuffle=True,
			num_workers=4
		)

		test_loader = DataLoader(
			dataset=self.test_set,
			batch_size=self.args.test_batch_size,
			shuffle=False,
			num_workers=4
		)
		
		return train_loader, test_loader

	def param_for_recon(self):
		if self.args.val_norm:
			return self.train_set.test_mean, self.train_set.test_std, self.test_set.spikes
		else:
			return self.train_set.train_mean, self.train_set.train_std, self.test_set.spikes
