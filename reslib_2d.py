import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_padding

class ResNeXtBottleNeck(nn.Module):
	def __init__(self, channels, ker, cardinality=1, dropRate=0):
		super(ResNeXtBottleNeck, self).__init__()
		inter_channels = int(channels / 2)
		self.conv_reduce = nn.Conv2d(channels, inter_channels, kernel_size=1, groups=cardinality, bias=False)
		self.bn_reduce = nn.BatchNorm2d(inter_channels)

		self.conv_conv = nn.Conv2d(inter_channels, inter_channels, kernel_size=ker, padding=get_padding(ker), groups=cardinality, bias=False)
		self.bn_conv = nn.BatchNorm2d(inter_channels)

		self.conv_expand = nn.Conv2d(inter_channels, channels, kernel_size=1, bias=False, groups=cardinality)
		self.bn_expand = nn.BatchNorm2d(channels)
		
		self.relu = nn.ReLU(inplace=True)
		self.dropout = nn.Dropout2d(p=dropRate)

	def forward(self, x):
		out = self.relu(self.bn_reduce(self.conv_reduce(x)))
		out = self.relu(self.bn_conv(self.conv_conv(out)))
		out = self.dropout(out)
		out = self.bn_expand(self.conv_expand(out))
		out = self.relu(out + x)
		return out

class BasicResBlock(nn.Module):
	def __init__(self, planes, ker, n_layers, decode=False, dropRate=0):
		super(BasicResBlock, self).__init__()
		self.relu = nn.ReLU(inplace=True)

		layers = []
		for i in range(n_layers):
			if decode:
				layers.append(nn.ConvTranspose2d(planes, planes, ker, padding=get_padding(ker), bias=False))
			else:
				layers.append(nn.Conv2d(planes, planes, ker, padding=get_padding(ker), bias=False))
			layers.append(nn.BatchNorm2d(planes))
			if i < (n_layers-1):
				layers.append(nn.ReLU(inplace=True))
			layers.append(nn.Dropout2d(p=dropRate))

		self.main_conv = nn.Sequential(*layers)

	def forward(self, x):
		out = self.main_conv(x)
		out += x
		out = self.relu(out)
		return out