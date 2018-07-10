import torch
import torch.nn as nn

class GroupNorm(nn.Module):
	def __init__(self, num_channels, num_groups=32, eps=1e-5):
		super(GroupNorm, self).__init__()
		self.weight = nn.Parameter(torch.ones(1,num_channels,1))
		self.bias = nn.Parameter(torch.zeros(1,num_channels,1))
		self.num_groups = num_groups
		self.eps = eps

	def forward(self, x):
		N,C,L = x.size()
		G = self.num_groups
		assert C % G == 0

		x = x.view(N,G,-1)
		mean = x.mean(-1, keepdim=True)
		var = x.var(-1, keepdim=True)

		x = (x-mean) / (var+self.eps).sqrt()
		x = x.view(N,C,L)
		return x * self.weight + self.bias