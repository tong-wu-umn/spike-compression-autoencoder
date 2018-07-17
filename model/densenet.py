import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
	def __init__(self, in_planes, out_planes, dropRate=0.0):
		super(BasicBlock, self).__init__()
		self.bn1 = nn.BatchNorm1d(in_planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.droprate = dropRate

	def forward(self, x):
		out = self.conv1(self.relu(self.bn1(x)))
		if self.droprate > 0:
			out = F.dropout(out, p=self.droprate, training=self.training)
		return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
	def __init__(self, in_planes, out_planes, dropRate=0.0):
		super(BottleneckBlock, self).__init__()
		inter_planes = out_planes * 4
		self.bn1 = nn.BatchNorm1d(in_planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv1d(in_planes, inter_planes, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn2 = nn.BatchNorm1d(inter_planes)
		self.conv2 = nn.Conv1d(inter_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.droprate = dropRate

	def forward(self, x):
		out = self.conv1(self.relu(self.bn1(x)))
		if self.droprate > 0:
			out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
		out = self.conv2(self.relu(self.bn2(out)))
		if self.droprate > 0:
			out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
		return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
	def __init__(self, in_planes, out_planes, downsample=True, dropRate=0.0):
		super(TransitionBlock, self).__init__()
		self.bn1 = nn.BatchNorm1d(in_planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
		self.droprate = dropRate
		self.downsample = downsample

	def forward(self, x):
		out = self.conv1(self.relu(self.bn1(x)))
		if self.droprate > 0:
			out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
		if self.downsample:
			return F.avg_pool1d(out, 2)
		else:
			return F.upsample(out, scale_factor=2, mode='linear')

class DenseBlock(nn.Module):
	def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
		super(DenseBlock, self).__init__()
		self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)

	def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
		layers = []
		for i in range(nb_layers):
			layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
		return nn.Sequential(*layers)

	def forward(self, x):
		return self.layer(x)

# class DenseNet_VAE(nn.Module):
# 	def __init__(self, growth_rate=12, reduction=0.5, bottleneck=True, dropRate=0.0):
# 		super(DenseNet_VAE, self).__init__()
# 		in_planes = 2 * growth_rate
# 		n = 10
# 		if bottleneck == True:
# 			block = BottleneckBlock
# 		else:
# 			block = BasicBlock

# 		# 1st encoder conv before any dense block
# 		self.conv1 = nn.Conv1d(1, in_planes, kernel_size=3, stride=1, padding=1, bias=False)

# 		# 1st encoder block
# 		self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
# 		in_planes = int(in_planes+n*growth_rate)
# 		self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), downsample=True, dropRate=dropRate)
# 		in_planes = int(math.floor(in_planes*reduction))
# 		encoder_blk1_planes = in_planes

# 		# 2nd encoder block
# 		self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
# 		in_planes = int(in_planes+n*growth_rate)
# 		self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), downsample=True, dropRate=dropRate)
# 		in_planes = int(math.floor(in_planes*reduction))
# 		encoder_blk2_planes = in_planes

# 		# 3rd encoder block
# 		self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
# 		in_planes = int(in_planes+n*growth_rate)
# 		encoder_blk3_planes = in_planes

# 		for m in self.modules():
# 			if isinstance(m, nn.Conv1d):
# 				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
# 				m.weight.data.normal_(0, math.sqrt(2. / n))
# 			elif isinstance(m, nn.BatchNorm1d):
# 				m.weight.data.fill_(1)
# 				m.bias.data.zero_()
# 			elif isinstance(m, nn.Linear):
# 				m.bias.data.zero_()

# 	def forward(self, x):
# 		out = self.conv1(x)
# 		out = self.trans1(self.block1(out))
# 		out = self.trans2(self.block2(out))
# 		out = self.block3(out)
# 		out = self.relu(self.bn1(out))
# 		out = F.avg_pool1d(out, 8)
# 		out = out.view(-1, self.in_planes)
# 		return self.fc(out)
