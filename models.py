# model specifications
import os,sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from convlstm_SreenivasVRao import *

# Define normalization type
def define_norm(n_channel,norm_type,n_group=None):
	# define and use different types of normalization steps 
	# Referred to https://pytorch.org/docs/stable/_modules/torch/nn/modules/normalization.html
	if norm_type is 'bn':
		return nn.BatchNorm2d(n_channel)
	elif norm_type is 'gn':
		if n_group is None: n_group=2 # default group num is 2
		return nn.GroupNorm(n_group,n_channel)
	elif norm_type is 'in':
		return nn.GroupNorm(n_channel,n_channel)
	elif norm_type is 'ln':
		return nn.GroupNorm(1,n_channel)
	elif norm_type is None:
		return
	else:
		return ValueError('Normalization type - '+norm_type+' is not defined yet')

# conv block
class ConvBlock(nn.Module):
	# Conv building block 
	def __init__(self, inplane, outplane, kernel_size=3, stride=1,padding=1,norm_type=None):
		super(ConvBlock,self).__init__()
		# parameters
		self.norm_type = norm_type

		# layers
		self.conv      = nn.Conv2d(inplane,outplane,kernel_size=kernel_size,stride=stride,padding=padding)
		self.relu      = nn.ReLU(inplace=True)
		self.maxpool   = nn.MaxPool2d(kernel_size=2, stride=2)		
		self.norm_layer= define_norm(outplane,norm_type)

	def forward(self,x):

		x = self.conv(x)
		x = self.relu(x)
		x = self.maxpool(x)
		if self.norm_layer is not None:
			x = self.norm_layer(x)

		return x

# Baseline network 
class BaseNet(nn.Module):

	def __init__(self, in_channels, n_classes, n_convBlocks=2, norm_type='bn', 
		         conv_n_feats=[3, 32, 64], clstm_hidden=[128, 256], fc_n_hidden=None,
		         return_all_layers=False):
		super(BaseNet, self).__init__()

		# initial parameter settings
		self.conv_n_feats = conv_n_feats
		self.clstm_hidden = clstm_hidden
		if fc_n_hidden is None:
			self.fc_n_hidden = n_classes*5
		else: 
			self.fc_n_hidden = fc_n_hidden

		# primary convolution blocks for preprocessing and feature extraction
		layers = []
		for ii in range(n_convBlocks): 
			block = ConvBlock(self.conv_n_feats[ii],self.conv_n_feats[ii+1],norm_type=norm_type)
			layers.append(block)

		self.primaryConv = nn.Sequential(*layers)
		# Two layers of convLSTM
		self.convlstm   = ConvLSTM(in_channels=self.conv_n_feats[n_convBlocks], 
			                       hidden_channels=self.clstm_hidden, kernel_size=(3,3),
								   num_layers=2, batch_first=True, bias=True, return_all_layers=return_all_layers)
		self.avgpool    = nn.AdaptiveAvgPool2d((2, 2))
		self.norm_layer = define_norm(self.clstm_hidden[-1],norm_type)
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(2*2*self.clstm_hidden[-1], self.fc_n_hidden),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(self.fc_n_hidden, n_classes))

	def forward(self,x):
		# arg: x is a list of images
		# pass images in the list x to primaryConv layer and then stack/concat them to make 5D tensor
		img = []

		for ii, w in enumerate(x):
			w = self.primaryConv(w)
			img.append(w)

		img = torch.stack(img, 1)    # stacked img: 5D tensor => B x T x C x H x W
		
		img, _ = self.convlstm(img)  # img: 5D tensor => B x T x Filters x H x W

		# Base Network: use the last layer only
		img = img[-1][:,-1,:,:,:].squeeze()
		img = self.avgpool(img)
		if self.norm_layer is not None:
			img = self.norm_layer(img)
		img = img.contiguous().view(img.shape[0],-1)
		img = self.classifier(img)

		return img

if __name__ == '__main__':
	# usage example 
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	net = BaseNet(in_channels=3, n_classes=5)
	print(net)
	net = net.to(device)

	loss_fn = torch.nn.CrossEntropyLoss()

	x1 = torch.randn([5, 3, 100, 100]).to(device)
	x2 = torch.randn([5, 3, 100, 100]).to(device)
	x3 = torch.randn([5, 3, 100, 100]).to(device)
	tar = torch.rand(5,5).to(device)

	x_in = [x1,x2,x3]
	out  = net(x_in)

	print(out.shape)
	out.sum().backward()

	# # gradient check
	# res = torch.autograd.gradcheck(loss_fn, (out, tar), eps=1e-6, atol=1e-2, raise_exception=True)
	# print(res)



		