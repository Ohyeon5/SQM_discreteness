# model specifications
import os,sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from convolution_lstm import *

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

	def __init__(self, in_channels, n_classes, n_blocks=2):
		super(BaseNet, self).__init__()

		# initial parameter settings
		
		# primary convolution blocks for preprocessing and feature extraction
		layers = []
		for ii in range(n_blocks): 
			block = ConvBlock(self.n_feats[ii],self.n_feats[ii+1],norm_type=norm_type)
			layers.append(block)

		self.primaryConv = nn.Sequential(*layers)
		self.convlstm = 

	def forward(self, x):
		