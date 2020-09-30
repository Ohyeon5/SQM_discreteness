# model specifications
import os,sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from convlstm_SreenivasVRao import *

###############
#   Networks  #
###############

# Discrete network: High level 
class Net_disc_high(nn.Module):

	'''
	High level discrete network with 'simple' or 'redundant' secondary convLSTM 
	Input images are processed continuously in the primary convlstm 
	and then outputs from every window frame are fed to secondary convlstm
	'''

	def __init__(self, n_classes, window, disc_type='simple', n_convBlocks=2, norm_type='bn', conv_n_feats=[3, 32, 64], 
				 clstm_hidden=[128, 256], return_all_layers=True, device='cpu',
				 fc_n_hidden=None):
		super(Net_disc_high, self).__init__()

		# initial parameter settings
		self.disc_type    = disc_type	# 'simple' or 'redundant' is available in the moment
		self.n_classes    = n_classes
		self.window       = window
		self.device       = device
		self.conv_n_feats = conv_n_feats
		self.clstm_hidden = clstm_hidden
		if fc_n_hidden is None:
			self.fc_n_hidden = n_classes*5
		else: 
			self.fc_n_hidden = fc_n_hidden

		# primary convolution blocks for preprocessing and feature extraction
		self.primary_Conv3D = Primary_conv3D(n_convBlocks=n_convBlocks, norm_type=norm_type,conv_n_feats=self.conv_n_feats)

		# Two layers of convLSTM
		self.primary_convlstm   = ConvLSTM_block(in_channels=self.conv_n_feats[n_convBlocks],hidden_channels=self.clstm_hidden[0], 
												 return_all_layers=True, device=self.device)
		self.secondary_convlstm = ConvLSTM_block(in_channels=self.clstm_hidden[0],hidden_channels=self.clstm_hidden[1], 
												 return_all_layers=return_all_layers, device=self.device)
		self.ff_classifier      = FF_classifier(in_channels=self.clstm_hidden[-1], n_classes=self.n_classes, 
												hidden_channels=self.fc_n_hidden, norm_type=norm_type)

	def forward(self,x):
		if self.disc_type is 'simple':
			return forward_simple(self,x)
		elif self.disc_type is 'redundant':
			return forward_redundant(self,x)

	def forward_redundant(self,x):
		# arg: x is a list of images
		x    = self.primaryConv3D(x)  # Primary feature extraction: list x -> B x C x T x H x W transposed to -> B x T x C x H x W
		x, _ = self.primary_convlstm(x)

		# discrete step: high level - redundant - repeat the output of nth frame to have same T
		imgs = []
		for t in range(0, x.shape[1], self.window):
			mm = x[:,t*self.window,:,:,:].unsqueeze(1).repeat(1,2,1,1,1)
			imgs.append(mm)
		img = torch.cat(imgs,1)
		print(self.disc_type, img.shape)

		img, _ = self.secondary_convlstm(img)  	# img: 5D tensor => B x T x Filters x H x W

		# Base Network: use the last layer only
		img = img[-1][:,-1,:,:,:].squeeze()
		img = self.ff_classifier(img)

		return img

	def forward_simple(self,x):
		# arg: x is a list of images
		x = self.primaryConv3D(x)  # Primary feature extraction: list x -> B x C x T x H x W transposed to -> B x T x C x H x W
		x, _ = self.primary_convlstm(x)

		# discrete step: high level - simple - every window frame
		img = x[:,slice(self.window-1,None,self.window),:,:,:]
		print(self.disc_type, img.shape)

		img, _ = self.secondary_convlstm(img)  	# img: 5D tensor => B x T x Filters x H x W

		# Base Network: use the last layer only
		img = img[-1][:,-1,:,:,:].squeeze()
		img = self.ff_classifier(img)

		return img

# Discrete network: Low level - simple
class Net_disc_low(nn.Module):

	'''
	Low level discrete network with 'simple' or 'redundant' secondary convLSTM 
	Input images are divided every 'window' frames and are processed in individual primary_convlstm 
	and only the last output from each window are stacked and fed to secondary convlstm
	'''

	def __init__(self, n_classes, window, disc_type='simple', n_convBlocks=2, norm_type='bn', conv_n_feats=[3, 32, 64], 
				 clstm_hidden=[128, 256], return_all_layers=True, device='cpu',
				 fc_n_hidden=None):
		super(Net_disc_low, self).__init__()

		# initial parameter settings
		self.disc_type    = disc_type	# 'simple' or 'redundant' is available in the moment
		self.n_classes    = n_classes
		self.window       = window
		self.device       = device
		self.conv_n_feats = conv_n_feats
		self.clstm_hidden = clstm_hidden
		if fc_n_hidden is None:
			self.fc_n_hidden = n_classes*5
		else: 
			self.fc_n_hidden = fc_n_hidden

		# primary convolution blocks for preprocessing and feature extraction
		self.primary_Conv3D = Primary_conv3D(n_convBlocks=n_convBlocks, norm_type=norm_type,conv_n_feats=self.conv_n_feats)

		# Two layers of convLSTM
		self.primary_convlstm   = ConvLSTM_block(in_channels=self.conv_n_feats[n_convBlocks],hidden_channels=self.clstm_hidden[0], 
												 return_all_layers=True, device=self.device)
		self.secondary_convlstm = ConvLSTM_block(in_channels=self.clstm_hidden[0],hidden_channels=self.clstm_hidden[1], 
												 return_all_layers=return_all_layers, device=self.device)
		self.ff_classifier      = FF_classifier(in_channels=self.clstm_hidden[-1], n_classes=self.n_classes, 
												hidden_channels=self.fc_n_hidden, norm_type=norm_type)

	def forward(self,x):
		if self.disc_type is 'simple':
			return forward_simple(self,x)
		elif self.disc_type is 'redundant':
			return forward_redundant(self,x)

	def forward_redundant(self,x):
		# arg: x is a list of images
		x = self.primaryConv3D(x)  # Primary feature extraction: list x -> B x C x T x H x W transposed to -> B x T x C x H x W
		
		# discrete step: input is fed every window frames individually, and only the last output of the primary convlstm is saved
		imgs = []
		for t in range(0, x.shape[1], self.window):
			ind_end = (t+1)*self.window if (t+1)*self.window<x.shape[1] else None
			mm, _ = self.primary_convlstm(x[:,t*self.window:ind_end,:,:,:]) # mm: 5D tensor => B x T x Filters x H x W
			imgs.append(mm[-1][:,-1,:,:,:].unsqueeze(1).repeat(1,self.window,1,1,1))
		img = torch.cat(imgs,1) # stacked img: 5D tensor => B x T x C x H x W
		print(self.disc_type, img.shape)

		img, _ = self.secondary_convlstm(img)  	# img: 5D tensor => B x T x Filters x H x W

		# Base Network: use the last layer only
		img = img[-1][:,-1,:,:,:].squeeze()
		img = self.ff_classifier(img)

		return img

	def forward_simple(self,x):
		# arg: x is a list of images
		x = self.primaryConv3D(x)  # Primary feature extraction: list x -> B x C x T x H x W transposed to -> B x T x C x H x W
		
		# discrete step: input is fed every window frames individually, and only the last output of the primary convlstm is saved
		imgs = []
		for t in range(0, x.shape[1], self.window):
			ind_end = (t+1)*self.window if (t+1)*self.window<x.shape[1] else None
			mm, _ = self.primary_convlstm(x[:,t*self.window:ind_end,:,:,:]) # mm: 5D tensor => B x T x Filters x H x W
			imgs.append(mm[-1][:,-1,:,:,:])
		img = torch.stack(imgs,1) # stacked img: 5D tensor => B x T x C x H x W
		print(self.disc_type, img.shape)
		
		img, _ = self.secondary_convlstm(img)  	# img: 5D tensor => B x T x Filters x H x W

		# Base Network: use the last layer only
		img = img[-1][:,-1,:,:,:].squeeze()
		img = self.ff_classifier(img)

		return img


# Baseline network (continuous)
class Net_continuous(nn.Module):

	def __init__(self, n_classes, n_convBlocks=2, norm_type='bn', conv_n_feats=[3, 32, 64], 
				 clstm_hidden=[128, 256], return_all_layers=True, device='cpu',
				 fc_n_hidden=None):
		super(Net_continuous, self).__init__()

		# initial parameter settings
		self.device = device
		self.conv_n_feats = conv_n_feats
		self.clstm_hidden = clstm_hidden
		self.n_classes    = n_classes
		if fc_n_hidden is None:
			self.fc_n_hidden = n_classes*5
		else: 
			self.fc_n_hidden = fc_n_hidden

		# primary convolution blocks for preprocessing and feature extraction
		self.primary_Conv3D = Primary_conv3D(n_convBlocks=n_convBlocks, norm_type=norm_type,conv_n_feats=self.conv_n_feats)

		# Two layers of convLSTM
		self.primary_convlstm   = ConvLSTM_block(in_channels=self.conv_n_feats[n_convBlocks],hidden_channels=self.clstm_hidden[0], 
												 return_all_layers=True, device=self.device)
		self.secondary_convlstm = ConvLSTM_block(in_channels=self.clstm_hidden[0],hidden_channels=self.clstm_hidden[1], 
												 return_all_layers=return_all_layers, device=self.device)
		self.ff_classifier      = FF_classifier(in_channels=self.clstm_hidden[-1], n_classes=self.n_classes, 
												hidden_channels=self.fc_n_hidden, norm_type=norm_type)

	def forward(self,x):
		# arg: x is a list of images

		img = self.primaryConv3D(x)  # Primary feature extraction: list x -> B x C x T x H x W transposed to -> B x T x C x H x W
		img, _ = self.primary_convlstm(img)  	# img: 5D tensor => B x T x Filters x H x W
		img, _ = self.secondary_convlstm(img)  	# img: 5D tensor => B x T x Filters x H x W

		# Base Network: use the last layer only
		img = img[-1][:,-1,:,:,:].squeeze()
		img = self.ff_classifier(img)

		return img


###########################
# Network Building Blocks #
###########################

# 1) Primary feature extraction conv layer
class Primary_conv3D(nn.Module):
	'''
	Primary feedforward feature extraction convolution layers 
	'''
	def __init__(self, n_convBlocks=2, norm_type='bn',conv_n_feats=[3, 32, 64]):
		super(Primary_conv3D, self).__init__()

		# initial parameter settings
		self.conv_n_feats = conv_n_feats

		# primary convolution blocks for preprocessing and feature extraction
		layers = []
		for ii in range(n_convBlocks): 
			block = Conv3D_Block(self.conv_n_feats[ii],self.conv_n_feats[ii+1],norm_type=norm_type)
			layers.append(block)

		self.primary_Conv3D = nn.Sequential(*layers)

	def forward(self, x):	
		# arg: x is a list of images
		# Stack to 5D layer and then pass 5d (BxCxTxHxW) to primaryConv3D and transpose it to BxTxCxHxW

		img = torch.stack(x,2) # stacked img: 5D tensor => B x C x T x H x W
		img = self.primaryConv3D(img)
		img = torch.transpose(img,2,1)  # Transpose B x C x T x H x W --> B x T x C x H x W

		return img		

# 2) Primary and Secondary convLSTMs
class ConvLSTM_block(nn.Module):
	'''
	ConvLSTM blocks 
	'''
	def __init__(self, in_channels, hidden_channels, kernel_size=(3,3), num_layers=1, return_all_layers=True, device='cpu'):
		super(ConvLSTM_block, self).__init__()


		self.convlstm_block   = ConvLSTM(in_channels=in_channels, hidden_channels=hidden_channels, 
										   kernel_size=kernel_size, num_layers=num_layers, bias=True, 
										   batch_first=True, return_all_layers=return_all_layers, device=self.device)

	def forward(self, x):	
		# arg: x is a 5D tensor => B x T x Filters x H x W
		x, _ = self.convlstm_block(x) 
		return x

# 3) Feedforward classifier
class FF_classifier(nn.Module):
	'''
	Feedforward fully connected classifier
	'''
	def __init__(self, in_channels, n_classes, hidden_channels, norm_type=None):
		super(FF_classifier, self).__init__()

		self.avgpool    = nn.AdaptiveAvgPool2d((2, 2))
		self.norm_layer = define_norm(in_channels,norm_type,dim_mode=2)
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(2*2*in_channels, hidden_channels),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(hidden_channels, n_classes))

	def forward(self, x):	
		# arg: x is a 4D tensor B x C x H x W

		x = self.avgpool(x)
		if self.norm_layer is not None:
			x = self.norm_layer(x)
		x = x.contiguous().view(x.shape[0],-1)
		x = self.classifier(x)

		return x		


# Conv3D block 
class Conv3D_Block(nn.Module):
	''' 
	use conv3D than multiple Conv2D blocks (for a sake of reducing computational burden)
	INPUT dimension: BxCxTxHxW
	'''
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding=1,norm_type=None):
		# kernel_size, stride, padding should be int scalar value, not tuple nor list
		super(Conv3D_Block,self).__init__()
		# parameters
		self.norm_type = norm_type

		# layers
		self.conv      = nn.Conv3d(in_channels,out_channels,kernel_size=(1,kernel_size,kernel_size),
								   stride=(1,stride,stride),padding=(1,padding,padding))
		self.relu      = nn.ReLU(inplace=True)
		self.maxpool   = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))		
		self.norm_layer= define_norm(out_channels,norm_type,dim_mode=3)

	def forward(self,x):

		x = self.conv(x)
		x = self.relu(x)
		x = self.maxpool(x)
		if self.norm_layer is not None:
			x = self.norm_layer(x)

		return x



##################
#  Aid functions #
################## 

# Define normalization type
def define_norm(n_channel,norm_type,n_group=None,dim_mode=2):
	# define and use different types of normalization steps 
	# Referred to https://pytorch.org/docs/stable/_modules/torch/nn/modules/normalization.html
	if norm_type is 'bn':
		if dim_mode == 2:
			return nn.BatchNorm2d(n_channel)
		elif dim_mode==3:
			return nn.BatchNorm3d(n_channel)
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

if __name__ == '__main__':
	# usage example 
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	net = Net_continuous(in_channels=3, n_classes=5)
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



		