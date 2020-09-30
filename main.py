# RUN code
from hdf5_loader import *
from models      import *
from utils 	     import *

from math import floor, ceil
import matplotlib.pyplot as plt
import time

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def train_net(device, param):

	# 
	# data transformations
	data_transform = transforms.Compose([Normalize(), ToTensor()])
	hdf5_params    = {'load_data': False, 'data_cache_size': 4, 'transform': data_transform}

	labels       = param['labels']
	fn_postfix   = str(len(labels))
	train_fn     = param['data_path'] + os.sep + 'train_hdf5' + fn_postfix + '.h5'
	val_fn       = param['data_path'] + os.sep + 'val_hdf5'   + fn_postfix + '.h5'

	train_dataset = HDF5Dataset(file_path =train_fn, **hdf5_params)
	val_dataset   = HDF5Dataset(file_path =val_fn, **hdf5_params)

	# Load train and validation data in batches
	batch_size   = param['batch_size']
	n_epochs     = param['epochs']
	lr = 1e-2
	loader_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2}

	train_loader = DataLoader(train_dataset, **loader_params)
	val_loader   = DataLoader(val_dataset, **loader_params)

	# get and print number of classes and batches
	n_classes = len(labels)
	print('There are {} classes'.format(n_classes))
	tot_batch = len(train_loader)
	print('There are {} batches'.format(tot_batch))

	if 'Net_continuous' in param['model_name']:
		net = Net_continuous(n_classes=n_classes, device=device)
		print('Training Net_continuous ...')
	elif 'Net_disc_low' in param['model_name']:
		disc_type = 'simple' if 'simple' in param['model_name'] else 'redundant'
		net = Net_disc_low(n_classes=n_classes, device=device, window=3,disc_type=disc_type)
		print('Training Net_disc_low ' +disc_type+'...')
	elif 'Net_disc_high' in param['model_name']:
		disc_type = 'simple' if 'simple' in param['model_name'] else 'redundant'
		net = Net_disc_high(n_classes=n_classes, device=device, window=3,disc_type=disc_type)
		print('Training Net_disc_high ' +disc_type+'...')
	net = net.to(device)

	criterion = nn.CrossEntropyLoss().to(device)

	optimizer   = torch.optim.Adam(net.parameters(), lr=lr)
	epoch_start = 0
	# save ckpt and loggers every epoch
	logger      = {'train_loss': np.zeros(n_epochs), 'train_pc': np.zeros(n_epochs),
	               'val_loss': np.zeros(n_epochs),   'val_pc': np.zeros(n_epochs)} 

	model_path = param['model_path']
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	this_model_path = model_path + param['model_name'] + '.pt'

	if os.path.exists(this_model_path):
		print('=> Loading checkpoint' + this_model_path)
		checkpoint = torch.load(this_model_path)
		net.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		epoch_start = checkpoint['n_epoch']
		if 'loggers' in checkpoint.keys():
			logger = {log_key: log_val if len(logger[log_key]) >= n_epochs else np.append(log_val, np.zeros(n_epochs-len(log_val))) for log_key,log_val in checkpoint['loggers'].items()}

	# train the network
	net.train()

	for epoch in range(epoch_start,n_epochs):  # loop over the dataset multiple times
		time_start = time.time()
		net.train()
		
		running_loss = 0.0
		pc = 0.0

		# train on batches of data, assumes you already have train_loader
		for batch_i, data in enumerate(train_loader):
			images = data['images']
			label_id = data['label_id'].to(device)

			# get output label_ids from the network
			output_ids = net(images)

			# calculate the loss between predicted and target label_ids
			loss = criterion(output_ids, label_id)

			# zero the parameter (weight) gradients
			optimizer.zero_grad()

			# backward pass to calculate the weight gradients
			loss.backward()

			# update the weights
			optimizer.step()

			# calculate the accuracy
			running_loss += loss.item()
			pc           += sum(output_ids.to('cpu').detach().numpy().argmax(axis=1)==label_id.to('cpu').detach().numpy())/len(label_id)
			
		# print loss statistics every epoch
		print('Epoch: {}, Avg. Loss: {}, Avg. pc: {}, took {}'.format(epoch, running_loss/(batch_i+1), pc/(batch_i+1), time.strftime("%H:%M:%S", time.gmtime(time.time()-time_start))))
		logger['train_loss'][epoch] = running_loss/(batch_i+1)
		logger['train_pc'][epoch]   = pc/(batch_i+1)
		running_loss = 0.0
		pc = 0.0

		# validate every 5 epochs
		if epoch%5 == 4:
			with torch.no_grad():
				val_loss = 0.0
				val_pc   = 0.0
				for b, val in enumerate(val_loader):
					net.eval()
					val_img   = val['images']
					val_labid = val['label_id'].to(device)
					val_out   = net(val_img)
					val_bloss = criterion(val_out, val_labid)
					val_loss += val_bloss.item()
					val_pc   += sum(val_out.to('cpu').detach().numpy().argmax(axis=1)==val_labid.to('cpu').detach().numpy())/len(val_labid)

				print('\nEpoch: {}, Validation Avg. Loss: {}, Avg. pc: {}'.
			      format(epoch, val_loss/val_loader.__len__(), val_pc/val_loader.__len__()))
				logger['val_loss'][epoch] = val_loss/val_loader.__len__()
				logger['val_pc'][epoch]   = val_pc/val_loader.__len__()

		state = {'state_dict': net.state_dict(),
				 'optimizer' : optimizer.state_dict(),
				 'n_epoch': epoch, 
				 'loggers': {log_key: log_val for log_key, log_val in logger.items()}}
		torch.save(state, this_model_path)


	print('Finished Training')

	plt.figure()

	for pi,(log_key,log_val) in enumerate(logger.items()):
		print(pi,log_key)
		plt.subplot(2,2,pi+1)
		plt.scatter(np.arange(0,len(log_val)),log_val,label=log_key, s=5)
		plt.legend(loc='upper right')
		plt.title(log_key)

	plt.savefig(this_model_path[:-3]+'.png')

# main run function 
if __name__ == '__main__':

	# parse configuration
	param = get_configs()

	# device 
	use_gpu=1
	device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
	print(torch.cuda.is_available())
	print('Running with '+str(device)+'... model '+param['model_name'])

	train_net(device, param)