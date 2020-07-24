# RUN code
from data_loader import *
from models      import *

from math import floor, ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

def train_net(device):

	# Label subset
	labels = ['Swiping Left','Swiping Right','Swiping up','Swiping Down','Sliding Two Fingers Left']

	# Load data and apply transforms
	data_transform = transforms.Compose([RandomCrop(90),
										Normalize(),
										ToTensor()])

	# create the transformed dataset
	transformed_dataset = Jester20BnDataset(csv_file='./data/jester-v1-train.csv', 
		                                    img_path='./data/20bn-jester-v1/', 
		                                    label_file='./data/jester-v1-labels.csv', 
		                                    labels=labels, 
		                                    transform=data_transform)
	# Split train/test/validatition set (train:70%, val:10%, test:20%)
	n_dataset = len(transformed_dataset)
	train_set, val_set, test_set = torch.utils.data.random_split(transformed_dataset,(n_dataset-(floor(n_dataset*0.2)+floor(n_dataset*0.1)),floor(n_dataset*0.1),floor(n_dataset*0.2)))
	
	# load train data in batches
	batch_size   = 20
	n_epochs = 300
	lr = 1e-4
	train_loader = DataLoader(train_set, 
	                          batch_size=batch_size,
	                          shuffle=True, 
	                          num_workers=4)
	val_loader   = DataLoader(val_set, 
	                          batch_size=batch_size,
	                          shuffle=True, 
	                          num_workers=4)
	test_loader  = DataLoader(test_set, 
	                          batch_size=batch_size,
	                          shuffle=True, 
	                          num_workers=4)
	tot_batch = len(train_loader)
	print('There are {} batches'.format(tot_batch))

	net = BaseNet(in_channels=3, n_classes=5, dimMode=3, device=device)
	net = net.to(device)

	criterion = nn.CrossEntropyLoss().to(device)

	optimizer   = torch.optim.Adam(net.parameters(), lr=lr)
	epoch_start = 0
	lossLogger = np.zeros(n_epochs*(tot_batch//10)+1)

	model_path = './saved_models/'
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	this_model_path = model_path+'baseNet.pt'

	if os.path.exists(this_model_path):
		print('=> Loading checkpoint' + this_model_path)
		checkpoint = torch.load(this_model_path)
		net.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		epoch_start = checkpoint['n_epoch']
		if 'lossLogger' in checkpoint.keys():
			lossLogger = checkpoint['lossLogger']
			if len(lossLogger) < n_epochs*(tot_batch//10)+1 :
				temp = lossLogger
				lossLogger = np.zeros(n_epochs*(tot_batch//10)+1)
				lossLogger[:len(temp)] = temp

	# train the network
	net.train()

	for epoch in range(epoch_start,n_epochs):  # loop over the dataset multiple times

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
			
			# print loss statistics every 10 batches
			if batch_i%10 == 0:
				print('Epoch: {}, Batch: {}, Avg. Loss: {}, Avg. pc: {}'.format(epoch + 1, batch_i+1, running_loss/10, pc/10))
				lossLogger[epoch*(tot_batch//10) + batch_i//10] = running_loss/10
				running_loss = 0.0
				pc = 0.0

		state = {'state_dict': net.state_dict(),
				 'optimizer' : optimizer.state_dict(),
				 'n_epoch': epoch, 
				 'lossLogger': lossLogger}
		torch.save(state, this_model_path)

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
			      format(epoch + 1, val_loss/val_loader.__len__(), val_pc/val_loader.__len__()))


	print('Finished Training')
	plt.figure()
	plt.plot(lossLogger,label=this_model_path[:-3])
	plt.ylim([0,0.4])
	plt.legend(loc='upper right')
	plt.savefig(this_model_path[:-3]+'.png')



# main run function 
if __name__ == '__main__':
	use_gpu=1
	device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
	print('Running with '+str(device)+'...')

	train_net(device)