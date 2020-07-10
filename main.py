# RUN code
from data_loader import *
from models      import *

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
	# load train data in batches
	batch_size   = 20
	n_epochs = 15
	lr = 1e-4
	train_loader = DataLoader(transformed_dataset, 
	                          batch_size=batch_size,
	                          shuffle=True, 
	                          num_workers=4)
	tot_batch = len(train_loader)
	print('There are {} batcheds'.format(tot_batch))

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

			# print loss statistics every 10 batches
			if batch_i%10 == 0:
				running_loss += loss.item()
				print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
				lossLogger[epoch*(tot_batch//10) + batch_i//10] = running_loss/10
				running_loss = 0.0

		state = {'state_dict': net.state_dict(),
				 'optimizer' : optimizer.state_dict(),
				 'n_epoch': epoch, 
				 'lossLogger': lossLogger}
		torch.save(state, this_model_path)

	print('Finished Training')
	plt.figure()
	plt.plot(lossLogger,label=this_model_path[:-3])
	plt.ylim([0,0.4])
	plt.legend(loc='upper right')
	plt.savefig(this_model_path[:-3]+'.png')




# main run function 
if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print('Running with '+str(device)+'...')

	train_net(device)