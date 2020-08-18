# create input sequence (use torch Dataset and DataLoader)
import os, sys
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from math import sqrt, ceil, floor

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# make subset of data with designated labels only
def make_data_subset_labels(csv_file, img_path='./data/20bn-jester-v1/', label_file='./data/jester-v1-labels.csv',
                            labels=None):
    '''
	output pandas dataframe: 'ID': img ids, 'label': labeled motion, 'label_id': each label's ID (0~n_classes]
	'''

    df = pd.read_csv(csv_file, header=None, sep=';',
                     names=['ID', 'label'])  # customized for the csv_file, data specific customization is needed

    img_list = [int(img) for img in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, img))]

    # keep only downloaded images
    _, ind, _ = np.intersect1d(df['ID'], img_list, return_indices=True)
    df = df.loc[ind, :]
    # print(df.values.shape)

    # if test set
    if df['label'].isnull().all():
        return df

    # if labels is not specified, use all the labels in the label file
    if labels is None:
        labels = pd.read_csv(label_file, header=None,
                             sep=';').values.squeeze().tolist()  # customized for the csv_file, data specific customization is needed

    # select designated labels only and save in a new_df
    new_df = pd.DataFrame(columns=['ID', 'label', 'label_id'])
    for ii, lab in enumerate(labels):
        sub_df = df.loc[df['label'] == lab, :]
        sub_df['label_id'] = ii
        new_df = pd.concat([new_df, sub_df], ignore_index=True)

    # print(ii, new_df)

    return new_df


# torch Dataset
class Jester20BnDataset(Dataset):
    ''' load the 20BN jester dataset : https://20bn.com/datasets/jester/v1#download'''

    def __init__(self, csv_file, img_path='./data/20bn-jester-v1/', label_file='./data/jester-v1-labels.csv',
                 labels=None, transform=None):
        """
		Args:
		    csv_file (string): Path to the csv file with annotations.
		    img_path (string): Directory with all the images.
		    label_file (string): path to the csv file with labels
		    labels (list): type of labels to be used 
		    transform (callable, optional): Optional transform to be applied
		        on a sample.
		"""
        self.img_path = img_path
        self.transform = transform
        self.labels = labels

        self.img_id_label_frame = make_data_subset_labels(csv_file, img_path=self.img_path, label_file=label_file,
                                                          labels=self.labels)

    def __len__(self):
        return self.img_id_label_frame.shape[0]

    def __getitem__(self, idx):
        # load all images of idx
        idx_path = os.path.join(self.img_path, str(self.img_id_label_frame['ID'][idx]))
        image_list = [os.path.join(self.img_path, str(self.img_id_label_frame['ID'][idx]), img) for img in
                      os.listdir(idx_path) if '.jpg' in img or '.png' in img]

        images = [mpimg.imread(image_name) for image_name in image_list]

        # if image has an alpha color channel, get rid of it
        if (images[0].shape[2] == 4):
            new_images = [image[:, :, 0:3] for image in images]
            images = new_images

        label = self.img_id_label_frame['label'][idx]
        label_id = self.img_id_label_frame['label_id'][idx]
        sample = {'images': images, 'label': label, 'label_id': label_id}

        if self.transform:
            sample = self.transform(sample)

        return sample


#######################
#   Transformations   #
#######################

## TODOLIST: Add transformations ex) RegularCrop, Rescale, Normalize, ReduceTemporalFreq ....

class RandomCrop(object):
    """Crop randomly the image in a sample.

	Args:
	    output_size (tuple or int): Desired output size. If int, square crop
	        is made.
	"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        images, label, label_id = sample['images'], sample['label'], sample['label_id']

        for ii, image in enumerate(images):
            h, w = image.shape[:2]
            new_h, new_w = self.output_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image = image[top: top + new_h,
                    left: left + new_w]
            images[ii] = image

        return {'images': images, 'label': label, 'label_id': label_id}


class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""

    def __call__(self, sample):
        images, label, label_id = sample['images'], sample['label'], sample['label_id']

        for ii, image in enumerate(images):
            image_copy = np.copy(image)

            # scale color range from [0, 255] to [0, 1]
            image_copy = image_copy / 255.0
            images[ii] = image_copy

        return {'images': images, 'label': label, 'label_id': label_id}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        images, label, label_id = sample['images'], sample['label'], sample['label_id']

        for ii, image in enumerate(images):
            # if image has no grayscale color channel, add one
            if (len(image.shape) == 2):
                # add that third color dim
                image = image.reshape(image.shape[0], image.shape[1], 1)

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            image = image.transpose((2, 0, 1))
            images[ii] = torch.from_numpy(image).float()

        return {'images': images, 'label': label,
                'label_id': torch.from_numpy(np.array(label_id))}


if __name__ == '__main__':

    # example plots

    data_transform = transforms.Compose([RandomCrop(90),
                                         Normalize(),
                                         ToTensor()])

    # create the transformed dataset
    transformed_dataset = Jester20BnDataset(csv_file='./data/jester-v1-train.csv',
                                            img_path='./data/20bn-jester-v1/',
                                            label_file='./data/jester-v1-labels.csv',
                                            labels=['Swiping Left', 'Swiping Right'],
                                            transform=data_transform)
    # load train data in batches
    train_loader = DataLoader(transformed_dataset,
                              batch_size=20,
                              shuffle=True,
                              num_workers=4)

    for ii, sample in enumerate(train_loader):
        images = sample['images']
        label = sample['label']
        label_id = sample['label_id']
        r, c = ceil(sqrt(len(images))), ceil(len(images) / ceil(sqrt(len(images))))
        plt.figure()
        for ii, img in enumerate(images):
            print(img.to('cpu').numpy().shape)
            img_plot = img.to('cpu').numpy()[0].squeeze().transpose(1, 2, 0)
            plt.subplot(r, c, ii + 1)
            plt.imshow(img_plot)
            plt.axis('off')
        plt.suptitle(
            'Label: {}, label id: {}, total: {} images'.format(label[0], label_id.to('cpu').numpy()[0], len(images)))

        plt.show()
