import numpy as np
import h5py
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm


def resize_images(image_list, im_size):
    return_list = []
    for im in image_list:
        img = Image.open(im)
        img = img.resize((im_size, im_size), Image.ANTIALIAS)
        np_img = np.array(img)
        return_list.append(np_img)
    return return_list


def create_image_label_list(file_direc, group, im_size, skip, all_labels):
    label = all_labels['label'].loc[int(group)]
    image_list = os.listdir(file_direc + '/' + group)
    if len(image_list) < 22:
        return [], []
    image_list = image_list[:22:skip]
    images = resize_images([file_direc + '/' + group + '/' + i for i in image_list], im_size)
    return images, label


def make_hdf5(file_direc, im_size, skip, all_labels, desired_labels):
    indices = list(all_labels[all_labels['label'].isin(desired_labels)].index)
    hf = h5py.File('/Volumes/MAXTOR/data/data2.h5', 'w')
    for group in tqdm(indices):
        group = str(group)
        images, label = create_image_label_list(file_direc, group, im_size, skip, all_labels)
        if not images:
            print(group)
            continue
        hfgroup = hf.create_group(group)
        hfgroup.create_dataset('images', data=images)
        hfgroup.create_dataset('label', data=label)

    hf.close()


if __name__ == "__main__":
    file_direc = "/Volumes/MAXTOR/data/20bn-jester-v1"
    train_labels = pd.read_csv("/Volumes/MAXTOR/data/jester-v1-train.csv", names=['label'], sep=';')
    val_labels = pd.read_csv("/Volumes/MAXTOR/data/jester-v1-validation.csv", names=['label'], sep=';')
    all_labels = val_labels.append(train_labels)
    all_labels = all_labels.sort_index()
    labels = ['Swiping Left', 'Swiping Right', 'Swiping Up', 'Swiping Down']
    make_hdf5(file_direc, im_size=50, skip=2, all_labels=all_labels, desired_labels=labels)