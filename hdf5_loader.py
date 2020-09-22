import h5py
from pathlib import Path

from scipy.ndimage import gaussian_filter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class HDF5Dataset(Dataset):
    """Represents an abstract HDF5 dataset.
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(self, file_path, spatial=None, load_data=False, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform
        self.spatial = spatial

        # Search for all h5 file 
        p = Path(file_path)
        assert (p.is_file())

        self._add_data_infos(str(p.resolve()), load_data)

    def __getitem__(self, index):
        # get data
        x = self.get_data("images", index)
        images = []

        for ii, image in enumerate(x):

            # Spatial blurring
            if self.spatial is not None:
                image = gaussian_filter(image, sigma=self.spatial)

            # normalise
            image = image / 255.0

            # if image has no grayscale color channel, add one
            if len(image.shape) == 2:
                # add that third color dim
                image = image.reshape(image.shape[0], image.shape[1], 1)

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            image = image.transpose((2, 0, 1))
            images.append(torch.from_numpy(image).float())

        ########## changes needed!!!!! #################

        if self.transform:
            sample = self.transform(sample)

        # get label
        y = self.get_data("label", index)
        return images, y

    def __len__(self):
        return len(self.get_data_infos('images'))

    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path, mode='r') as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(ds.value, file_path)

                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'label' to identify its type
                    # we also store the shape of the data in case we need it
                    self.data_info.append(
                        {'file_path': file_path, 'type': dname, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path, mode='r') as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    idx = self._add_to_cache(ds[()], file_path)

                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i, v in enumerate(self.data_info) if v['file_path'] == file_path)

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [
                {'file_path': di['file_path'], 'type': di['type'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di
                for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]


# plot hdf5_loader examples

if __name__ == '__main__':

    # example plots

    # parse configuration
    from utils import *
    param = get_configs()

    train_dataset = HDF5Dataset(file_path =param['data_path']+'train_hdf5.h5', load_data=False, data_cache_size=4, transform=None)
    val_dataset   = HDF5Dataset(file_path =param['data_path']+'val_hdf5.h5', load_data=False, data_cache_size=4, transform=None)

    print(len(train_dataset))

    # load train data in batches
    batch_size   = 20
    n_epochs = 300
    lr = 1e-4
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size,
                              shuffle=True, 
                              num_workers=4)
    val_loader   = DataLoader(val_dataset, 
                              batch_size=batch_size,
                              shuffle=True, 
                              num_workers=4)

    for ii,sample in enumerate(train_loader):
        images   = sample['images']
        label    = sample['label']
        label_id = sample['label_id']
        r,c = ceil(sqrt(len(images))), ceil(len(images)/ceil(sqrt(len(images))))
        plt.figure()
        for ii,img in enumerate(images):
            print(img.to('cpu').numpy().shape)
            img_plot = img.to('cpu').numpy()[0].squeeze().transpose(1,2,0)
            plt.subplot(r,c,ii+1)
            plt.imshow(img_plot)
            plt.axis('off')
        plt.suptitle('Label: {}, label id: {}, total: {} images'.format(label[0], label_id.to('cpu').numpy()[0],len(images)))

        plt.show()
