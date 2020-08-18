# RUN code
from data_loader import *
from models import *
from h5_data_loader import *
from networks import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def load_in_data(batch_size, disc1=False):
    labels_dict = pd.read_csv("../data/jester-v1-labels.csv", names=['label'], sep=';')
    labels_dict['label_id'] = labels_dict.index
    labels_dict.set_index('label', inplace=True)
    labels_dict = labels_dict.to_dict()['label_id']

    if disc1:
        files = Path("../data/discrete").glob("*.h5")
        datasets = [HDF5Dataset(file, load_data=False, data_cache_size=4, transform=None) for file in files]
        train_loaders = []
        val_loaders = []
        for dataset in datasets:
            n_dataset = len(dataset)
            train_set = torch.utils.data.Subset(dataset, np.arange(n_dataset - (floor(n_dataset * 0.2))))
            val_set = torch.utils.data.Subset(dataset, np.arange(n_dataset - (floor(n_dataset * 0.2)), n_dataset))
            train_loaders.append(DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True))
            val_loaders.append(DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True))

        return labels_dict, val_loaders, train_loaders

    else:
        dataset = HDF5Dataset('../data/data.h5', load_data=False, data_cache_size=4, transform=None)
        n_dataset = len(dataset)
        train_set, val_set = torch.utils.data.random_split(dataset, (
            n_dataset - (floor(n_dataset * 0.2)), floor(n_dataset * 0.2)))

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

        return labels_dict, val_loader, train_loader


def get_n_classes(train, val, labels):
    for batch_i, data in enumerate(train):
        train_labels = np.asarray([labels[v] for v in data[1]])
    for batch_i, data in enumerate(val):
        val_labels = np.asarray([labels[v] for v in data[1]])
    n_classes = len(set(np.concatenate([train_labels, val_labels])))
    return n_classes


def plot_graphs(valLossLogger, valAccLogger, lossLogger, accLogger, this_model_path, index=-1):
    lossMask = np.isfinite(valLossLogger)
    accMask = np.isfinite(valAccLogger)
    x = np.arange(len(lossLogger) - 1)
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].plot(lossLogger, label="Training")
    axs[0].plot(x[lossMask], np.asarray(valLossLogger)[lossMask], label="Validation")
    axs[0].legend(loc='best')
    axs[1].plot(accLogger, label="Training")
    axs[1].plot(x[accMask], np.asarray(valAccLogger)[accMask], label="Validation")
    axs[1].legend(loc='best')
    if index != -1:
        plt.savefig(this_model_path[:-3] + '.png', bbox_inches='tight')
    else:
        plt.savefig(this_model_path[:-3] + str(index) + '.png', bbox_inches='tight')


def check_model_saved(net, optimizer, n_epochs, tot_batch, epoch_start, lossLogger):
    model_path = './saved_models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    this_model_path = model_path + 'baseNet.pt'
    #
    if os.path.exists(this_model_path):
        print('=> Loading checkpoint' + this_model_path)
        checkpoint = torch.load(this_model_path)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['n_epoch']
        if 'lossLogger' in checkpoint.keys():
            lossLogger = checkpoint['lossLogger']
            if len(lossLogger) < n_epochs * (tot_batch // 10) + 1:
                temp = lossLogger
                lossLogger = np.zeros(n_epochs * (tot_batch // 10) + 1)
                lossLogger[:len(temp)] = temp

    return lossLogger, epoch_start, this_model_path, net


def validate(device, dataset, labels_dict, criterion, net):
    with torch.no_grad():
        val_loss = 0.0
        val_accuracy = 0.0

        # validate on batches of data, assumes you already have val_loader
        for batch_i, data in enumerate(dataset):
            images = data[0]
            label_id = torch.from_numpy(np.asarray([labels_dict[v] for v in data[1]])).to(device)

            # get output label_ids from the network
            output_ids = net(images)
            y_pred_softmax = torch.log_softmax(output_ids, dim=1)
            _, y_pred = torch.max(y_pred_softmax, dim=1)

            # get accuracy
            accuracy = np.mean(np.asarray(y_pred.tolist()) == np.asarray(label_id.tolist()))

            # calculate the loss between predicted and target label_ids
            loss = criterion(output_ids, label_id)

            val_loss += loss.item()
            val_accuracy += accuracy

        val_loss /= len(dataset)
        val_accuracy /= len(dataset)

        print('Avg. Validation Loss: {}, Avg. Validation Accuracy: {}%'.format(val_loss, np.round(val_accuracy*100, 2)))
        return val_loss, val_accuracy

