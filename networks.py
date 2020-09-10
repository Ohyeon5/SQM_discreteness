from utils import *
import torch


def network(device, type, temporal, spatial):
    if type == 'cont':
        continuous_discrete(device, temporal, spatial)
    elif type == 'disc1':
        continuous_discrete(device, temporal, spatial, 1, discrete=True)
    elif type == 'disc2':
        continuous_discrete(device, temporal, spatial, 2, discrete=True)
    elif type == 'disc3':
        discrete2(device, temporal, spatial)


def continuous_discrete(device, temporal, spatial, disc=0, discrete=False):
    # constants
    batch_size = 20
    n_epochs = 15
    lr = 1e-4

    # load data
    labels_dict, val_loader, train_loader = load_in_data(batch_size, spatial)

    # get and print number of classes and batches
    n_classes = get_n_classes(train_loader, val_loader, labels_dict)
    print('There are {} classes'.format(n_classes))

    tot_batch = len(train_loader)
    print('There are {} batches'.format(tot_batch))

    # initialise network
    net = BaseNet(in_channels=3, n_classes=n_classes, dimMode=3, device=device, discrete=discrete, disc=disc)
    net = net.to(device)

    # initialise criterion, optimizer, and loggers
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    epoch_start = 0
    lossLogger = np.zeros(n_epochs * (tot_batch // 10) + 1)
    valLossLogger = []
    accLogger = np.zeros(n_epochs * (tot_batch // 10) + 1)
    valAccLogger = []

    # check if model is saved already
    lossLogger, epoch_start, model_path, net = check_model_saved(net, optimizer, n_epochs, tot_batch, epoch_start,
                                                                      lossLogger)

    for epoch in range(epoch_start, n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        running_accuracy = 0.0
        counter = 0
        # train the network
        net.train()

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):

            images = data[0][::temporal]
            label_id = torch.from_numpy(np.asarray([labels_dict[v] for v in data[1]])).to(device)

            # print(len(images))
            # print(len(label_id))

            # get output label_ids from the network
            output_ids = net(images)
            y_pred_softmax = torch.log_softmax(output_ids, dim=1)
            _, y_pred = torch.max(y_pred_softmax, dim=1)

            # print(len(output_ids))

            # get accuracy
            accuracy = np.mean(np.asarray(y_pred.tolist()) == np.asarray(label_id.tolist()))

            # calculate the loss between predicted and target label_ids
            loss = criterion(output_ids, label_id)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # update loss and accuracy
            running_loss += loss.item()
            running_accuracy += accuracy

            # print loss statistics every 10 batches
            if (batch_i + 1) % 10 == 0:
                counter += 1
                print('Epoch: {}, Batch: {}, Avg. Loss: {}, Avg. Acc: {}%'.format(epoch + 1, batch_i + 1,
                                                                                  running_loss / 10,
                                                                                  np.round(running_accuracy * 10, 2)))
                lossLogger[epoch * (tot_batch // 10) + batch_i // 10] = running_loss / 10
                accLogger[epoch * (tot_batch // 10) + batch_i // 10] = running_accuracy / 10
                valLossLogger.append(np.nan)
                valAccLogger.append(np.nan)

                running_loss = 0.0
                running_accuracy = 0.0

        # net.eval()
        print('Validating...')
        valLoss, valAcc = validate(device, val_loader, labels_dict, criterion, net)
        valLossLogger[-1] = valLoss
        valAccLogger[-1] = valAcc

        # state = {'state_dict': net.state_dict(),
        #          'optimizer': optimizer.state_dict(),
        #          'n_epoch': epoch,
        #          'lossLogger': lossLogger}
        # torch.save(state, this_model_path)

    print('Finished Training')
    plot_graphs(valLossLogger, valAccLogger, lossLogger, accLogger, disc, spatial, temporal, model_path)


def discrete1(device):
    # constants
    batch_size = 20
    n_epochs = 15
    lr = 1e-4

    # load data
    labels_dict, val_loaders, train_loaders = load_in_data(batch_size, disc1=True)

    # get and print number of classes and batches
    n_classes = get_n_classes(train_loaders[0], val_loaders[0], labels_dict)
    print('There are {} classes'.format(n_classes))

    # initialise list to save nets
    nets = []
    for index in range(len(train_loaders)):
        train_loader = train_loaders[index]
        val_loader = val_loaders[index]

        tot_batch = len(train_loader)
        print('There are {} batches'.format(tot_batch))

        # initialise network
        net = BaseNet(in_channels=3, n_classes=n_classes, dimMode=3, device=device)
        net = net.to(device)

        # initialise criterion, optimizer, and loggers
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        epoch_start = 0
        lossLogger = np.zeros(n_epochs * (tot_batch // 10) + 1)
        valLossLogger = []
        accLogger = np.zeros(n_epochs * (tot_batch // 10) + 1)
        valAccLogger = []

        # check if model is saved already
        lossLogger, epoch_start, this_model_path, net = check_model_saved(net, optimizer, n_epochs, tot_batch,
                                                                          epoch_start, lossLogger)

        # train the network
        net_collector = torch.Tensor([]).to(device)
        label_collector = torch.Tensor([]).to(device)

        for epoch in range(epoch_start, n_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            running_accuracy = 0.0
            counter = 0
            net.train()

            # train on batches of data, assumes you already have train_loader
            for batch_i, data in enumerate(train_loader):

                images = data[0]
                label_id = torch.from_numpy(np.asarray([labels_dict[v] for v in data[1]])).to(device)

                # get output label_ids from the network
                output_ids = net(images).to(device)
                y_pred_softmax = torch.log_softmax(output_ids, dim=1)
                _, y_pred = torch.max(y_pred_softmax, dim=1)

                if epoch == n_epochs - 1:
                    net_collector = torch.cat((net_collector, output_ids))
                    label_collector = torch.cat((label_collector, label_id))

                # get accuracy
                accuracy = np.mean(np.asarray(y_pred.tolist()) == np.asarray(label_id.tolist()))

                # calculate the loss between predicted and target label_ids
                loss = criterion(output_ids, label_id)

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward pass to calculate the weight gradients
                loss.backward()

                # update the weights
                optimizer.step()

                # update loss and accuracy
                running_loss += loss.item()
                running_accuracy += accuracy

                # print loss statistics every 10 batches
                if (batch_i + 1) % 10 == 0:
                    counter += 1
                    print('Epoch: {}, Batch: {}, Avg. Loss: {}, Avg. Acc: {}%'.format(epoch + 1, batch_i + 1,
                                                                                      running_loss / 10,
                                                                                      np.round(running_accuracy * 10,
                                                                                               2)))
                    lossLogger.append(running_loss / 10)
                    accLogger.append(running_accuracy / 10)
                    valLossLogger.append(np.nan)
                    valAccLogger.append(np.nan)

                    running_loss = 0.0
                    running_accuracy = 0.0

            print('Validating...')
            net.eval()
            valLoss, valAcc = validate(device, val_loader, labels_dict, criterion, net)
            valLossLogger[-1] = valLoss
            valAccLogger[-1] = valAcc

            state = {'state_dict': net.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'n_epoch': epoch,
                     'lossLogger': lossLogger}
            torch.save(state, this_model_path)

        print('Finished Training')
        plot_graphs(valLossLogger, valAccLogger, lossLogger, accLogger, this_model_path, index)
        nets.append(net_collector)

    averaged_nets = torch.mean(torch.stack(nets), dim=0)
    y_pred_softmax = torch.log_softmax(averaged_nets, dim=1)
    _, y_pred = torch.max(y_pred_softmax, dim=1)
    final_acc = np.mean(np.asarray(y_pred.tolist()) == np.asarray(label_collector.tolist()))

    print("Final accuracy = {}%".format(final_acc * 100))


def discrete2(device, temporal, spatial):
    # constants
    batch_size = 20
    n_epochs = 15
    lr = 1e-4

    # load data
    labels_dict, val_loaders, train_loaders = load_in_data(batch_size, spatial, disc1=True)

    # get and print number of classes and batches
    n_classes = get_n_classes(train_loaders[0], val_loaders[0], labels_dict)
    print('There are {} classes'.format(n_classes))

    # initialise list to save nets
    # nets = []
    nets_dict = {}
    epoch_start = 0
    loss_calc = nn.CrossEntropyLoss().to(device)

    for index in range(len(train_loaders)):
        train_loader = train_loaders[index]
        val_loader = val_loaders[index]

        tot_batch = len(train_loader)
        print('There are {} batches'.format(tot_batch))

        # initialise network

        net = BaseNet(in_channels=3, n_classes=n_classes, dimMode=3, device=device)
        net = net.to(device)

        # initialise criterion, optimizer, and loggers
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        epoch_start = 0
        lossLogger = np.zeros(n_epochs * (tot_batch // 10) + 1)

        # check if model is saved already
        lossLogger, epoch_start, model_path, net = check_model_saved(net, optimizer, n_epochs, tot_batch,
                                                                          epoch_start, lossLogger)

        nets_dict[index] = {'net': net,
                            'criterion': criterion,
                            'optimizer': optimizer,
                            'epoch_start': epoch_start,
                            'val_loader': val_loader,
                            'train_loader': train_loader}

    lossLogger = []
    valLossLogger = []
    accLogger = []
    valAccLogger = []

    for epoch in range(epoch_start, n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        running_accuracy = 0.0
        counter = 0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data_n in enumerate(
                zip(nets_dict[0]['train_loader'], nets_dict[1]['train_loader'], nets_dict[2]['train_loader'],
                    nets_dict[3]['train_loader'])):
            batch_output = []
            for net_n in nets_dict.keys():
                net = nets_dict[net_n]['net']
                criterion = nets_dict[net_n]['criterion']
                optimizer = nets_dict[net_n]['optimizer']
                data = data_n[net_n]
                net.train()

                images = data[0][::temporal]
                label_id = torch.from_numpy(np.asarray([labels_dict[v] for v in data[1][::temporal]])).to(device)

                # get output label_ids from the network
                output_ids = net(images).to(device)
                batch_output.append(output_ids)

                # calculate the loss between predicted and target label_ids
                loss = criterion(output_ids, label_id)

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward pass to calculate the weight gradients
                loss.backward()

                # update the weights
                optimizer.step()

                # evaluate
                # print('Validating...')
                # net.eval()
                # valLoss, valAcc = validate(device, val_loader, labels_dict, criterion, net)
                # valLossLogger[-1] = valLoss
                # valAccLogger[-1] = valAcc

            # average labels
            stack = torch.stack(batch_output)
            averaged_nets = torch.mean(stack, dim=0)
            y_pred_softmax = torch.log_softmax(averaged_nets, dim=1)
            _, y_pred = torch.max(y_pred_softmax, dim=1)

            # get loss
            overall_loss = loss_calc(averaged_nets, label_id)

            # get accuracy
            overall_accuracy = np.mean(np.asarray(y_pred.tolist()) == np.asarray(label_id.tolist()))

            # update loss and accuracy
            running_loss += overall_loss.item()
            running_accuracy += overall_accuracy

            # print loss statistics every 10 batches
            if (batch_i + 1) % 10 == 0:
                counter += 1
                print('Epoch: {}, Batch: {}, Avg. Loss: {}, Avg. Acc: {}%'.format(epoch + 1, batch_i + 1,
                                                                                  running_loss / 10,
                                                                                  np.round(running_accuracy * 10,
                                                                                           2)))
                lossLogger.append(running_loss / 10)
                accLogger.append(running_accuracy / 10)
                valLossLogger.append(np.nan)
                valAccLogger.append(np.nan)

                running_loss = 0.0
                running_accuracy = 0.0

            # state = {'state_dict': net.state_dict(),
            #          'optimizer': optimizer.state_dict(),
            #          'n_epoch': epoch,
            #          'lossLogger': lossLogger}
            # torch.save(state, this_model_path)

    print('Finished Training')
    disc = 3
    plot_graphs(valLossLogger, valAccLogger, lossLogger, accLogger, disc, spatial, temporal, model_path)
