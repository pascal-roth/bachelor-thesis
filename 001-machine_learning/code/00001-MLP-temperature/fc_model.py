#######################################################################################################################
# functions to automatically construct and find the best model
######################################################################################################################

# import packages
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


# model building class ################################################################################################
class Net(nn.Module):
    def __init__(self, n_input, n_output, hidden_layers):
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(n_input, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], n_output)

        self.dropout = nn.Dropout(p=0)

    def forward(self, x):
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return x


# update lines of loss plot ############################################################################################
def updateLines(ax, train_losses, validation_losses):
    """
    updates line values for loss plot

    :parameter
    :param ax:                              axis object to be updated
    :param train_losses:        - list -    list of floats of training losses
    :param validation_losses:   - list -    list of floats of validation losses
    """

    x = list(range(1, len(train_losses)+1))
    lines = ax.get_lines()
    lines[0].set_xdata(x)
    lines[1].set_xdata(x)
    lines[0].set_ydata(train_losses)
    lines[1].set_ydata(validation_losses)
    plt.draw()
    plt.pause(1e-17)


# training function for the model ####################################################################################
def train(model, train_loader, valid_loader, criterion, optimizer, epochs, nbr_net, plot, inf_print, device):
    """Optimize the weights of a given MLP.

    :parameter
    :param model:                           SimpleMLP -> model to optimize
    :param train_loader:    - dataloader -  Data Loader of the Train samples and labels
    :param valid_loader:    - dataloader -  Data Loader of the Validation samples and labels
    :param criterion:                       pytorch Loss function
    :param optimizer:                       pytorch Optimizer function, include the learning rate
    :param epochs:          - int -         number of epochs to train
    :param nbr_net:         - str -         number to identify the trained network
    :param plot:            - bool -        if True, losses will be plotted while training the network
    :param inf_print:       - bool -        should some information be printed out
    :param device:          - str -         cpu, single gpu or multi gpu to use for training the model
    """

    # select device to train on
    if device == 'cpu':
        device = "cpu"
    elif device == 'gpu':
        device = "cuda:0"
    elif device == 'gpu_multi':
        device = "cuda:0"
        model = nn.DataParallel(model)

    model = model.to(device)
    print('Starting training on device: {}'.format(device))

    # the minimal validation loss at the beginning of the training
    valid_loss_min = 0
    for data, target in valid_loader:
        data, target = data.to(device), target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model.forward(data)
        # calculate the loss
        loss = criterion(output, target)
        # update running validation loss
        valid_loss_min += loss.item() * data.size(0)

    valid_loss_min = valid_loss_min / len(valid_loader)
    print("\nInitial validation loss is: {:6.5e}\n".format(valid_loss_min))

    # initialize loss array
    train_losses, validation_losses = [], []

    # Prepare plot
    if plot:
        # print("backend: "+plt.get_backend())
        xdata = []
        plt.show()
        ax = plt.gca()
        ax.set_xlim(0, epochs)
        ax.set_ylim(1e-4, valid_loss_min)
        plt.yscale('log')
        ax.plot(xdata, train_losses, 'r-', label="Training loss")
        ax.plot(xdata, validation_losses, 'b-', label="Validation loss")
        ax.legend()
        plt.draw()
        plt.pause(1e-17)

    # prepare tqdm progress bars
    # https://medium.com/@philipplies/progress-bar-and-status-logging-in-python-with-tqdm-35ce29b908f5
    outer = tqdm(total=epochs, position=0)
    inner = tqdm(total=int(len(train_loader.dataset) / train_loader.batch_size), position=1)
    loss_log = tqdm(total=0, position=3, bar_format='{desc}')
    best_log = tqdm(total=0, position=4, bar_format='{desc}')

    for epoch in range(epochs):
        running_loss = 0
        # Model in training mode, dropout is on
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model.forward(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            running_loss += loss.item() * data.size(0)
            inner.update(1)

        outer.update(1)
        inner.refresh()
        inner.reset()

        # Track training loss
        train_losses.append(running_loss/len(train_loader))

        # Turn off gradients for validation, will speed up inference
        with torch.no_grad():
            # Model in inference mode, dropout is off
            model.eval()
            valid_loss = 0

            for data, target in valid_loader:
                # move tensors to GPU if CUDA is available
                data, target = data.to(device), target.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model.forward(data)
                # calculate the loss
                loss = criterion(output, target)
                # update running validation loss
                valid_loss += loss.item() * data.size(0)

            valid_loss = valid_loss / len(valid_loader)
            validation_losses.append(valid_loss)

        if plot:
            updateLines(ax, train_losses, validation_losses)

        if valid_loss <= valid_loss_min:
            if device == 'gpu_multi':
                torch.save(model.module.state_dict(), 'model.pt')
            else:
                torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss

        outer.write("Epoch: {:05d}, Training loss: {:6.5e}, Validation loss: {:6.5e}".format
                    (epoch, train_losses[epoch], validation_losses[epoch]))
        loss_log.set_description_str("Epoch: {:05d}, Training loss: {:6.5e}, Validation loss: {:6.5e}".format
                                     (epoch, train_losses[epoch], validation_losses[epoch]))
        best_log.set_description_str("Best validation loss {}".format(valid_loss_min))

    inner.close()
    outer.close()

    # save losses
    path = Path(__file__).resolve()
    path_dir = path.parents[2] / 'data/00001-MLP-temperature'
    path_loss = path.parents[2] / 'data/00001-MLP-temperature/{}_losses.csv'.format(nbr_net)

    # check if directory exists
    try:
        os.makedirs(path_dir)
    except OSError:
        if inf_print is True:
            print('Directories already exist')
    else:
        if inf_print is True:
            print('Necessary directories created')

    train_losses = np.asarray(train_losses)
    validation_losses = np.asarray(validation_losses)

    train_losses = train_losses.reshape((len(train_losses), 1))
    validation_losses = validation_losses.reshape((len(validation_losses), 1))

    losses = np.concatenate((train_losses, validation_losses), axis=1)

    try:
        losses_pre = pd.read_csv(path_loss)
        losses_pre = losses_pre.values
        losses = np.append(losses_pre[:, 1:], losses, axis=0)
        losses = pd.DataFrame(losses)
    except FileNotFoundError:
        losses = pd.DataFrame(losses)

    losses.columns = ['train_loss', 'valid_loss']
    losses.to_csv(path_loss)


# save latest models in checkpoint files ##############################################################################
def save_model(model, n_input, n_output, optimizer, criterion, number_net, features, labels, x_scaler,
               y_scaler, number_train_run, inf_print, device):
    """Save model together with important parameters

    :parameter
    -----------
    :param model:                                   Simple MLP model
    :param n_input:             - int -             number of input features
    :param n_output:            - int -             number of output features
    :param optimizer:                               pytorch optimizer function, including the learning rate
    :param criterion:                               pytorch loss function
    :param number_net:          - str -             number to identify the network
    :param features:            - list of str -     input features of the network
    :param labels:              - list of str -     output features of the network
    :param x_scaler:                                MinMaxScaler of the samples
    :param y_scaler:                                MinMasxScaler of the labels
    :param number_train_run:    - str -             number to identify the train run used for training
    :param inf_print:           - bool -            should information be printed out
    :param device:              - str -             cpu, single gpu or multi gpu to use for training the model
    """

    # create path for the model
    path = Path(__file__).resolve()
    path_dir = path.parents[2] / 'data/00001-MLP-temperature'
    path_mlp = path.parents[2] / 'data/00001-MLP-temperature/{}_checkpoint.pth'.format(number_net)

    # check if directory already exists
    try:
        os.makedirs(path_dir)
    except OSError:
        if inf_print is True:
            print('Directories already exist')
    else:
        if inf_print is True:
            print('Necessary directories created')

    try:

        if device == 'gpu_multi':

            from collections import OrderedDict
            new_state_dict = OrderedDict()

            checkpoint_unchanged = torch.load('model.pt')
            
            for k, v in checkpoint_unchanged.items():
                name = k.replace("module.", "")  # removing ‘module.’ from key
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(torch.load('model.pt'))

        # Save model with structure
        checkpoint = {'input_size': n_input,
                      'output_size': n_output,
                      'hidden_layers': [each.out_features for each in model.hidden_layers],
                      'optimizer': optimizer.state_dict(),
                      'criterion': criterion.state_dict(),
                      'state_dict': model.state_dict(),
                      'features': features,
                      'labels': labels,
                      'x_scaler': x_scaler,
                      'y_scaler': y_scaler,
                      'number_train_run': number_train_run}

        torch.save(checkpoint, path_mlp)
        print('Model saved ...')

        # Delete the model pt files
        os.remove('model.pt')
    except FileNotFoundError:
        print('Training has not improved model, no new model will be saved ...')
