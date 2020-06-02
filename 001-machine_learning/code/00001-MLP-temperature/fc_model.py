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


# validation function for the model ###################################################################################
def validation(model, valid_loader, test_loader, criterion, valid_test_loss_min, valid_train_loss_min):
    valid_samples_loss = 0
    valid_test_loss = 0

    for data, target in valid_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model.forward(data)
        # calculate the loss
        loss = criterion(output, target)
        # update running validation loss
        valid_samples_loss += loss.item() * data.size(0)

    for data, target in test_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model.forward(data)
        # calculate the loss
        loss = criterion(output, target)
        # update running validation loss
        valid_test_loss += loss.item() * data.size(0)

    valid_samples_loss = valid_samples_loss / len(valid_loader)
    valid_test_loss = valid_test_loss / len(test_loader)

    #     correct = np.zeros((len(output)))
    #
    #     for i in range(len(output)):
    #         if labels.data[i] * 0.95 < output[i] < labels.data[i] * 1.05:
    #             correct[i] = 1
    #         else:
    #             correct[i] = 0
    #
    #     acc = np.append(acc, np.sum(correct) / len(output))
    #
    # acc = np.mean(acc)
    return valid_samples_loss, valid_test_loss, valid_test_loss_min, valid_train_loss_min


# training function for the model ####################################################################################
def train(model, train_loader, val_loader, test_loader, criterion, optimizer, epochs, nbr_net, valid_test_loss_min, valid_train_loss_min):
    losses = np.zeros((epochs, 3))

    for epoch in range(epochs):

        # Model in training mode, dropout is on
        model.train()
        for data, labels in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model.forward(data)
            # calculate the loss
            loss = criterion(output, labels)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            losses[epoch, 0] += loss.item() * data.size(0)

        losses[epoch, 0] = losses[epoch, 0] / len(train_loader)

        # Model in inference mode, dropout is off
        model.eval()

        # Turn off gradients for validation, will speed up inference
        with torch.no_grad():
            losses[epoch, 1], losses[epoch, 2], valid_test_loss_min, valid_train_loss_min = validation \
                (model, val_loader, test_loader, criterion, valid_test_loss_min, valid_train_loss_min)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation loss {:.6f} \tInterpolation loss {:.6f}'.format
              (epoch + 1, losses[epoch, 0], losses[epoch, 1], losses[epoch, 2]))

        # save model if validation loss has decreased
        if losses[epoch, 2] <= valid_test_loss_min:
            print(
                'Interpolation test  loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_test_loss_min,
                                                                                                   losses[epoch, 2]))
            torch.save(model.state_dict(), 'model_test.pt')
            valid_test_loss_min = losses[epoch, 2]

        if losses[epoch, 1] <= valid_train_loss_min:
            print(
                'Interpolation train loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_train_loss_min,
                                                                                                   losses[epoch, 1]))
            torch.save(model.state_dict(), 'model_train.pt')
            valid_train_loss_min = losses[epoch, 1]

    # save losses
    path = Path(__file__).resolve()
    path_loss = path.parents[2] / 'data/00001-MLP-temperature/{}_losses.csv'.format(nbr_net)

    try:
        losses_pre = pd.read_csv(path_loss)
        losses_pre = losses_pre.values
        losses = np.append(losses_pre[:, 1:], losses, axis=0)
        losses = pd.DataFrame(losses)
    except FileNotFoundError:
        losses = pd.DataFrame(losses)

    losses.columns = ['train_loss', 'valid_loss', 'interpolation_loss']
    losses.to_csv(path_loss)

    return valid_test_loss_min, valid_train_loss_min


# save latest models in checkpoint files ##############################################################################
def save_model(model, n_input, n_output, optimizer, criterion, number_net, s_paras, l_paras, scaler_samples,
               scaler_labels, valid_test_loss_min, valid_train_loss_min, number_train_run, typ):

    try:
        model.load_state_dict(torch.load('model_{}.pt'.format(typ)))

        # Save model with structure
        checkpoint = {'input_size': n_input,
                      'output_size': n_output,
                      'hidden_layers': [each.out_features for each in model.hidden_layers],
                      'optimizer': optimizer.state_dict(),
                      'criterion': criterion.state_dict(),
                      'state_dict': model.state_dict(),
                      's_paras': s_paras,
                      'l_paras': l_paras,
                      'scaler_samples': scaler_samples,
                      'scaler_labels': scaler_labels,
                      'valid_test_loss_min': valid_test_loss_min,
                      'valid_train_loss_min': valid_train_loss_min,
                      'number_train_run': number_train_run}

        path = Path(__file__).resolve()
        path_pth = path.parents[2] / 'data/00001-MLP-temperature/{}_{}_checkpoint.pth'.format(number_net, typ)
        torch.save(checkpoint, path_pth)
        print('Model with {} validation saved ...'.format(typ))

        # Delete the model pt files
        os.remove('model_{}.pt'.format(typ))
    except FileNotFoundError:
        print('Training has not improved model with {} validation, no new model will be saved ...'.format(typ))