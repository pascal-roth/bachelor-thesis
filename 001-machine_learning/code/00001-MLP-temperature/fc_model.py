#######################################################################################################################
# functions to automatically construct and find the best model
######################################################################################################################

# import packages
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


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
def validation(model, valid_loader, test_loader, criterion, valid_loss_min):
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

    valid_samples_loss= valid_samples_loss/len(valid_loader)
    valid_test_loss = valid_test_loss/len(test_loader)

    # save model if validation loss has decreased
    if valid_test_loss <= valid_loss_min:
        print('Interpolation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                           valid_test_loss))
        torch.save(model.state_dict(), 'model.pt')
        valid_loss_min = valid_test_loss

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
    return valid_samples_loss, valid_test_loss, valid_loss_min


# training function for the model ####################################################################################
def train(model, train_loader, val_loader, test_loader, criterion, optimizer, epochs):
    valid_loss_min = np.Inf
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

        losses[epoch, 0] = losses[epoch, 0]/len(train_loader)

        # Model in inference mode, dropout is off
        model.eval()

        # Turn off gradients for validation, will speed up inference
        with torch.no_grad():
            losses[epoch, 1], losses[epoch, 2], valid_loss_min = validation(model, val_loader, test_loader, criterion,
                                                                            valid_loss_min)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation loss {:.6f} \tInterpolation loss {:.6f}'.format
              (epoch + 1, losses[epoch, 0], losses[epoch, 1], losses[epoch, 2]))

    return losses
