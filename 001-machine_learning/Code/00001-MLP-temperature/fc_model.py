import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self, n_input, hidden_layers, n_output, drop_p):
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(n_input, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], n_output)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):

        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return x


def validation(model, valid_loader, criterion, valid_loss_min):
    valid_loss = 0
    acc = []
    for data, labels in valid_loader:
        output = model.forward(data)
        valid_loss += criterion(output, labels).item()

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss

        correct = np.zeros((len(output)))

        for i in range(len(output)):
            if labels.data[i] * 0.95 < output[i] < labels.data[i] * 1.05:
                correct[i] = 1
            else:
                correct[i] = 0

        acc = np.append(acc, np.sum(correct) / len(output))

    acc = np.mean(acc)
    return valid_loss, acc, valid_loss_min


def train(model, trainloader, testloader, criterion, optimizer, epochs):
    valid_loss_min = np.Inf
    steps = 0
    running_loss = 0
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for data, labels in trainloader:
            steps += 1

            optimizer.zero_grad()

            output = model.forward(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Model in inference mode, dropout is off
        model.eval()

        # Turn off gradients for validation, will speed up inference
        with torch.no_grad():
            test_loss, accuracy, valid_loss_min = validation(model, testloader, criterion, valid_loss_min)

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.6f}.. ".format(running_loss),
                  "Validation Loss: {:.6f}.. ".format(test_loss / len(testloader)),
                  "Validation Accuracy: {:.6f}".format(accuracy / len(testloader)))