# import packages
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
from torch.utils import data
from sklearn import preprocessing
import matplotlib.pyplot as plt


def loaddata_samples(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, pode, t_start, t_end, t_step):
    path = Path(__file__).parents[
               3] / '000-homogeneous_reactor/data/00002-reactor-OME/{}_PODE{}_{}_{:.0f}_{}_{}_{}/samples_{}.csv'.format(
        mechanism[0], pode, equivalence_ratio, reactorPressure, t_start, t_end,
        t_step, reactorTemperature)
    data = pd.read_csv(path)
    max_T = np.amax(data[['T']])
    return data, max_T


def loaddata_delays(mechanism, equivalence_ratio, reactorPressure, pode, t_start, t_end, t_step):
    path = Path(__file__).parents[
               3] / '000-homogeneous_reactor/data/00002-reactor-OME/{}_PODE{}_{}_{:.0f}_{}_{}_{}/delays.csv'.format(
        mechanism[0], pode, equivalence_ratio, reactorPressure, t_start, t_end, t_step)
    data = pd.read_csv(path)
    return data


def normalize_df(df):
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df


def cut_data(samples, labels):
    pos = np.argmax(samples[['PV']])
    print(pos)
    samples = samples.iloc[:(pos + 1), :]
    labels = labels.iloc[:(pos + 1), :]
    return samples, labels


mechanism_all = np.array([['he_2018.xml'], ['cai_ome14_2019.xml'], ['sun_2017.xml']])
mechanism = mechanism_all[1]
equivalence_ratio = 1.0
reactorPressure = 20
reactorTemperature = 740
pode = 3
t_start = 650
t_end = 1250
t_step = 15

data_h, max_T = loaddata_samples(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, pode, t_start,
                                 t_end, t_step)
samples = data_h[['PV', 'phi', 'Q', 'P', 'PODE', 'CO2', 'O2', 'CO', 'H2O']]
labels = data_h[['T']]

samples, labels = cut_data(samples, labels)

samples = normalize_df(samples)
labels = normalize_df(labels)

# %% Separate Data in train, test and validation Data
# Validation split can probably be performed by pytorch later in the NN
from sklearn.model_selection import train_test_split

x_train, y_train, x_test, y_test = train_test_split(samples, labels, test_size=0.2)

x_train = torch.tensor(x_train.values).float()
y_train = torch.tensor(y_train.values).float()
x_test = torch.tensor(x_test.values).float()
y_test = torch.tensor(y_test.values).float()

tensor_train = data.TensorDataset(x_train, x_test)
tensor_test = data.TensorDataset(y_train, y_test)

# prepare data loaders
batch_size = 100
num_workers = 8

train_loader = torch.utils.data.DataLoader(tensor_train, batch_size=batch_size, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(tensor_train, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(tensor_test, batch_size=batch_size, num_workers=num_workers)

# %% decide if model testing with CPU or training with cluster
# train_on_gpu = torch.cuda.is_available()
train_on_gpu = False


# %% Network implementation
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


# initialize the NN
n_input = 9
hidden_layers = [256, 128]
n_output = 1
drop_p = 0.2

model = Net(n_input, hidden_layers, n_output, drop_p)
print(model)

# %%

# specify loss function (categorical cross-entropy)
# criterion = nn.KLDivLoss()
criterion = nn.MSELoss()

# specify optimizer (stochastic gradient descent) and learning rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# %%
# number of epochs to train the model
n_epochs = 75

# initialize tracker
valid_loss_min = 10 #np.Inf  # set initial "min" to infinity
train_loss = np.zeros((n_epochs))
valid_loss = np.zeros((n_epochs))

for epoch in range(n_epochs):
    ###################
    # train the model #
    ###################
    model.train()  # prep model for training
    for data, target in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss[epoch] += loss.item() * data.size(0)

    ######################
    # validate the model #
    ######################
    model.eval()  # prep model for evaluation
    for data, target in valid_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update running validation loss
        valid_loss[epoch] += loss.item() * data.size(0)

    # print training/validation statistics
    # calculate average loss over an epoch
    #    train_loss = train_loss / len(train_loader.sampler)
    #    valid_loss = valid_loss / len(valid_loader.sampler)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation loss {:.6f}'.format(epoch + 1, train_loss[epoch],
                                                                              valid_loss[epoch]))

    # save model if validation loss has decreased
    if valid_loss[epoch] <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                        valid_loss[epoch]))
        torch.save(model.state_dict(), 'model.pt')
        valid_loss_min = valid_loss[epoch]

model.load_state_dict(torch.load('model.pt'))

# initialize lists to monitor test loss and accuracy
test_loss = 0.0

# %%
model.eval()  # prep model for evaluation
acc = []
for data, target in test_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss
    test_loss += loss.item() * data.size(0)
    # compare predictions to true label
    correct = np.zeros((len(output)))

    for i in range(len(output)):
        if target.data[i] * 0.95 < output[i] < target.data[i] * 1.05:
            correct[i] = 1
        else:
            correct[i] = 0

    acc = np.append(acc, np.sum(correct) / len(output))

acc_mean = np.mean(acc)
print(acc_mean)

# %% Save model with structure and
checkpoint = {'input_size': n_input,
              'output_size': n_output,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'drop_p': drop_p,
              'optimizer': optimizer.state_dict(),
              'criterion': criterion.state_dict(),
              'state_dict': model.state_dict()}

path = Path(__file__).parents[2] / 'Data/00001-MLP-temperature/checkpoint.pth'

torch.save(checkpoint, path)

#%% PLot losses
plt.plot(train_loss[1:], 'b-', label='training loss')
plt.plot(valid_loss[1:], 'r-', label='validation loss')
plt.legend()
plt.show()