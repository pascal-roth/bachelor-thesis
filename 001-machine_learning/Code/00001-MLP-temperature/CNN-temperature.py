# import packages
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
from torch.utils import data
from sklearn import preprocessing


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
    samples = samples.iloc[:(pos + 100), :]
    labels = labels.iloc[:(pos + 100), :]
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
num_workers = 0

train_loader = torch.utils.data.DataLoader(tensor_train, batch_size=batch_size, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(tensor_train, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(tensor_test, batch_size=batch_size, num_workers=num_workers)

# %% decide if model testing with CPU or training with cluster
# train_on_gpu = torch.cuda.is_available()
train_on_gpu = False


# %% Network implementation
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 9x1 image tensor)
        self.conv1 = nn.Conv1d(1, 16, (3, 1), padding=1)
        # convolutional layer (sees 3x16 tensor)
        self.conv2 = nn.Conv1d(16, 32, (3, 1), padding=1)
        # convolutional layer (sees 1x32 tensor)
        self.conv3 = nn.Conv1d(32, 64, (1, 1), padding=1)
        # max pooling layer
        self.pool = nn.MaxPool1d(2, 1)
        # linear layer (32 -> 512)
        self.fc1 = nn.Linear(32, 512)
        # linear layer (512 -> 1)
        self.fc2 = nn.Linear(512, 1)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 9)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x


# initialize the NN
model = Net()
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

# initialize tracker for minimum validation loss
valid_loss_min = np.Inf  # set initial "min" to infinity

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    valid_loss = 0.0

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
        train_loss += loss.item() * data.size(0)

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
        valid_loss += loss.item() * data.size(0)

    # print training/validation statistics
    # calculate average loss over an epoch
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation loss {:.6f}'.format(epoch + 1, train_loss, valid_loss))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), 'model.pt')
        valid_loss_min = valid_loss

model.load_state_dict(torch.load('model.pt'))

# initialize lists to monitor test loss and accuracy
test_loss = 0.0

# %%
model.eval()  # prep model for evaluation

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

    acc = np.sum(correct) / len(output)
    print(acc)

