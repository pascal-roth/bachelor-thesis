import torch
import fc_model
from fc_pre_processing_load import load_samples, load_dataloader
from torch import nn
from fc_post_processing import load_checkpoint

mechanism_input = 'cai'
number_train_run = '000'
equivalence_ratio = [0.0]
pressure = [0]
temperature = [0]
pode = [0]
number_net = '004'
n_epochs = 2
feature_set = 2
plot = True
labels = ['T']

try:
    model, criterion, features, labels, x_scaler, y_scaler, n_input, n_output, _ = load_checkpoint(number_net)
    print('Pretrained model found, training will be continued ...')
except FileNotFoundError:
    n_input = 5
    n_output = 1
    n_hidden = [64, 64, 32]
    model = fc_model.Net(n_input, n_output, n_hidden)
    criterion = nn.MSELoss()
    print('New model created')
    x_scaler = None
    y_scaler = None
    if feature_set == 1:
        features = ['pode', 'phi', 'P_0', 'T_0', 'PV']
    elif feature_set == 2:
        features = ['pode', 'Z', 'U', 'P', 'PV']
    labels = labels

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
print(model)

# %% Load training, validation and test tensors
print('Load data ...')

x_samples, y_samples = load_samples(mechanism_input, number_train_run, equivalence_ratio, pressure,
                                    temperature, pode, features, labels, select_data='exclude',
                                    category='train')

train_loader, valid_loader, x_scaler, y_scaler = load_dataloader(x_samples, y_samples, split=True, x_scaler=x_scaler,
                                                                 y_scaler=y_scaler, features=features)

# free space because only dataloaders needed for training
x_samples = None
y_samples = None

print('Data loaded, start training ...')

# %% number of epochs to train the model
fc_model.train(model, train_loader, valid_loader, criterion, optimizer, n_epochs, number_net, plot=True)

# %% save best models depending the validation loss
fc_model.save_model(model, n_input, n_output, optimizer, criterion, number_net, features, labels, x_scaler,
                    y_scaler, number_train_run)