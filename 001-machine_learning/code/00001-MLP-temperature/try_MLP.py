import argparse
import torch
import fc_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fc_pre_processing_load import loaddata
from torch import nn
from fc_post_processing import load_checkpoint
from pathlib import Path
import fc_post_processing

mechanism_input = 'cai'
number_run = '000'
equivalence_ratio = [0.0]
pressure = [0]
temperature = [0]
pode = [0]
number_net = '002'
number_test_run = '001'
n_epochs = 2
typ = 'train'

# samples, labels, scaler_samples, scaler_labels = fc_pre_processing_load.loaddata_samples\
#     (mechanism_input, number_run, equivalence_ratio=[0], reactorPressure=[0], reactorTemperature=[0], pode=[0],
#      category='train', s_paras=s_paras, l_paras=l_paras)
# fc_post_processing.plot_train(model, samples, labels, scaler_samples, scaler_labels, number_net,
#                                 pode, equivalence_ratio, pressure, temperature)
try:
    model, criterion, s_paras, l_paras, scaler_samples, scaler_labels, n_input, n_output, valid_test_loss_min, \
    valid_train_loss_min = load_checkpoint(number_net, typ)
    print('Pretrained model found, training will be continued ...')
except FileNotFoundError:
    n_input = 5
    n_output = 1
    n_hidden = [32, 32]
    model = fc_model.Net(n_input, n_output, n_hidden)
    criterion = nn.MSELoss()
    print('New model created')
    scaler_samples = None
    scaler_labels = None
    s_paras = ['pode', 'phi', 'P_0', 'T_0', 'PV']
    l_paras = ['T']
    valid_test_loss_min = np.Inf
    valid_train_loss_min = np.Inf

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
print(model)

# %% Load training, validation and test tensors
train_loader, valid_loader, scaler_samples, scaler_labels = loaddata \
    (mechanism_input, number_run, equivalence_ratio, pressure, temperature, pode,
     s_paras, l_paras, category='train', scaler_samples=scaler_samples,
     scaler_labels=scaler_labels)

test_loader = loaddata(mechanism_input, number_test_run, equivalence_ratio, pressure,
                       temperature, pode, s_paras, l_paras,
                       category='test', scaler_samples=scaler_samples, scaler_labels=scaler_labels)

print('Data loaded, start training ...')

# %% number of epochs to train the model
valid_test_loss_min, valid_train_loss_min = fc_model.train(model, train_loader, valid_loader, test_loader, criterion,
                                                           optimizer, n_epochs, number_net,
                                                           valid_test_loss_min, valid_train_loss_min)

# %% save best models depending the validation loss
fc_model.save_model(model, n_input, n_output, optimizer, criterion, number_net, s_paras, l_paras, scaler_samples,
                    scaler_labels, valid_test_loss_min, valid_train_loss_min, typ='train')

fc_model.save_model(model, n_input, n_output, optimizer, criterion, number_net, s_paras, l_paras, scaler_samples,
                    scaler_labels, valid_test_loss_min, valid_train_loss_min, typ='test')

