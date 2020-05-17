#######################################################################################################################
# MLP to predict the temperature for each value of the PV while only the input of the homogeneous reactor is given
#######################################################################################################################

# import packages
import pandas as pd
import numpy as np
import argparse
from fc_pre_processing_load import loaddata
from fc_post_processing import load_checkpoint
import torch
from torch import nn
from pathlib import Path
import fc_model

# %% Collect arguments
parser = argparse.ArgumentParser(description="Run homogeneous reactor model")

parser.add_argument("-mech", "--mechanism_input", type=str, choices=['he', 'sun', 'cai'], default='cai',
                    help="chose reaction mechanism")

parser.add_argument("-nbr_run", "--number_run", type=str, default='000',
                    help="define which training data should be used")

parser.add_argument("-nbr_test", "--number_test_run", type=str, default='000',
                    help="define which test data should be for validation")

parser.add_argument("--n_epochs", type=int, default=50,
                    help="chose number of epochs for training")

parser.add_argument("-nbr_net", "--number_net", type=str, default='000',
                    help="chose number of the network")

parser.add_argument("-phi", "--equivalence_ratio", nargs='+', type=float, default=[0.0],
                    help="chose equivalence ratio")

parser.add_argument("-p", "--pressure", nargs='+', type=int, default=[0],
                    help="chose reactor pressure")

parser.add_argument("--pode", type=int, nargs='+', default=[0],
                    help="chose degree of polymerization")

parser.add_argument("-temp", "--temperature", type=int, nargs='+', default=[0],
                    help="chose certain starting temperatures")

parser.add_argument("-inf_print", "--information_print", default=True, action='store_false',
                    help="chose if basic information are displayed")

args = parser.parse_args()
if args.information_print is True:
    print('\n{}\n'.format(args))

# %% Load training, validation and test tensors
train_loader, valid_loader = loaddata(args.mechanism_input, args.number_run, args.equivalence_ratio, args.pressure,
                                      args.temperature, args.pode, category='train', val_split=True)

test_loader, scaler = loaddata(args.mechanism_input, args.number_test_run, args.equivalence_ratio, args.pressure,
                               args.temperature, args.pode, category='test', val_split=False)

# %% Network implementation
n_input = 5
n_output = 1
n_hidden = [32, 32]

try:
    model, criterion = load_checkpoint(args.number_net)
    print('Pretrained model found, training will be continued ...')
except FileNotFoundError:
    model = fc_model.Net(n_input, n_output, n_hidden)
    criterion = nn.MSELoss()
    print('New model created')

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
print(model)

# number of epochs to train the model
losses = fc_model.train(model, train_loader, valid_loader, test_loader, criterion, optimizer, args.n_epochs)
model.load_state_dict(torch.load('model.pt'))

# save losses
path = Path(__file__).resolve()
path_loss = path.parents[2] / 'data/00001-MLP-temperature/{}_losses.csv'.format(args.number_net)


try:
    losses_pre = pd.read_csv(path_loss)
    losses_pre = losses_pre.values
    losses = np.append(losses_pre[:, 1:], losses, axis=0)
    losses = pd.DataFrame(losses)
except FileNotFoundError:
    losses = pd.DataFrame(losses)

losses.columns = ['train_loss', 'valid_loss']
losses.to_csv(path_loss)

# Save model with structure
checkpoint = {'input_size': n_input,
              'output_size': n_output,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'optimizer': optimizer.state_dict(),
              'criterion': criterion.state_dict(),
              'state_dict': model.state_dict()}
path_pth = path.parents[2] / 'data/00001-MLP-temperature/{}_checkpoint.pth'.format(args.number_net)
torch.save(checkpoint, path_pth)
print('Model saved ...')