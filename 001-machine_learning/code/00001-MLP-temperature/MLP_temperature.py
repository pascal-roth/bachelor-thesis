#######################################################################################################################
# MLP to predict the temperature for each value of the PV while only the input of the homogeneous reactor is given
#######################################################################################################################

# import packages
import argparse
from fc_pre_processing_load import loaddata
from fc_post_processing import load_checkpoint
import torch
from torch import nn
import fc_model
import numpy as np

# %% Collect arguments
parser = argparse.ArgumentParser(description="Run homogeneous reactor model")

parser.add_argument("-mech", "--mechanism_input", type=str, choices=['he', 'sun', 'cai'], default='cai',
                    help="chose reaction mechanism")

parser.add_argument("-nbr_run", "--number_train_run", type=str, default='000',
                    help="define which training data should be used")

parser.add_argument("-nbr_test", "--number_test_run", type=str, default='000',
                    help="define which test data should be for validation")

parser.add_argument("-s_paras", "--sample_parameters", nargs='+', type=str, default=['pode', 'phi', 'T_0', 'P_0', 'PV'],
                    help="chose input parameters for the NN")

parser.add_argument("-l_paras", "--label_parameters", nargs='+', type=str, default=['T'],
                    help="chose output parameters for the NN")

parser.add_argument("--n_epochs", type=int, default=50,
                    help="chose number of epochs for training")

parser.add_argument("-nbr_net", "--number_net", type=str, default='000',
                    help="chose number of the network")

parser.add_argument("--typ", type=str, choices=['train', 'test'], default='test',
                    help="chose validation method of pre-trained NN")

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

# %% Network implementation
try:
    model, criterion, s_paras, l_paras, scaler_samples, scaler_labels, n_input, n_output, valid_test_loss_min, \
    valid_train_loss_min, _ = load_checkpoint(args.number_net, args.typ)
    print('Pretrained model found, training will be continued ...')
except FileNotFoundError:
    n_input = 5
    n_output = 1
    n_hidden = [64, 64, 32]
    model = fc_model.Net(n_input, n_output, n_hidden)
    criterion = nn.MSELoss()
    print('New model created')
    scaler_samples = None
    scaler_labels = None
    s_paras = args.sample_parameters
    l_paras = args.label_parameters
    valid_test_loss_min = np.Inf
    valid_train_loss_min = np.Inf

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
print(model)

# %% Load training, validation and test tensors
train_loader, valid_loader, scaler_samples, scaler_labels = loaddata \
    (args.mechanism_input, args.number_train_run, args.equivalence_ratio, args.pressure, args.temperature, args.pode,
     s_paras, l_paras, category='train', scaler_samples=scaler_samples,
     scaler_labels=scaler_labels)

test_loader = loaddata(args.mechanism_input, args.number_test_run, args.equivalence_ratio, args.pressure,
                       args.temperature, args.pode, s_paras, l_paras,
                       category='test', scaler_samples=scaler_samples, scaler_labels=scaler_labels)

print('Data loaded, start training ...')

# %% number of epochs to train the model
valid_test_loss_min, valid_train_loss_min = fc_model.train(model, train_loader, valid_loader, test_loader, criterion,
                                                           optimizer, args.n_epochs, args.number_net,
                                                           valid_test_loss_min, valid_train_loss_min)

# %% save best models depending the validation loss
fc_model.save_model(model, n_input, n_output, optimizer, criterion, args.number_net, s_paras, l_paras, scaler_samples,
                    scaler_labels, valid_test_loss_min, valid_train_loss_min, args.number_train_run, typ='train')

fc_model.save_model(model, n_input, n_output, optimizer, criterion, args.number_net, s_paras, l_paras, scaler_samples,
                    scaler_labels, valid_test_loss_min, valid_train_loss_min, args.number_train_run, typ='test')
