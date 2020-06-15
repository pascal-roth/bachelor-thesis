#######################################################################################################################
# MLP to predict the temperature for each value of the PV while only the input of the homogeneous reactor is given
#######################################################################################################################

# import packages
import argparse
from fc_pre_processing_load import load_samples, load_dataloader
from fc_post_processing import load_checkpoint
import torch
from torch import nn
import fc_model

# %% Collect arguments
parser = argparse.ArgumentParser(description="Run homogeneous reactor model")

parser.add_argument("-mech", "--mechanism_input", type=str, choices=['he', 'sun', 'cai'], default='cai',
                    help="chose reaction mechanism")

parser.add_argument("-nbr_run", "--number_train_run", type=str, default='000',
                    help="define which training data should be used")

parser.add_argument("--feature_set", type=int, choices=[1, 2, 3],
                    help="chose set of features")

parser.add_argument("--hidden", nargs='+', type=int, default=[64, 64, 64],
                    help="chose size of NN")

parser.add_argument("--labels", nargs='+', type=str, default=['T'],
                    help="chose output parameters for the NN")

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

# %% Network implementation
try:
    model, criterion, features, labels, x_scaler, y_scaler, n_input, n_output, _ = load_checkpoint(args.number_net)
    print('Pretrained model found, training will be continued ...')
except FileNotFoundError:
    x_scaler = None
    y_scaler = None

    # Select which features and labels to use
    if args.feature_set == 1:
        features = ['pode', 'phi', 'P_0', 'T_0', 'PV']
    elif args.feature_set == 2:
        features = ['pode', 'Z', 'H', 'P', 'PV']
    elif args.feature_set == 3:
        features = ['pode', 'Z', 'H', 'PV']
    labels = args.labels

    # Parameters of the NN
    n_input = len(features)
    n_output = len(args.labels)
    n_hidden = args.hidden
    model = fc_model.Net(n_input, n_output, n_hidden)
    criterion = nn.MSELoss()
    print('New model created')

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
print(model)

# %% Load training, validation and test tensors
print('Load data ...')
feature_select = {'pode': args.pode, 'phi': args.equivalence_ratio, 'P_0': args.pressure, 'T_0': args.temperature}

x_samples, y_samples = load_samples(args.mechanism_input, args.number_train_run, feature_select, features, labels,
                                    select_data='exclude', category='train')

train_loader, valid_loader, x_scaler, y_scaler = load_dataloader(x_samples, y_samples, split=True, x_scaler=x_scaler,
                                                                 y_scaler=y_scaler, features=features)

# free space because only dataloaders needed for training
x_samples = None
y_samples = None

print('Data loaded, start training ...')

# %% number of epochs to train the model
fc_model.train(model, train_loader, valid_loader, criterion, optimizer, args.n_epochs, args.number_net, plot=True)

# %% save best models depending the validation loss
fc_model.save_model(model, n_input, n_output, optimizer, criterion, args.number_net, features, labels, x_scaler,
                    y_scaler, args.number_train_run)
