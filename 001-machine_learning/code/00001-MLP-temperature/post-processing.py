#######################################################################################################################
# build, train and validate the MLP
######################################################################################################################

# %% import packages
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import fc_pre_processing_load
from pathlib import Path
import fc_post_processing
import cantera as ct
import numpy as np


#%% get arguments
parser = argparse.ArgumentParser(description="Run post processing of NN")

parser.add_argument("--post", type=str, choices=['loss', 'test', 'plt_train'], default='test',
                    help="chose which post processing method should be performed")

parser.add_argument("-mech", "--mechanism_input", type=str, choices=['he', 'sun', 'cai'], default='cai',
                    help="chose reaction mechanism")

parser.add_argument("-nbr_run", "--number_test_run", type=str, default='001',
                    help="define which test data should be used")

parser.add_argument("-nbr_net", "--number_net", type=str, default='012',
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


if args.post == 'loss':
    # PLot losses
    path = Path(__file__).resolve()
    path_loss = path.parents[2] / 'data/00001-MLP-temperature/{}_losses.csv'.format(args.number_net)
    losses = pd.read_csv(path_loss)
    
    losses = losses.to_numpy()
    plt.plot(losses[:, 0], losses[:, 1], 'b-', label='training_loss')
    plt.plot(losses[:, 0], losses[:, 2], 'r-', label='valid_loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.legend()
    path_plt_loss = path.parents[2] / 'data/00001-MLP-temperature/{}_plt_losses.pdf'.format(args.number_net)
    plt.savefig(path_plt_loss)
    plt.show()

elif args.post == 'plt_train':
    model, criterion, features, labels, x_scaler, y_scaler, _, _, number_train_run = fc_post_processing.\
        load_checkpoint(args.number_net)

    print(model)

    pode = args.pode
    feature_select = {'pode': pode[0], 'phi': args.equivalence_ratio, 'P_0': args.pressure, 'T_0': args.temperature}

    x_samples, y_samples = fc_pre_processing_load.load_samples(args.mechanism_input, number_train_run, feature_select,
                                                               features, labels, select_data='include',
                                                               category='train')

    fc_post_processing.plot_train(model, x_samples, y_samples, x_scaler, y_scaler, args.number_net, features)

elif args.post == 'test':     # PLot interpolation capability of Network
    # get the model
    model, _, features, labels, x_scaler, y_scaler, _, _, number_train_run = fc_post_processing.\
        load_checkpoint(args.number_net)
    print('Model loaded, begin to load data ...')

    # get train and test samples
    feature_select = {}

    x_train, y_train = fc_pre_processing_load.load_samples(args.mechanism_input, number_train_run,
                                                           feature_select, features, labels, select_data='exclude',
                                                           category='train')

    feature_select = {'pode': args.pode, 'phi': args.equivalence_ratio, 'P_0': args.pressure, 'T_0': args.temperature}

    x_test, y_test = fc_pre_processing_load.load_samples(args.mechanism_input, args.number_test_run,
                                                         feature_select, features, labels,
                                                         select_data='exclude', category='test')

    #  Load  test tensors
    test_loader = fc_pre_processing_load.load_dataloader(x_test, y_test, batch_fraction=100, split=False,
                                                         x_scaler=x_scaler, y_scaler=y_scaler, features=None)

    print('Data loaded!')

    # calculate accuracy
    fc_post_processing.calc_acc(model, test_loader, y_scaler, labels)

    # normalize the data
    x_train, _ = fc_pre_processing_load.normalize_df(x_train, scaler=x_scaler)
    y_train, _ = fc_pre_processing_load.normalize_df(y_train, scaler=y_scaler)

    x_test, _ = fc_pre_processing_load.normalize_df(x_test, scaler=x_scaler)
    y_test, _ = fc_pre_processing_load.normalize_df(y_test, scaler=y_scaler)

    # plot the output of NN and reactor together with the closest parameter in the training set (data between the
    # interpolation took place)
    fc_post_processing.plot_data(model, x_train, y_train, x_test, y_test, x_scaler,
                                 y_scaler, args.number_net, plt_nbrs=False, features=features, labels=labels)

