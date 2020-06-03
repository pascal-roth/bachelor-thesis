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


#%% get arguments
parser = argparse.ArgumentParser(description="Run post processing of NN")

parser.add_argument("--post", type=str, choices=['loss', 'test', 'plt_train'], default='loss',
                    help="chose which post processing method should be performed")

parser.add_argument("-mech", "--mechanism_input", type=str, choices=['he', 'sun', 'cai'], default='cai',
                    help="chose reaction mechanism")

parser.add_argument("-nbr_run", "--number_test_run", type=str, default='000',
                    help="define which test data should be used")

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


if args.post == 'loss':
    # PLot losses
    path = Path(__file__).resolve()
    path_loss = path.parents[2] / 'data/00001-MLP-temperature/{}_losses.csv'.format(args.number_net)
    losses = pd.read_csv(path_loss)
    
    losses = losses.to_numpy()
    plt.plot(losses[:, 0], losses[:, 1], 'r-', label='valid_loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.legend()
    path_plt_loss = path.parents[2] / 'data/00001-MLP-temperature/{}_plt_losses.pdf'.format(args.number_net)
    plt.savefig(path_plt_loss)
    plt.show()

elif args.post == 'plt_train':
    samples, labels, scaler_samples, scaler_labels = fc_pre_processing_load.loaddata_samples \
        (args.mechanism_input, args.number_run, equivalence_ratio=[0], reactorPressure=[0], reactorTemperature=[0],
         pode=[0], category='train')

    model, criterion = fc_post_processing.load_checkpoint(args.number_net)
    print(model)

    fc_post_processing.plot_train(model, samples, labels, scaler_samples, scaler_labels, args.number_net,
                                  args.pode, args.equivalence_ratio, args.pressure, args.temperature)

elif args.post == 'test':     # PLot interpolation capability of Network
    # get the model
    model, criterion, s_paras, l_paras, scaler_samples, scaler_labels, _, _, _, number_train_run = fc_post_processing.load_checkpoint(args.number_net)
    print(model)

    #  Load  test tensors
    test_loader = fc_pre_processing_load.loaddata(args.mechanism_input, args.number_test_run, args.equivalence_ratio,
                                                  args.pressure, args.temperature, args.pode, s_paras, l_paras,
                                                  category='test', scaler_samples=scaler_samples,
                                                  scaler_labels=scaler_labels)

    # calculate accuracy
    acc_mean = fc_post_processing.calc_acc(model, criterion, test_loader, scaler_labels)
    print('The mean accuracy with a 5% tolerance is {}'. format(acc_mean))

    # get samples and labels not included in training data
    train_samples, train_labels = fc_pre_processing_load.loaddata_samples(args.mechanism_input, number_train_run,
                                                                          args.equivalence_ratio, args.pressure,
                                                                          args.temperature, args.pode, category='train',
                                                                          s_paras=s_paras, l_paras=l_paras)

    test_samples, test_labels = fc_pre_processing_load.loaddata_samples(args.mechanism_input, args.number_test_run,
                                                                        args.equivalence_ratio, args.pressure,
                                                                        args.temperature, args.pode, category='test',
                                                                        s_paras=s_paras, l_paras=l_paras)

    train_samples, _ = fc_pre_processing_load.normalize_df(train_samples, scaler=scaler_samples)
    train_labels, _ = fc_pre_processing_load.normalize_df(train_labels, scaler=scaler_labels)

    test_samples, _ = fc_pre_processing_load.normalize_df(test_samples, scaler=scaler_samples)
    test_labels, _ = fc_pre_processing_load.normalize_df(test_labels, scaler=scaler_labels)

    # plot the output of NN and reactor together with the closest parameter in the training set (data between the
    # interpolation took place)
    fc_post_processing.plot_data(model, train_samples, train_labels, test_samples, test_labels, scaler_samples,
                                 scaler_labels, args.number_net, plt_nbrs=True)

