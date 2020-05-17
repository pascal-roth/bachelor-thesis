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

parser.add_argument("-nbr_run", "--number_run", type=str, default='000',
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
    plt.plot(losses[:, 0], losses[:, 1], 'b-', label='train_loss')
    plt.plot(losses[:, 0], losses[:, 2], 'r-', label='valid_loss')
    plt.plot(losses[:, 0], losses[:, 3], 'r-', label='interpolation_loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.legend()
    path_plt_loss = path.parents[2] / 'data/00001-MLP-temperature/{}_plt_losses.pdf'.format(args.number_net)
    plt.savefig(path_plt_loss)
    plt.show()
    
elif args.post == 'test':     # PLot interpolation capability of Network
    #  Load  test tensors
    test_loader, scaler = fc_pre_processing_load.loaddata(args.mechanism_input, args.number_run,
                                                          args.equivalence_ratio, args.pressure,
                                                          args.temperature, args.pode, category='test',
                                                          val_split=False)

    # get the model
    model, criterion = fc_post_processing.load_checkpoint(args.number_net)
    print(model)

    # calculate accuracy
    acc_mean = fc_post_processing.calc_acc(model, criterion, test_loader, scaler)
    print('The mean accuracy with a 5% tolerance is {}'. format(acc_mean))

    # get samples and labels not prepared in tensor
    samples, labels, scaler_samples, scaler_labels = fc_pre_processing_load.loaddata_samples\
        (args.mechanism_input, args.number_run, args.equivalence_ratio, args.pressure, args.temperature, args.pode,
         category='test')

    # plot the output of NN and reactor
    fc_post_processing.plot_data(model, samples, labels, scaler_samples, scaler_labels, args.number_net)

elif args.post == 'plt_train':
    samples, labels, scaler_samples, scaler_labels = fc_pre_processing_load.loaddata_samples \
        (args.mechanism_input, args.number_run, equivalence_ratio=[0], reactorPressure=[0], reactorTemperature=[0],
         pode=[0], category='train')

    model, criterion = fc_post_processing.load_checkpoint(args.number_net)
    print(model)

    fc_post_processing.plot_train(model, samples, labels, scaler_samples, scaler_labels, args.number_net,
                                  args.pode, args.equivalence_ratio, args.pressure, args.temperature)