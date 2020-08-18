#######################################################################################################################
# build, train and validate the MLP
######################################################################################################################

# %% import packages
import torch
import fc_model
import numpy as np
import cantera as ct
from torch import nn
from pathlib import Path
from fc_pre_processing_load import normalize_df
import matplotlib.pyplot as plt
plt.style.use('stfs')


# load trained model ##################################################################################################
def load_checkpoint(nbr_net):
    """Load Checkpoint of a saved model

    :parameter
    :param nbr_net: - int -     number to identify the saved MLP
    """

    path = Path(__file__).resolve()
    path_pth = path.parents[2] / 'data/00001-MLP-temperature/{}_checkpoint.pth'.format(nbr_net)
    checkpoint = torch.load(path_pth, map_location=torch.device('cpu'))

    # Create model and load its criterion
    model = fc_model.Net(checkpoint['input_size'],
                         checkpoint['output_size'],
                         checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    criterion = nn.MSELoss()
    criterion.load_state_dict(checkpoint['criterion'])

    # load the parameters of the x_samples and y_samples, as well as the corresponding MinMaxScalers
    number_train_run = checkpoint['number_train_run']
    features = checkpoint['features']
    labels = checkpoint['labels']
    x_scaler = checkpoint['x_scaler']
    y_scaler = checkpoint['y_scaler']

    # parameters needed to save the model again
    n_input = checkpoint['input_size']
    n_output = checkpoint['output_size']

    return model, criterion, features, labels, x_scaler, y_scaler, n_input, n_output, \
           number_train_run


# calculate mean accuracy over test data ###############################################################################
def calc_acc(model, test_loader, scaler, labels):
    """ Calculate the percentage of MLP outputs which are in a range of 5% of the reactor output

    :param model:                           MLP-model
    :param test_loader: - pd dataframe -    data loader which includes data and target of the test samples
    :param scaler:                          MinMaxScaler which has been to used to normalize the samples
    :param labels:      - list of str -     List of labels of the MLP
    """

    model.eval()                                                # prep model for evaluation
    acc_abs = np.zeros((len(test_loader), len(labels)))         # create acc array
    acc_rel = np.zeros((len(test_loader), len(labels)))         # create acc array
    acc_abs_rel = np.zeros((len(test_loader), len(labels)))     # create acc array
    n = 0                                                       # tracking parameter of test_loader length
    abs_range = 0.05
    rel_range = 0.02

    for data, target in test_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # compare predictions to true label
        correct_abs = np.zeros((len(output), len(labels)))
        correct_rel = np.zeros((len(output), len(labels)))
        correct_abs_rel = np.zeros((len(output), len(labels)))
        # convert tensors to numpy arrays
        output = output.detach().numpy()
        target = target.detach().numpy()
        # denormalize target and output, because 5% of the normalized values can mean huge/small differences in the
        # denormalized values
        output_unnormalized = scaler.inverse_transform(output)
        target_unnormalized = scaler.inverse_transform(target)

        # check if samples are in 5% range
        for ii in range(len(labels)):
            target_run_abs = np.squeeze(target_unnormalized[:, ii])
            output_run_abs = np.squeeze(output_unnormalized[:, ii])

            target_run_rel = np.squeeze(target[:, ii])
            output_run_rel = np.squeeze(output[:, ii])

            for i in range(len(output)):
                # check absolute accuracy
                if target_run_abs[i] * (1-abs_range) < output_run_abs[i] < target_run_abs[i] * (1+abs_range):
                    correct_abs[i, ii] = 1
                else:
                    correct_abs[i, ii] = 0

                # check relative accuracy
                if target_run_rel[i] * (1-rel_range) < output_run_rel[i] < target_run_rel[i] * (1+rel_range):
                    correct_rel[i, ii] = 1
                else:
                    correct_rel[i, ii] = 0

                # check absolute accuracy
                if target_run_abs[i] * (1 - abs_range) < output_run_abs[i] < target_run_abs[i] * (1 + abs_range) and \
                        target_run_rel[i] * (1-rel_range) < output_run_rel[i] < target_run_rel[i] * (1+rel_range):
                    correct_abs_rel[i, ii] = 1
                else:
                    correct_abs_rel[i, ii] = 0

            acc_abs[n, ii] = np.sum(correct_abs[:, ii]) / len(output)
            acc_rel[n, ii] = np.sum(correct_rel[:, ii]) / len(output)
            acc_abs_rel[n, ii] = np.sum(correct_abs_rel[:, ii]) / len(output)

        n += 1

    # calculate mean accuracy
    for i in range(len(labels)):
        acc_mean_abs = np.mean(acc_abs[:, i])
        acc_mean_rel = np.mean(acc_rel[:, i])
        acc_mean_abs_rel = np.mean(acc_abs_rel[:, i])
        print('For {} the absolute accuracy is: {:6.5e} \t relative accuracy is: {:6.5e} \t overall accuracy is: '
              '{:6.5e}'.format(labels[i], acc_mean_abs, acc_mean_rel, acc_mean_abs_rel))


# separate initial states, calc output of reactor and call function to plot ###########################################
def plot_data(model, x_train, y_train, x_test, y_test, x_scaler, y_scaler, plt_nbrs, features, labels, args):
    """ Plot the MLP output next to the output of the reactor to see how good the network interpolates

    :parameter
    :param model:                           pytorch MLP model
    :param x_train:     - pd dataframe -    data of the training samples
    :param y_train:     - pd dataframe -    target of the training samples
    :param x_test:      - pd dataframe -    data of the test samples
    :param y_test:      - pd dataframe -    target of the test data
    :param x_scaler:                        MinMaxScaler of the data
    :param y_scaler:                        MinMaxScaler of the targets
    :param number_net:  - int -             Number to identify the MLP
    :param plt_nbrs:    - boolean -         plot the the two parameter settings between them the network interpolated
    :param features:    - list of str -     list of features which has been used for training
    :param labels:      - list of str -     list of labels of the MLP
    """

    # rename the 3 features which identify the initial parameter setting
    x_test = x_test.rename(columns={'{}'.format(features[1]): 'feature_1',
                                    '{}'.format(features[2]): 'feature_2'})

    # if three features to identify state
    if len(features) == 5:
        x_test = x_test.rename(columns={'{}'.format(features[3]): 'feature_3'})
        x_test[['feature_3']] = x_test[['feature_3']].round(decimals=1)

    # separate the different initial conditions and iterate over them
    values_feature_1 = x_test.drop_duplicates(['feature_1'])
    values_feature_1 = values_feature_1[['feature_1']].to_numpy()

    for i, feature_1_run in enumerate(values_feature_1):
        samples_feature_1_run = x_test[x_test.feature_1 == feature_1_run[0]]
        values_feature_2 = samples_feature_1_run.drop_duplicates(['feature_2'])
        values_feature_2 = values_feature_2[['feature_2']].to_numpy()

        for ii, feature_2_run in enumerate(values_feature_2):
            samples_feature_2_run = samples_feature_1_run[samples_feature_1_run.feature_2 == feature_2_run[0]]

            if len(features) == 5:
                values_feature_3 = samples_feature_2_run.drop_duplicates(['feature_3'])
                values_feature_3 = values_feature_3[['feature_3']].to_numpy()

                for iii, feature_3_run in enumerate(values_feature_3):
                    samples_feature_3_run = samples_feature_2_run[samples_feature_2_run.feature_3 == feature_3_run[0]]

                    # decide if the neighbors (nbrs) should be plotted
                    if plt_nbrs:
                        nbr1_sample, nbr1_label, nbr2_sample, nbr2_label = find_neighbors(feature_1_run, feature_2_run,
                                                                                          feature_3_run, x_train,
                                                                                          y_train, features)
                        nbr1_label = y_scaler.inverse_transform(nbr1_label)
                        nbr2_label = y_scaler.inverse_transform(nbr2_label)
                        plt.plot(nbr1_sample[['PV']], nbr1_label, 'g-', label='Nbr1')
                        plt.plot(nbr2_sample[['PV']], nbr2_label, 'y-', label='Nbr2')

                    # for the selected initial parameter setting select the corresponding labels
                    indexes = samples_feature_3_run.index
                    y_test_run = y_test.iloc[indexes, :]
                    y_test_run = y_test_run.to_numpy()

                    # transform data to tensor and compute output of MLP
                    samples_tensor = torch.tensor(samples_feature_3_run.values).float()
                    model.eval()
                    output = model(samples_tensor)

                    # denormalize output for plotting
                    output = output.detach().numpy()
                    output = y_scaler.inverse_transform(output)
                    y_test_run = y_scaler.inverse_transform(y_test_run)

                    plot_outputs(output, y_test_run, samples_feature_3_run, features, labels, x_scaler, args)

            else:

                if plt_nbrs:
                    feature_3_run = None
                    nbr1_sample, nbr1_label, nbr2_sample, nbr2_label = find_neighbors(feature_1_run, feature_2_run,
                                                                                      feature_3_run, x_train,
                                                                                      y_train, features)
                    nbr1_label = y_scaler.inverse_transform(nbr1_label)
                    nbr2_label = y_scaler.inverse_transform(nbr2_label)
                    plt.plot(nbr1_sample[['PV']], nbr1_label, 'g-', label='Nbr1')
                    plt.plot(nbr2_sample[['PV']], nbr2_label, 'y-', label='Nbr2')

                # for the selected initial parameter setting select the corresponding labels
                indexes = samples_feature_2_run.index
                y_test_run = y_test.iloc[indexes, :]
                y_test_run = y_test_run.to_numpy()

                # transform data to tensor and compute output of MLP
                samples_tensor = torch.tensor(samples_feature_2_run.values).float()
                model.eval()
                output = model(samples_tensor)

                # denormalize output for plotting
                output = output.detach().numpy()
                output = y_scaler.inverse_transform(output)
                y_test_run = y_scaler.inverse_transform(y_test_run)

                plot_outputs(output, y_test_run, samples_feature_2_run, features, labels, x_scaler, args)


# find closest runs between test and train samples ####################################################################
def find_neighbors(feature_1_test, feature_2_test, feature_3_test, x_train, y_train, features):
    """ Find the two runs in the training data with the initial conditions which are closest to the test data run
    settings

    :parameter
    :param feature_1_test:  - array -           float entry with the first feature of the training run (Phi or Z)
    :param feature_2_test:  - array -           float entry with the second feature of the training run (P_0 or P)
    :param feature_3_test:  - array -           float entry with the third feature of the training run (T or U/H)
    :param x_train:         - pd dataframe -    data of the training samples
    :param y_train:         - pd dataframe -    targets of the training samples
    :param features:        - list of str -     list of features which has been used for training

    :returns:
    :return nbr1_sample:    - pd dataframe -    data of the first training run with has been used to interpolate
    :return nbr1_label:     - pd dataframe -    targets of the first training run with has been used to interpolate
    :return nbr2_sample:    - pd dataframe -    data of the second training run with has been used to interpolate
    :return nbr2_label:     - pd dataframe -    targets of the second training run with has been used to interpolate
    """

    from sklearn.neighbors import NearestNeighbors

    # rename the 3 features which identify the initial parameter setting
    x_train = x_train.rename(columns={'{}'.format(features[1]): 'feature_1',
                                      '{}'.format(features[2]): 'feature_2'})

    # separate the different initial conditions and create an array with all combinations
    values_feature_1 = x_train.drop_duplicates(subset='feature_1')
    values_feature_1 = values_feature_1[['feature_1']].to_numpy()

    for i, feature_1_run in enumerate(values_feature_1):
        samples_feature_1_run = x_train[x_train.feature_1 == feature_1_run[0]]
        values_feature_2 = samples_feature_1_run.drop_duplicates(subset='feature_2')
        values_feature_2 = values_feature_2[['feature_2']].to_numpy()

        if len(features) == 5:
            x_train = x_train.rename(columns={'{}'.format(features[3]): 'feature_3'})

            for ii, feature_2_run in enumerate(values_feature_2):
                samples_feature_2_run = samples_feature_1_run[samples_feature_1_run.feature_2 == feature_2_run[0]]
                values_feature_3 = samples_feature_2_run.drop_duplicates(subset=['feature_3'])
                values_feature_3 = values_feature_3[['feature_3']].to_numpy()

                combination = np.concatenate((feature_1_run * np.ones((len(values_feature_3), 1)), feature_2_run *
                                              np.ones((len(values_feature_3), 1)), values_feature_3), axis=1)

                if ii == 0 and i == 0:
                    combinations = combination
                else:
                    combinations = np.append(combinations, combination, axis=0)

        else:

            combination = np.concatenate((feature_1_run * np.ones((len(values_feature_2), 1)), values_feature_2), axis=1)

            if ii == 0 and i == 0:
                combinations = combination
            else:
                combinations = np.append(combinations, combination, axis=0)

    # train NearestNeighbors with all combinations of the training data
    combinations.tolist()
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(combinations)

    # find the two closest settings of the test run
    sample = [[feature_1_test[0], feature_2_test[0], feature_3_test[0]]]
    nbrs = neigh.kneighbors(sample, n_neighbors=2, return_distance=False)
    print(nbrs)

    # identify the data and targets of the closest run
    nbr1_paras = combinations[nbrs[0, 0]]
    nbr1_sample = x_train[x_train.feature_1 == nbr1_paras[0]]
    nbr1_sample = nbr1_sample[nbr1_sample.feature_2 == nbr1_paras[1]]
    nbr1_sample = nbr1_sample[nbr1_sample.feature_3 == nbr1_paras[2]]
    indexes = nbr1_sample.index
    nbr1_label = y_train.iloc[indexes, :]

    # identify the data and targets of the second clostest run
    nbr2_paras = combinations[nbrs[0, 1]]
    nbr2_sample = x_train[x_train.feature_1 == nbr2_paras[0]]
    nbr2_sample = nbr2_sample[nbr2_sample.feature_2 == nbr2_paras[1]]
    nbr2_sample = nbr2_sample[nbr2_sample.feature_3 == nbr2_paras[2]]
    indexes = nbr2_sample.index
    nbr2_label = y_train.iloc[indexes, :]

    return nbr1_sample, nbr1_label, nbr2_sample, nbr2_label


# function to plot MLP output next to reactor output ##################################################################
def plot_outputs(output, y_test, samples_feature_run, features, labels, x_scaler, args):
    """
    Function to plot MLP output next to reactor output

    :param output:              - np array -        Output of the MLP
    :param y_test:              - np array -        corresponding output of the reactor
    :param samples_feature_run: - pd dataframe -    test samples which were given to MLP
    :param features:            - list of str -     list of features
    :param labels:              - list of str -     list of labels
    :param x_scaler:                                MinMaxScaler of samples
    :param args:                - argparse -        Input parameters given to the script
    """

    for i in range(len(labels)):
        if labels[i] == 'P':
            y_test_run = np.squeeze(y_test[:, i]) / 1.e+6
            output_run = np.squeeze(output[:, i]) / 1.e+6
        else:
            y_test_run = np.squeeze(y_test[:, i])
            output_run = np.squeeze(output[:, i])

        # plot the MLP and reactor output
        plt.plot(samples_feature_run[['PV']], y_test_run, 'b-', label='HR Output')
        plt.plot(samples_feature_run[['PV']], output_run, 'r-', label='NN Output')

        samples_feature_index = samples_feature_run.to_numpy()
        samples_feature_index = x_scaler.inverse_transform(samples_feature_index)

        if len(features) == 5:
            plt.title('PODE{:.0f} {}={:.2f} {}={:.0f} {}={}bar '.format(samples_feature_index[0, 0],
                                                                    features[1],
                                                                    samples_feature_index[0, 1],
                                                                    features[2],
                                                                    samples_feature_index[0, 2] / ct.one_atm,
                                                                    features[3],
                                                                    samples_feature_index[0, 3]))

            path = Path(__file__).resolve()
            path_plt = path.parents[2] / 'data/00001-MLP-temperature/{}_plt_{}_comp_PODE{}_{:.2f}_{:.0f}_{:.0f}.pdf'.\
                format(args.number_net, labels[i], samples_feature_index[0, 0], samples_feature_index[0, 1],
                       samples_feature_index[0, 2] / ct.one_atm, samples_feature_index[0, 3])


        else:
            plt.title('PODE{:.0f} {}={:.2f} {}={:.2f} [MJ/kg]'.format(samples_feature_index[0, 0],
                                                                      features[1],
                                                                      samples_feature_index[0, 1],
                                                                      features[2],
                                                                      samples_feature_index[0, 2] / 1.e+3))

            path = Path(__file__).resolve()
            path_plt = path.parents[2] / 'data/00001-MLP-temperature/{}_plt_{}_comp_PODE{}_{:.2f}_{:.0f}.pdf'.format \
                (args.number_net, labels[i], samples_feature_index[0, 0], samples_feature_index[0, 1],
                 samples_feature_index[0, 2])

        plt.legend()
        plt.xlabel('PV')
        label_unit = unit(labels[i])
        plt.ylabel('{}'.format(label_unit))

        plt.savefig(path_plt)

        plt.show()


# function to plot the fitting to the training data ###################################################################
def plot_train(model, x_samples, y_samples, x_scaler, y_scaler, number_net, features, labels):
    """Plot a training run to see how good model is already fitted to training data

    :param model:                           pytorch MLP model
    :param x_samples:   - pd dataframe -    data of the training samples
    :param y_samples:   - pd dataframe -    target of the training samples
    :param x_scaler:                        MinMaxScaler of the data
    :param y_scaler:                        MinMaxScaler of the targets
    :param number_net:  - int -             Number to identify the MLP
    :param features:    - list of str -     list of features which has been used for training
    :param labels:      - list of str -     list of labels
    """

    # calculate output of the NN
    output = NN_output(model, x_samples, x_scaler, y_scaler)

    y_samples = y_samples.to_numpy
    x_samples_normalized, _ = normalize_df(x_samples, scaler=x_scaler)

    plot_outputs(output, y_samples, x_samples_normalized, features, labels, x_scaler, number_net)


def unit(label):
    """ Corresponding unit for the selected label

    :parameter
    :param label:       - str -             label

    :returns:
    :return label_unit: - str -             label with corresponding unit
    """

    if label == 'P':
        label_unit = 'P [MPa]'
    elif label == 'T':
        label_unit = 'T [K]'
    elif label == 'HRR':
        label_unit = "HRR [W/$m^3$/kg]"
    else:
        label_unit = 'Y_{}'.format(label)

    return label_unit


def plot_IDT(IDT_MLP_PV, IDT_HR_PV, args):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    ax.semilogy(1000 / IDT_MLP_PV[:, 0], IDT_MLP_PV[:, 1], 'r-', label='MLP')
    ax.semilogy(1000 / IDT_HR_PV[:, 0], IDT_HR_PV[:, 1], 'b-', label='HR')

    ax.set_ylabel('$Y_c$ at ignition')
    ax.set_xlabel('1000/T [1/K]')

    # Add a second axis on top to plot the temperature for better readability
    ax2 = ax.twiny()
    ticks = ax.get_xticks()
    ax2.set_xticks(ticks)
    ax2.set_xticklabels((1000 / ticks).round(1))
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel('T [K]')

    textstr = '$\\Phi$={:.2f}\np={:.0f}bar\nPODE{}'.format(args.equivalence_ratio[0], args.pressure[0], args.pode[0])
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top')

    ax.set_yscale('log')

    ax.legend(loc='lower right')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=4,
    #           prop={'size': 14})

    plt.tight_layout()

    path = Path(__file__).resolve()
    path = path.parents[2] / 'data/00001-MLP-temperature/{}_plt_ID_pode{}_phi{}_p{}.pdf' \
        .format(args.number_net, args.pode[0], args.equivalence_ratio[0], args.pressure[0])
    plt.savefig(path)

    plt.show()


def NN_output(model, x_samples, x_scaler, y_scaler):
    x_samples_normalized, _ = normalize_df(x_samples, scaler=x_scaler)
    x_samples_normalized = torch.tensor(x_samples_normalized.values).float()
    model.eval()
    y_samples_nn = model.forward(x_samples_normalized)
    y_samples_nn = y_samples_nn.detach().numpy()
    y_samples_nn = y_scaler.inverse_transform(y_samples_nn)

    return y_samples_nn