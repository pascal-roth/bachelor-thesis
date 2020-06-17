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
    checkpoint = torch.load(path_pth)

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

    :return acc_mean:   - float -           return the accuracy of the model
    """

    model.eval()                                            # prep model for evaluation
    acc = np.zeros((len(test_loader), len(labels)))         # create acc array
    n = 0                                                   # tracking parameter of test_loader length
    for data, target in test_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # compare predictions to true label
        correct = np.zeros((len(output), len(labels)))
        # convert tensors to numpy arrays
        output = output.detach().numpy()
        target = target.detach().numpy()
        # denormalize target and output, because 5% of the normalized values can mean huge/small differences in the
        # denormalized values
        output = scaler.inverse_transform(output)
        target = scaler.inverse_transform(target)

        # check if samples are in 5% range
        for ii in range(len(labels)):
            target_run = np.squeeze(target[:, ii])
            output_run = np.squeeze(output[:, ii])

            for i in range(len(output)):
                if target_run[i] * 0.95 < output_run[i] < target_run[i] * 1.05:
                    correct[i, ii] = 1
                else:
                    correct[i, ii] = 0

            acc[n, ii] = np.sum(correct[:, ii]) / len(output)

    # calculate mean accuracy
    acc_mean = np.mean(acc)
    print(acc_mean)

    return acc_mean


# plot output of the reactor next to the output of the NN #############################################################
def plot_data(model, x_train, y_train, x_test, y_test, x_scaler, y_scaler, number_net, plt_nbrs, features, labels):
    """ Plot the MLP output next to the output of the reactor to see how good the network interpolates

    :parameter
    :param model:                           pytorch MLP model
    :param x_train:     - pd dataframe -    data of the training samples
    :param y_train:     - pd dataframe -    target of the training samples
    :param x_test:      - pd dataframe -    data of the test samples
    :param y_test:      - pd dataframe -    target of the test data
    :param x_scaler:                        MinMaxScaler of the data
    :param y_scaler:                        MinMaxScaler of the targets
    :param number_net:  - int -             Number to identify the MLPNumber to identify the MLP
    :param plt_nbrs:    - boolean -         plot the the two parameter settings between them the network interpolated
    :param features:    - list of str -     list of features which has been used for training
    :param labels:      - list of str -     list of labels of the MLP
    """

    # rename the 3 features which identify the initial parameter setting
    x_test = x_test.rename(columns={'{}'.format(features[1]): 'feature_1',
                                    '{}'.format(features[2]): 'feature_2',
                                    '{}'.format(features[3]): 'feature_3'})

    # round the third feature
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
            values_feature_3 = samples_feature_2_run.drop_duplicates(['feature_3'])
            values_feature_3 = values_feature_3[['feature_3']].to_numpy()

            for iii, feature_3_run in enumerate(values_feature_3):
                samples_feature_3_run = samples_feature_2_run[samples_feature_2_run.feature_3 == feature_3_run[0]]

                # decide if the neighbors (nbrs) should be plotted
                if plt_nbrs:
                    nbr1_sample, nbr1_label, nbr2_sample, nbr2_label = find_neighbors(feature_1_run, feature_2_run,
                                                                                      feature_3_run, x_train, y_train,
                                                                                      features)
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

                for iiii in range(len(labels)):

                    # plot the MLP and reactor output
                    plt.plot(samples_feature_3_run[['PV']], y_test_run, 'b-', label='Reactor output')
                    plt.plot(samples_feature_3_run[['PV']], output, 'r-', label='NN Output')

                    samples_feature_3_run = samples_feature_3_run.to_numpy()
                    samples_feature_3_run = x_scaler.inverse_transform(samples_feature_3_run)

                    plt.title('PODE{} {}={:.2f} {}={}bar {}={:.0f}'.format(samples_feature_3_run[0, 0],
                                                                           features[1],
                                                                           samples_feature_3_run[0, 1],
                                                                           features[2],
                                                                           samples_feature_3_run[0, 2] / ct.one_atm,
                                                                           features[3],
                                                                           samples_feature_3_run[0, 3]))

                    plt.legend()
                    plt.xlabel('PV')
                    plt.ylabel('T [K]')

                    path = Path(__file__).resolve()
                    path_plt = path.parents[2] / 'data/00001-MLP-temperature/{}_plt_comp_PODE{}_{}_{}_{}.pdf'.format \
                        (number_net, samples_feature_3_run[0, 0], samples_feature_3_run[0, 1],
                         samples_feature_3_run[0, 2] / ct.one_atm, samples_feature_3_run[0, 3])
                    plt.savefig(path_plt)

                plt.show()


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
                                      '{}'.format(features[2]): 'feature_2',
                                      '{}'.format(features[3]): 'feature_3'})

    # separate the different initial conditions and create an array with all combinations
    values_feature_1 = x_train.drop_duplicates(subset='feature_1')
    values_feature_1 = values_feature_1[['feature_1']].to_numpy()

    for i, feature_1_run in enumerate(values_feature_1):
        samples_feature_1_run = x_train[x_train.feature_1 == feature_1_run[0]]
        values_feature_2 = samples_feature_1_run.drop_duplicates(subset='feature_2')
        values_feature_2 = values_feature_2[['feature_2']].to_numpy()

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

    # train NearestNeighbors with all combinations of the training data
    combinations.tolist()
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(combinations)

    # find the two closest settings of the test run
    sample = [[feature_1_test[0], feature_2_test[0], feature_3_test[0]]]
    nbrs = neigh.kneighbors(sample, n_neighbors=2, return_distance=False) #, return_distance=False)
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


# function to plot the fitting to the training data ###################################################################
def plot_train(model, x_samples, y_samples, x_scaler, y_scaler, number_net, features):
    """Plot a training run to see how good model is already fitted to training data

    :param model:                           pytorch MLP model
    :param x_samples:   - pd dataframe -    data of the training samples
    :param y_samples:   - pd dataframe -    target of the training samples
    :param x_scaler:                        MinMaxScaler of the data
    :param y_scaler:                        MinMaxScaler of the targets
    :param number_net:  - int -             Number to identify the MLP
    :param features:    - list of str -     list of features which has been used for training
    """

    # normalize the training data to make it suitable as input for MLP
    x_samples_normalized, _ = normalize_df(x_samples, x_scaler)

    # calculate output of MLP
    x_samples_tensor = torch.tensor(x_samples_normalized.values).float()
    model.eval()
    output = model(x_samples_tensor)

    # denormalize output to plot data
    output = output.detach().numpy()
    output = y_scaler.inverse_transform(output)

    # plot the MLP and reactor output for the training run
    plt.plot(x_samples[['PV']], y_samples[['T']], 'b-', label='Reactor output')
    plt.plot(x_samples[['PV']], output, 'r-', label='NN Output')

    plt.title('PODE{} {}={:.1f} {}={}bar {}={:.0f}K'.format(x_samples.iloc[0, 0],
                                                             features[1],
                                                             x_samples.iloc[0, 1],
                                                             features[2],
                                                             x_samples.iloc[0, 2] / ct.one_atm,
                                                             features[3],
                                                             x_samples.iloc[0, 3]))

    plt.legend()
    plt.xlabel('PV')
    plt.ylabel('T [K]')

    path = Path(__file__).resolve()
    path_plt = path.parents[2] / 'data/00001-MLP-temperature/{}_plt_comp_PODE{}_{}_{}_{}.pdf'.\
        format(number_net, x_samples.iloc[0, 0], x_samples.iloc[0, 1], x_samples.iloc[0, 2] / ct.one_atm,
               x_samples.iloc[0, 3])
    plt.savefig(path_plt)

    plt.show()
