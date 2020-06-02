#######################################################################################################################
# build, train and validate the MLP
######################################################################################################################

# %% import packages
import argparse
import torch
import fc_model
import numpy as np
import cantera as ct
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path
from sklearn import preprocessing


# load trained model ##################################################################################################
def load_checkpoint(nbr_net, typ):
    path = Path(__file__).resolve()
    path_pth = path.parents[2] / 'data/00001-MLP-temperature/{}_{}_checkpoint.pth'.format(nbr_net, typ)
    checkpoint = torch.load(path_pth)

    # Create model and load its criterion
    model = fc_model.Net(checkpoint['input_size'],
                         checkpoint['output_size'],
                         checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    criterion = nn.MSELoss()
    criterion.load_state_dict(checkpoint['criterion'])

    # load the parameters of the samples and labels, as well as the corresponding MinMaxScalers
    number_train_run = checkpoint['number_train_run']
    sample_paras = checkpoint['s_paras']
    labels_paras = checkpoint['l_paras']
    scaler_samples = checkpoint['scaler_samples']
    scaler_labels = checkpoint['scaler_labels']

    # parameters needed to save the model again
    n_input = checkpoint['input_size']
    n_output = checkpoint['output_size']

    # parameters needed to continue training the model
    valid_test_loss_min = checkpoint['valid_test_loss_min']
    valid_train_loss_min = checkpoint['valid_train_loss_min']

    return model, criterion, sample_paras, labels_paras, scaler_samples, scaler_labels, n_input, n_output, \
           valid_test_loss_min, valid_train_loss_min, number_train_run


# calculate mean accuracy over test data ###############################################################################
def calc_acc(model, criterion, test_loader, scaler):
    model.eval()  # prep model for evaluation
    acc = []
    test_loss = 0
    for data, target in test_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item() * data.size(0)
        # compare predictions to true label
        correct = np.zeros((len(output)))

        output = output.detach().numpy()
        target = target.detach().numpy()
        output = scaler.inverse_transform(output)
        target = scaler.inverse_transform(target)

        for i in range(len(output)):
            if target[i] * 0.95 < output[i] < target[i] * 1.05:
                correct[i] = 1
            else:
                correct[i] = 0

        acc = np.append(acc, np.sum(correct) / len(output))

    acc_mean = np.mean(acc)
    return acc_mean


# plot output of the reactor next to the output of the NN #############################################################
def plot_data(model, train_samples, train_labels, test_samples, test_labels, scaler_samples, scaler_labels, number_net,
              typ, plt_nbrs):

    phi = test_samples.drop_duplicates(subset=['phi'])
    phi = phi[['phi']].to_numpy()

    for i, phi_run in enumerate(phi):
        samples_phi = test_samples[test_samples.phi == phi_run[0]]
        pressure = samples_phi.drop_duplicates(['P_0'])
        pressure = pressure[['P_0']].to_numpy()

        for ii, pressure_run in enumerate(pressure):
            samples_pressure = samples_phi[samples_phi.P_0 == pressure_run[0]]
            temperature = samples_pressure.drop_duplicates(['T_0'])
            temperature = temperature[['T_0']].to_numpy()

            for iii, temperature_run in enumerate(temperature):
                samples_temperature = samples_pressure[samples_pressure.T_0 == temperature_run[0]]

                if plt_nbrs is True:
                    nbr1_sample, nbr1_label, nbr2_sample, nbr2_label = find_neighbors(phi_run, pressure_run, temperature_run, train_samples, train_labels)
                    nbr1_label = scaler_labels.inverse_transform(nbr1_label)
                    nbr2_label = scaler_labels.inverse_transform(nbr2_label)
                    plt.plot(nbr1_sample[['PV']], nbr1_label, 'g-', label='Nbr1')
                    plt.plot(nbr2_sample[['PV']], nbr2_label, 'y-', label='Nbr2')

                indexes = samples_temperature.index
                labels_run = test_labels.iloc[indexes, :]
                labels_run = labels_run.to_numpy()

                samples_tensor = torch.tensor(samples_temperature.values).float()
                model.eval()
                output = model(samples_tensor)

                output = output.detach().numpy()
                output = scaler_labels.inverse_transform(output)
                labels_run = scaler_labels.inverse_transform(labels_run)

                plt.plot(samples_temperature[['PV']], labels_run, 'b-', label='Reactor output')
                plt.plot(samples_temperature[['PV']], output, 'r-', label='NN Output')


                samples_temperature = samples_temperature.to_numpy()
                samples_temperature = scaler_samples.inverse_transform(samples_temperature)

                plt.title('PODE{} $\\Phi$={:.2f} p={}bar $T_0$={:.0f}K'.format(samples_temperature[0, 0],
                                                                               samples_temperature[0, 1],
                                                                               samples_temperature[0, 2] / ct.one_atm,
                                                                               samples_temperature[0, 3]))

                plt.legend()
                plt.xlabel('PV')
                plt.ylabel('T [K]')

                path = Path(__file__).resolve()
                path_plt = path.parents[2] / 'data/00001-MLP-temperature/{}_{}_plt_comp_PODE{}_{}_{}_{}.pdf'.format \
                    (number_net, typ, samples_temperature[0, 0], samples_temperature[0, 1],
                     samples_temperature[0, 2] / ct.one_atm, samples_temperature[0, 3])
                plt.savefig(path_plt)

                plt.show()


# function to denormalize the data ####################################################################################
def denormalize_df(df, scaler):
    columns = df.columns
    x = df.values  # returns a numpy array
    x = scaler.inverse_transform(x)
    df = pd.DataFrame(x)
    df.columns = columns
    return df


# find closest runs between test and train samples ####################################################################
def find_neighbors(sample_phi, sample_p, sample_temp, train_samples, train_labels):
    from sklearn.neighbors import NearestNeighbors

    phi = train_samples.drop_duplicates(subset=['phi'])
    phi = phi[['phi']].to_numpy()

    for i, phi_run in enumerate(phi):
        samples_phi = train_samples[train_samples.phi == phi_run[0]]
        pressure = samples_phi.drop_duplicates(subset=['P_0'])
        pressure = pressure[['P_0']].to_numpy()

        for ii, pressure_run in enumerate(pressure):
            samples_pressure = samples_phi[samples_phi.P_0 == pressure_run[0]]
            temperature = samples_pressure.drop_duplicates(subset=['T_0'])
            temperature = temperature[['T_0']].to_numpy()

            combination = np.concatenate((phi_run*np.ones((len(temperature), 1)), pressure_run*np.ones((len(temperature), 1)), temperature), axis=1)
            if ii == 0 and i == 0:
                combinations = combination
            else:
                combinations = np.append(combinations, combination, axis=0)

    combinations.tolist()
    sample = [[sample_phi[0], sample_p[0], sample_temp[0]]]
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(combinations)
    nbrs = neigh.kneighbors(sample, n_neighbors=2, return_distance=False) #, return_distance=False)
    print(nbrs)

    nbr1_paras = combinations[nbrs[0, 0]]
    nbr1_sample = train_samples[train_samples.phi == nbr1_paras[0]]
    nbr1_sample = nbr1_sample[nbr1_sample.P_0 == nbr1_paras[1]]
    nbr1_sample = nbr1_sample[nbr1_sample.T_0 == nbr1_paras[2]]
    indexes = nbr1_sample.index
    nbr1_label = train_labels.iloc[indexes, :]

    nbr2_paras = combinations[nbrs[0, 1]]
    nbr2_sample = train_samples[train_samples.phi == nbr2_paras[0]]
    nbr2_sample = nbr2_sample[nbr2_sample.P_0 == nbr2_paras[1]]
    nbr2_sample = nbr2_sample[nbr2_sample.T_0 == nbr2_paras[2]]
    indexes = nbr2_sample.index
    nbr2_label = train_labels.iloc[indexes, :]

    return nbr1_sample, nbr1_label, nbr2_sample, nbr2_label


# function to plot the fitting to the training data ###################################################################
def plot_train(model, samples, labels, scaler_samples, scaler_labels, number_net, pode, equivalence_ratio,
               reactorPressure, reactorTemperature, typ):
    samples_denormalized = denormalize_df(samples, scaler_samples)

    samples_denormalized = samples_denormalized[samples_denormalized.pode == pode[0]]
    samples_denormalized = samples_denormalized[samples_denormalized.phi == equivalence_ratio[0]]
    samples_denormalized = samples_denormalized[samples_denormalized.P_0 == reactorPressure[0] * ct.one_atm]
    samples_denormalized = samples_denormalized[samples_denormalized.T_0 == reactorTemperature[0]]

    indexes = samples_denormalized.index
    labels = labels.iloc[indexes, :]
    samples = samples.iloc[indexes, :]

    labels = denormalize_df(labels, scaler_labels)

    samples_tensor = torch.tensor(samples.values).float()
    model.eval()
    output = model(samples_tensor)

    output = output.detach().numpy()
    output = scaler_labels.inverse_transform(output)

    plt.plot(samples[['PV']], labels[['T']], 'b-', label='Reactor output')
    plt.plot(samples[['PV']], output, 'r-', label='NN Output')

    plt.title('PODE{} $\\Phi$={:.1f} p={}bar $T_0$={:.0f}K'.format(samples_denormalized.iloc[0, 0],
                                                                   samples_denormalized.iloc[0, 1],
                                                                   samples_denormalized.iloc[0, 2] / ct.one_atm,
                                                                   samples_denormalized.iloc[0, 3]))

    plt.legend()
    plt.xlabel('PV')
    plt.ylabel('T [K]')

    path = Path(__file__).resolve()
    path_plt = path.parents[2] / 'data/00001-MLP-temperature/{}_{}_plt_comp_PODE{}_{}_{}_{}.pdf'.format \
        (number_net, typ, samples_denormalized.iloc[0, 0], samples_denormalized.iloc[0, 1],
         samples_denormalized.iloc[0, 2] / ct.one_atm, samples_denormalized.iloc[0, 3])
    plt.savefig(path_plt)

    plt.show()
