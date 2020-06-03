#######################################################################################################################
# Data pre-processing before entering the MLP
#######################################################################################################################

# import packages #####################################################################################################
import numpy as np
import pandas as pd
import cantera as ct
import torch
from torch.utils import data
from pathlib import Path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# help functions ######################################################################################################
def loaddata_samples(mechanism_input, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature, pode, category,
                     s_paras, l_paras):
    mechanism = chose_mechanism(mechanism_input)
    path = Path(__file__).parents[3] / '000-homogeneous_reactor/data/00002-reactor-OME/{}/{}_{}_samples.csv'.\
        format(mechanism, nbr_run, category)
#    path = '/media/pascal/TOSHIBA EXT/BA/{}_{}_samples.csv'.format(nbr_run, category)
    data = pd.read_csv(path)

    data = exclude(data, equivalence_ratio, reactorPressure, reactorTemperature, pode)

    samples = data[s_paras]
    labels = data[l_paras]

    return samples, labels


def loaddata_delays(mechanism_input, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature, pode, category):
    mechanism = chose_mechanism(mechanism_input)
    path = Path(__file__).parents[3] / '000-homogeneous_reactor/data/00002-reactor-OME/{}/{}_{}_delays.csv'.\
        format(mechanism, nbr_run, category)
    data = pd.read_csv(path)
    data.set_index(['pode', 'phi', 'P_0'])

    # Exclude data if demanded
    data = exclude(data, equivalence_ratio, reactorPressure, reactorTemperature, pode)
    
    data = normalize_df(data)

    return data


# function to normalize the data ######################################################################################
def normalize_df(df, scaler):
    columns = df.columns
    x = df.values  # returns a numpy array
    if scaler is None:
        min_max_scaler = preprocessing.MinMaxScaler()
        scaler = min_max_scaler.fit(x)
    x_scaled = scaler.transform(x)
    df = pd.DataFrame(x_scaled)
    df.columns = columns
    return df, scaler


# function to exclude certain data if demanded ########################################################################
def exclude(df, equivalence_ratio, reactorPressure, reactorTemperature, pode):
    # Exclude data if demanded
    for i, pode_run in enumerate(pode):
        df = df[df.pode != pode_run]
    for i, equivalence_ratio_run in enumerate(equivalence_ratio):
        df = df[df.phi != equivalence_ratio_run]
    for i, reactorPressure_run in enumerate(reactorPressure):
        df = df[df.P_0 != reactorPressure_run * ct.one_atm]
    for i, reactorTemperature_run in enumerate(reactorTemperature):
        df = df[df.T_0 != reactorTemperature_run]

    return df


# function to assign official mechanism name ##########################################################################
def chose_mechanism(mechanism_input):
    if mechanism_input == 'he':
        mechanism = 'he_2018.xml'
    elif mechanism_input == 'cai':
        mechanism = 'cai_ome14_2019.xml'
    elif mechanism_input == 'sun':
        mechanism = 'sun_2017.xml'

    return mechanism


# function called by MLP script #######################################################################################
def loaddata(mechanism_input, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature, pode, s_paras, l_paras,
             category, scaler_samples, scaler_labels):

    samples, labels = loaddata_samples(mechanism_input, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature, pode, category, s_paras, l_paras)

    if scaler_samples is None:
        samples, scaler_samples = normalize_df(samples, scaler=None)
    else:
        samples, _ = normalize_df(samples, scaler=scaler_samples)

    if scaler_labels is None:
        labels, scaler_labels = normalize_df(labels, scaler=None)
    else:
        labels, _ = normalize_df(labels, scaler_labels)

    if category == 'train':
#        samples, x_validation, labels, y_validation = train_test_split(samples, labels, test_size=0.15)
        samples, x_validation, labels, y_validation = train_valid_split_self(samples, labels)
        x_validation = torch.tensor(x_validation.values).float()
        y_validation = torch.tensor(y_validation.values).float()
        tensor_validation = data.TensorDataset(x_validation, y_validation)

    # transform to torch tensor
    samples = torch.tensor(samples.values).float()
    labels = torch.tensor(labels.values).float()
    tensor = data.TensorDataset(samples, labels)

    # prepare data loaders
    batch_size = int(len(tensor) / 1000)
    num_workers = 8

    loader = torch.utils.data.DataLoader(tensor, batch_size=batch_size, num_workers=num_workers)
    if category == 'train':
        valid_loader = torch.utils.data.DataLoader(tensor_validation, batch_size=batch_size, num_workers=num_workers)
        return loader, valid_loader, scaler_samples, scaler_labels
    else:
        return loader


# function to split samples in train and validation set ###############################################################
def train_valid_split_self(samples, labels):
    phis = samples.drop_duplicates(['phi'])
    temps = samples.drop_duplicates(['T_0'])
    pressures = samples.drop_duplicates(['P_0'])

    phis = phis[['phi']].to_numpy()
    temps = temps[['T_0']].to_numpy()
    pressures = pressures[['P_0']].to_numpy()

    index_phi = np.random.randint(0, len(phis), size=2)
    index_temp = np.random.randint(0, len(temps), size=2)
    index_pressure = np.random.randint(0, len(pressures), size=2)

    for i in range(2):
        phi_run = phis[index_phi[i]]
        pressure_run = pressures[index_pressure[i]]
        temp_run = temps[index_temp[i]]

        if i == 0:
            samples_train = samples[samples.phi != phi_run[0]]
            samples_valid = samples[samples.phi == phi_run[0]]
        else:
            samples_valid = samples_valid.append(samples_train[samples_train.phi == phi_run[0]])
            samples_train = samples_train[samples_train.phi != phi_run[0]]

        samples_valid = samples_valid.append(samples_train[samples_train.P_0 == pressure_run[0]])
        samples_train = samples_train[samples_train.P_0 != pressure_run[0]]
        samples_valid = samples_valid.append(samples_train[samples_train.T_0 == temp_run[0]])
        samples_train = samples_train[samples_train.T_0 != temp_run[0]]

    indexes = samples_train.index
    labels_train = labels.iloc[indexes, :]

    indexes = samples_valid.index
    labels_valid = labels.iloc[indexes, :]

    return samples_train, samples_valid, labels_train, labels_valid

