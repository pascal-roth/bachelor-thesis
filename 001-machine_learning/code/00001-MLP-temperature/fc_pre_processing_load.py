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
def loaddata_samples(mechanism_input, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature, pode, category):
    mechanism = chose_mechanism(mechanism_input)
    path = Path(__file__).parents[3] / '000-homogeneous_reactor/data/00002-reactor-OME/{}/{}_{}_samples.csv'.\
        format(mechanism, nbr_run, category)
    data = pd.read_csv(path)

    data = exclude(data, equivalence_ratio, reactorPressure, reactorTemperature, pode)

    samples = data[['pode', 'phi', 'P_0', 'T_0', 'PV']]
    labels = data[['T']]

    samples, scaler_samples = normalize_df(samples)
    labels, scaler_labels = normalize_df(labels)
    return samples, labels, scaler_samples, scaler_labels


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
def normalize_df(df):
    columns = df.columns
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    scaler = min_max_scaler.fit(x)
    x_scaled = min_max_scaler.transform(x)
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
def loaddata(mechanism_input, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature, pode, category, val_split):
    samples, labels, _, scaler_labels = loaddata_samples(mechanism_input, nbr_run, equivalence_ratio, reactorPressure,
                                                         reactorTemperature, pode, category)

    if val_split is True:
        samples, x_validation, labels, y_validation = train_test_split(samples, labels, test_size=0.2)
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
    if val_split is True:
        valid_loader = torch.utils.data.DataLoader(tensor_validation, batch_size=batch_size, num_workers=num_workers)
        return loader, valid_loader
    else:
        return loader, scaler_labels


