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
def load_samples(mechanism_input, nbr_run, feature_select, features, labels, select_data, category):
    """
    Function to load the samples generated by the homogeneous reactor, select certain parts of the samples and split
    them in input and target data

    :parameter
    :param mechanism_input:     - str -             Name of mechanism used in the homogeneous reactor
    :param nbr_run:             - int -             Number to identify the run of the reactor data
    :param feature_select:      - dict -            features values which should be included or excluded in training
    :param features:            - list of str -     features
    :param labels:              - list of str -     labels
    :param select_data:         - str -             if 'exclude' feature_select data is excluded, if 'include' only
                                                    feature_select data is included
    :param category:            - str -             category of the samples (train, test, exp)

    :returns:
    :return x_samples:          - pd dataframe -    samples
    :return y_samples:          - pd dataframe -    targets
    """

    # create Path to samples and load data as dataframe
    mechanism = chose_mechanism(mechanism_input)
    path = Path(__file__).parents[3] / '000-homogeneous_reactor/data/00002-reactor-OME/{}/{}_{}_samples.csv'.\
        format(mechanism, nbr_run, category)
#    path = '/media/pascal/TOSHIBA EXT/BA/{}_{}_samples.csv'.format(nbr_run, category)
    data = pd.read_csv(path)

    # exclude or only include certain data
    data = select_samples(data, feature_select, select_data)

    # split the data in samples and targets
    x_samples = data[features]
    y_samples = data[labels]

    return x_samples, y_samples


# function to normalize the data ######################################################################################
def normalize_df(df, scaler):
    """
    Function to normalize dataframes

    :parameter:
    :param df:      - pd dataframe -    Dataframe which should be normalized
    :param scaler:                      MinMaxScaler of the dataframe

    :returns:
    :return df:     - pd dataframe -    normalized dataframe
    :return scaler:                     MinMaxScaler of the dataframe
    """

    columns = df.columns
    x = df.values  # returns a numpy array
    if scaler is None:
        min_max_scaler = preprocessing.MinMaxScaler()
        scaler = min_max_scaler.fit(x)
    x_scaled = scaler.transform(x)
    df = pd.DataFrame(x_scaled)
    df.columns = columns
    return df, scaler


# function to denormalize the data ####################################################################################
def denormalize_df(df, scaler):
    """
    Function to denormalize dataframes; inverse function to normalize dataframe

    :parameter:
    :param df:      - pd dataframe -    Dataframe which should be denormalized
    :param scaler:                      MinMaxScaler of the dataframe

    :returns:
    :return df:     - pd dataframe -    normalized dataframe
    """

    columns = df.columns
    x = df.values  # returns a numpy array
    x = scaler.inverse_transform(x)
    df = pd.DataFrame(x)
    df.columns = columns
    return df


# function to include certain data if demanded ########################################################################
def select_samples(df, feature_select, select_data):
    """
    Function to select data with certain feature values and then exclude it from the data or reduce the data to only the
    corresponding samples

    :parameter
    :param df:              - pd dataframe -    Dataframe of samples, targets
    :param feature_select:  - dict -            features values which should be included or excluded in training
    :param select_data:     - bool -            if 'exclude' feature_select data is excluded, if 'include' only
                                                feature_select data is included

    :returns:
    :return df:             - pd dataframe -    reduced dataframe
    """

    for feature, value in feature_select.items():
        if feature == 'P_0':
            value = np.array(value) * ct.one_atm

        if value is not None and select_data == 'include':

            # rename the column in df to feature in order to call it later
            df = df.rename(columns={'{}'.format(feature): 'feature'})

            for i, value_run in enumerate(value):
                df = df[df.feature == value_run]

            # rename the column back to its original name
            df = df.rename(columns={'feature': '{}'.format(feature)})

        elif value is not None and select_data == 'exclude':

            # rename the column in df to feature in order to call it later
            df = df.rename(columns={'{}'.format(feature): 'feature'})

            for i, value_run in enumerate(value):
                df = df[df.feature != value_run]

            # rename the column back to its original name
            df = df.rename(columns={'feature': '{}'.format(feature)})

    return df


# function to assign official mechanism name ##########################################################################
def chose_mechanism(mechanism_input):
    """
    Function to chose the official name of the entered mechanism

    :parameter
    :param mechanism_input:     - str -     Entered mechanism name

    :returns:
    :return mechanism:          - str -     Official mechanism name
    """

    if mechanism_input == 'he':
        mechanism = 'he_2018.xml'
    elif mechanism_input == 'cai':
        mechanism = 'cai_ome14_2019.xml'
    elif mechanism_input == 'sun':
        mechanism = 'sun_2017.xml'

    return mechanism


# function called by MLP script #######################################################################################
def load_dataloader(x_samples, y_samples, split, x_scaler, y_scaler, features):
    """
    Function to normalize, split into training and validation and combine samples and targets in dataloader

    :parameter:
    :param x_samples:       - pd dataframe -    samples
    :param y_samples:       - pd dataframe -    targets
    :param split:           - bool -            if true, samples will be split in train and validation
    :param x_scaler:                            MinMaxScaler of samples
    :param y_scaler:                            MinMaxScaler of targets
    :param features:        - list of str -     features

    :returns:
    :return loader:         - dataloader -      pytorch dataloader, combination of training samples and targets
    :return valid_loader:   - dataloader -      pytorch dataloader, combination of training samples and targets
    :return x_scaler:                           MinMaxScaler of samples
    :return y_scaler:                           MinMaxScaler of targets
    """

    # Normalize samples and targets
    if x_scaler is None:
        x_samples, x_scaler = normalize_df(x_samples, scaler=None)
    else:
        x_samples, _ = normalize_df(x_samples, scaler=x_scaler)

    if y_scaler is None:
        y_samples, y_scaler = normalize_df(y_samples, scaler=None)
    else:
        y_samples, _ = normalize_df(y_samples, y_scaler)

    if split:  # split into training and validation

        if len(x_samples.drop_duplicates([features[1]])) < 10:
            x_samples, x_validation, y_samples, y_validation = train_test_split(x_samples, y_samples, test_size=0.15)
        else:
            x_samples, x_validation, y_samples, y_validation = train_valid_split_self(x_samples, y_samples, features)

        x_validation = torch.tensor(x_validation.values).float()
        y_validation = torch.tensor(y_validation.values).float()
        tensor_validation = data.TensorDataset(x_validation, y_validation)

    # transform to torch tensor
    x_samples = torch.tensor(x_samples.values).float()
    y_samples = torch.tensor(y_samples.values).float()
    tensor = data.TensorDataset(x_samples, y_samples)

    # prepare data loaders
    batch_size = int(len(tensor) / 1000)
    num_workers = 8

    loader = torch.utils.data.DataLoader(tensor, batch_size=batch_size, num_workers=num_workers)

    if split:
        valid_loader = torch.utils.data.DataLoader(tensor_validation, batch_size=batch_size, num_workers=num_workers)
        return loader, valid_loader, x_scaler, y_scaler
    else:
        return loader


# function to split samples in train and validation set ###############################################################
def train_valid_split_self(x_samples, y_samples, features):
    """Function to split data in train and validation data

    :parameter
    :param x_samples:   - pd dataframe -    samples
    :param y_samples:   - pd dataframe -    targets
    :param features:    - list of str -     features

    :returns:
    :return x_train:    - pd dataframe -    training samples
    :return x_valid:    - pd dataframe -    validation samples
    :return y_train:    - pd dataframe -    training targets
    :return y_valid:    - pd dataframe -    validation targets
    """

    # rename feature columns, so that they can be called later
    x_samples = x_samples.rename(columns={'{}'.format(features[1]): 'feature_1',
                                          '{}'.format(features[2]): 'feature_2'})

    # find the values for the single features and select the ones for the validation
    samples_feature_1 = x_samples.drop_duplicates(['feature_1'])
    samples_feature_2 = x_samples.drop_duplicates(['feature_2'])

    samples_feature_1 = samples_feature_1[['feature_1']].to_numpy()
    samples_feature_2 = samples_feature_2[['feature_2']].to_numpy()

    index_samples_feature_1 = np.random.randint(0, len(samples_feature_1), size=int(len(samples_feature_1)*0.2))
    index_samples_feature_2 = np.random.randint(0, len(samples_feature_2), size=2)

    # if feature_set 1 or 2 selected in MLP, Pressure is used as additional feature
    if len(features) == 5:
        x_samples = x_samples.rename(columns={'{}'.format(features[3]): 'feature_3'})
        x_samples[['feature_3']] = x_samples[['feature_3']].round(decimals=5)
        samples_feature_3 = x_samples.drop_duplicates(['feature_3'])
        samples_feature_3 = samples_feature_3[['feature_3']].to_numpy()
        index_samples_feature_3 = np.random.randint(0, len(samples_feature_3), size=2)

    # separate samples into training and validation
    for i in range(2):
        samples_feature_1_run = samples_feature_1[index_samples_feature_1[i]]
        samples_feature_2_run = samples_feature_2[index_samples_feature_2[i]]

        if len(features) == 5:
            samples_feature_3_run = samples_feature_3[index_samples_feature_3[i]]

        if i == 0:
            x_train = x_samples[x_samples.feature_1 != samples_feature_1_run[0]]
            x_valid = x_samples[x_samples.feature_1 == samples_feature_1_run[0]]
        else:
            x_valid = x_valid.append(x_train[x_train.feature_1 == samples_feature_1_run[0]])
            x_train = x_train[x_train.feature_1 != samples_feature_1_run[0]]

        x_valid = x_valid.append(x_train[x_train.feature_2 == samples_feature_2_run[0]])
        x_train = x_train[x_train.feature_2 != samples_feature_2_run[0]]

        if len(features) == 5:
            x_valid = x_valid.append(x_train[x_train.feature_3 == samples_feature_3_run[0]])
            x_train = x_train[x_train.feature_3 != samples_feature_3_run[0]]

    indexes = x_train.index
    y_train = y_samples.iloc[indexes, :]

    indexes = x_valid.index
    y_valid = y_samples.iloc[indexes, :]

    return x_train, x_valid, y_train, y_valid
