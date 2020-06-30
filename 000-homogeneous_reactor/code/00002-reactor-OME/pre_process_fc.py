#######################################################################################################################
# Pre-processing steps to save data, delays and initials
#######################################################################################################################
from pathlib import Path
import os
import pandas as pd
import numpy as np


#######################################################################################################################
def chose_mechanism(mechanism_input):
    """
    Function to chose the official name of the entered mechanism

    :parameter
    :param mechanism_input:         - str -     Entered mechanism name

    :returns:
    :return mechanism:              - str -     Official mechanism name
    """

    if mechanism_input == 'he':
        mechanism = 'he_2018.xml'
    elif mechanism_input == 'cai':
        mechanism = 'cai_ome14_2019.xml'
    elif mechanism_input == 'sun':
        mechanism = 'sun_2017.xml'

    return mechanism


#######################################################################################################################
def create_path(mechanism_input, nbr_run):
    """
    create path for direction to save samples and for directory to save corresponding plots

    :parameter:
    :param mechanism_input:         - str -         Entered mechanism name
    :param nbr_run:                 - str -         Number to identify run

    :returns:
    :return path_dir:                               Path to directories for samples
    :return path_plt:                               Path to directories for plots
    """

    mechanism = chose_mechanism(mechanism_input)
    path = Path(__file__).resolve()
    path_dir = path.parents[2] / 'data/00002-reactor-OME/{}'.format(mechanism)
    path_plt = path.parents[2] / 'data/00004-post-processing/{}/{}/'.format(mechanism, nbr_run)
    return path_dir, path_plt


#######################################################################################################################
def make_dir(mechanism_input, nbr_run, inf_print):
    """
    check if directories for samples and plots already exists, if not make them

    :parameter
    :param mechanism_input:         - str -         Entered mechanism name
    :param nbr_run:                 - str -         Number to identify run
    :param inf_print:               - bool -        if true, information will be printed
    """

    # create path
    path_dir, path_plt = create_path(mechanism_input, nbr_run)

    # make dir
    try:
        os.makedirs(path_dir)
        os.makedirs(path_plt)
    except OSError:
        if inf_print is True:
            print('Samples directory already exist')
    else:
        if inf_print is True:
            print('Samples directory created')

    try:
        os.makedirs(path_plt)
    except OSError:
        if inf_print is True:
            print('Plot directory already exist')
    else:
        if inf_print is True:
            print('Plot directory created')


#######################################################################################################################
def save_df(typ, category, mechanism_input, nbr_run, reactorTemperature_end, reactorTemperature_start,
            reactorTemperature_step, phi_end, phi_0, phi_step, p_end, p_0, p_step, pode, size):
    """
    create arrays to save samples and if category test, check if samples have to be connected with previous samples of
    the same run

    :parameter
    :param typ:                         - str -         samples or delays
    :param category:                    - str -         test, train or exp-comparison samples
    :param mechanism_input:             - str -         entered mechanism name
    :param nbr_run:                     - str -         number to identify run
    :param reactorTemperature_end:      - int -         final value of the initial temperature
    :param reactorTemperature_start:    - int -         start value of the initial temperature
    :param reactorTemperature_step:     - int -         step between different initial temperatures
    :param phi_end:                     - float -       final value of the equivalence ratio
    :param phi_0:                       - float -       start value of the equivalence ratio
    :param phi_step:                    - float -       step between different equivalence ratios
    :param p_end:                       - int -         final value of the initial pressure
    :param p_0:                         - int -         start value of the initial pressure
    :param p_step:                      - int -         step between different initial pressures
    :param pode:                        - array -       array of degree(s) of polymerization
    :param size:                        - int -         number of columns of the array

    :returns:
    :return data:                       - np array -    array with needed size
    :return n:                          - int -         if category==test and there have been samples of the same run,
                                                        n indicates the row number where the first additional sample
                                                        will be added
    """

    if typ == 'samples' and not category == 'test':
        n = 0
        data = np.zeros((((reactorTemperature_end + reactorTemperature_step - reactorTemperature_start) // reactorTemperature_step) *
                         int((phi_end + phi_step - phi_0) / phi_step) * ((p_end + p_step - p_0) // p_step) * len(pode) * 12500, size))

    elif typ != 'samples' and not category == 'test':
        data = np.zeros((((reactorTemperature_end + reactorTemperature_step - reactorTemperature_start) // reactorTemperature_step) *
                         int((phi_end + phi_step - phi_0) / phi_step) * ((p_end + p_step - p_0) // p_step) * len(pode), size))
        n = 0
    elif category == 'test':
        try:
            path_dir, _ = create_path(mechanism_input, nbr_run)
            path = '{}/{}_{}_{}.csv'.format(path_dir, nbr_run, category, typ)
            data = pd.read_csv(path)
            data = data.to_numpy()

            if typ != 'samples':
                data = data[:, 1:]
                n = len(data)
                data_new = np.zeros((((reactorTemperature_end + reactorTemperature_step - reactorTemperature_start) // reactorTemperature_step) *
                                    int((phi_end + phi_step - phi_0) / phi_step) * ((p_end + p_step - p_0) // p_step) * len(pode), size))
                data = np.append(data, data_new, axis=0)
            else:
                n = len(data)
                data_new = np.zeros((((reactorTemperature_end + reactorTemperature_step - reactorTemperature_start) // reactorTemperature_step) *
                                    int((phi_end + phi_step - phi_0) / phi_step) * ((p_end + p_step - p_0) // p_step) * len(pode) * 12500, size))
                data = np.append(data, data_new, axis=0)

        except FileNotFoundError:

            if typ != 'samples':
                data = np.zeros((((reactorTemperature_end + reactorTemperature_step - reactorTemperature_start) // reactorTemperature_step) *
                                int((phi_end + phi_step - phi_0) / phi_step) * ((p_end + p_step - p_0) // p_step) * len(pode), size))
            else:
                data = np.zeros((((reactorTemperature_end + reactorTemperature_step - reactorTemperature_start) // reactorTemperature_step) *
                                 int((phi_end + phi_step - phi_0) / phi_step) * ((p_end + p_step - p_0) // p_step) * len(pode) * 12500, size))
            n = 0
            print('No pre existing {}_{} array'.format(category, typ))

    return data, n