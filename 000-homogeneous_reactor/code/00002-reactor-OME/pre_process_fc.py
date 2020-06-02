#######################################################################################################################
# Pre-processing steps to save data, delays and initials
#######################################################################################################################
from pathlib import Path
import os
import pandas as pd
import numpy as np


def chose_mechanism(mechanism_input):
    if mechanism_input == 'he':
        mechanism = 'he_2018.xml'
    elif mechanism_input == 'cai':
        mechanism = 'cai_ome14_2019.xml'
    elif mechanism_input == 'sun':
        mechanism = 'sun_2017.xml'

    return mechanism


def create_path(mechanism_input, nbr_run):
    mechanism = chose_mechanism(mechanism_input)
    path = Path(__file__).resolve()
    path_dir = path.parents[2] / 'data/00002-reactor-OME/{}'.format(mechanism)
    path_plt = path.parents[2] / 'data/00004-post-processing/{}/{}/'.format(mechanism, nbr_run)
    return path_dir, path_plt


def make_dir(mechanism_input, nbr_run, inf_print):
    path_dir, path_plt = create_path(mechanism_input, nbr_run)
    try:
        os.makedirs(path_dir)
        os.makedirs(path_plt)
    except OSError:
        if inf_print is True:
            print('Directories already exist')
    else:
        if inf_print is True:
            print('Necessary directories created')


def save_df(typ, category, mechanism_input, nbr_run, reactorTemperature_end, reactorTemperature_start,
            reactorTemperature_step, phi_end, phi_0, phi_step, p_end, p_0, p_step, pode, size):

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