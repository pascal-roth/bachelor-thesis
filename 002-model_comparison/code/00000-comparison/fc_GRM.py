#######################################################################################################################
# Functions to load the data of the Global Reaction Mechanims (GRM)
#######################################################################################################################

# import packages #####################################################################################################
import pandas as pd
from pathlib import Path
import cantera as ct
import numpy as np


# function to load output of GRM ######################################################################################
def load_GRM(pode, phi, pressure, temperature):
    # Path to samples
    path = Path(__file__).parents[2] / 'data/00000-comparison/data-global-mechanism/OME{}_phi{}_p{}.0_T{}.0.txt'. \
        format(pode, phi, pressure, temperature)
    df = pd.read_csv(path, sep=" ", header=None)

    # rename columns
    df.columns = ['time', 'P', 'T', 'HRR', 'PODE', 'O2', 'H2O', 'CO', 'CO2', 'I1', 'Y', 'I2', 'N2']

    # calculate the PV as used in the HR and MLP
    PV = np.zeros((len(df), 1))
    for i in range(len(df)):
        PV[i] = df['H2O'].iloc[i] * 0.5 + df['CO2'].iloc[i] * 0.25

    # add PV to df
    PV = pd.DataFrame(PV)
    PV.columns = ['PV']
    df = pd.concat([PV, df], axis=1)

    return df


# calculate IDTs ######################################################################################################
def load_IDTs(pode, phi, pressure, temperature):
    # load data of the GRM
    df = load_GRM(pode, phi, pressure, temperature)
    # calculate IDT
    IDT_location = np.argmax(df['HRR'])
    IDT_PV = df['PV'].iloc[IDT_location]

    return IDT_PV


# get GRM data and add all information to df ##########################################################################
def load_GRM_data(pode, phi, pressure, temperature):
    # load data of the GRM
    df = load_GRM(pode, phi, pressure, temperature)

    # insert additional information in df
    df.insert(0, 'T_0', temperature)
    df.insert(0, 'P_0', pressure)
    df.insert(0, 'phi', phi)
    df.insert(0, 'pode', pode)

    return df