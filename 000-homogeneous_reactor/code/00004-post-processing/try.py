# Import packages
import matplotlib.pyplot as plt
import numpy as np
import cantera as ct
from pathlib import Path
import pandas as pd


#######################################################################################################################
def loaddata_samples(mechanism, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature, scale, pode):
    path = Path(__file__).parents[2] / 'data/00002-reactor-OME/{}/{}_train_data.csv'.format(mechanism[0], nbr_run)
    data = pd.read_csv(path)

    # Select only the data needed for the plot
    data = data[data.pode == pode]
    data = data[data.phi == equivalence_ratio]
    data = data[data.P_0 == reactorPressure * ct.one_atm]
    data = data[data.T_0 == reactorTemperature]

    # Normalize/ fit certain data
    data[['P']] = data[['P']] / ct.one_atm
    data[['PV']] = data[['PV']] / np.amax(data[['PV']])
    data[['time']] = data[['time']] * 1.e+3

    # define the name of the x_axis
    if scale == 'PV':
        scale_name = 'PV (normalized)'
    elif scale == 'time':
        scale_name = 'time in ms'
    else:
        print('wrong scale parameter')

    return data, scale_name


def create_text(mechanism, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature, pode):
    path = Path(__file__).parents[2] / 'data/00002-reactor-OME/{}/{}_delays.csv'.format(mechanism[0], nbr_run)
    data = pd.read_csv(path)

    # Select only the data needed for the plot
    data = data[data.pode == pode]
    data = data[data.phi == equivalence_ratio]
    data = data[data.P_0 == reactorPressure * ct.one_atm]

    if data.empty:
        print('WARNING:Entered parameter setting not capturad in data!')

    result = np.where(data == reactorTemperature)
    first = data.iloc[result[0], 4]
    main = data.iloc[result[0], 5]
    textstr = 'first={:.4f}ms \nmain={:.4f}ms'.format(first.iloc[0], main.iloc[0])
    return textstr


#######################################################################################################################
mechanism = np.array(['cai_ome14_2019.xml'])
equivalence_ratio = 1.0
reactorPressure = 20
reactorTemperature = 660
scale = 'time'
pode = 3
nbr_run = '001'

a = True

if a is True:
    # get data
    samples, scale_name = loaddata_samples(mechanism, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature,
                                           scale, pode)

    samples.plot(scale, 'Q', style='b-')
    plt.xlabel(scale_name)
    plt.ylabel('Heat release [W/m$^3$]')

    textstr = create_text(mechanism, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature, pode)
    plt.text(0.6 * np.amax(samples[[scale]]), 0.4 * np.amax(samples['Q']), textstr, fontsize=12)

    plt.title('{} PODE{} $\\Phi$={:.1f} p={}bar $T_0$={:.0f}K'.format(mechanism[0], pode, equivalence_ratio,
                                                                      reactorPressure, reactorTemperature))

    path = Path(__file__).parents[2] / 'data/00004-post-processing/{}/{}/plot_HR_PODE{}_{}_{:.0f}_{}_{}.pdf'. \
        format(mechanism[0], nbr_run, pode, equivalence_ratio, reactorPressure, reactorTemperature, scale)
    plt.savefig(path)

    plt.show()
