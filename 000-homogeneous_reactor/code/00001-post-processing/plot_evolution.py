#######################################################################################################################
# This file includes the function to plot
#   - thermodynamic properities (temperature, pressure, volume)
#   - species development
#   - progress variable
#   - Heat release
#######################################################################################################################

# Import packages
import numpy as np
import cantera as ct
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('stfs')


#######################################################################################################################
def loaddata_samples(mechanism, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature, pode, category):
    path = Path(__file__).parents[2] / 'data/00000-reactor-OME/{}/{}_{}_samples.csv'.format(mechanism, nbr_run,
                                                                                            category)
    #    path = '/media/pascal/TOSHIBA EXT/BA/{}_{}_samples.csv'.format(nbr_run, category)
    data = pd.read_csv(path)

    # Select only the data needed for the plot
    data = data[data.pode == pode]
    data_phi = data.phi.round(2)
    data.phi = data_phi
    data = data[data.phi == equivalence_ratio]
    data = data[data.P_0 == reactorPressure * ct.one_atm]
    data = data[data.T_0 == reactorTemperature]

    if data.empty:
        print('WARNING:Entered parameter setting not capturad in data!')

    # Normalize/ fit certain data
    data[['P']] = data[['P']] / ct.one_atm
    data[['time']] = data[['time']] * 1.e+3

    return data


samples = loaddata_samples(mechanism='cai_ome14_2019.xml', nbr_run='000', equivalence_ratio=1.0, reactorPressure=20,
                           reactorTemperature=950, pode=3, category='train')

time_min = np.round(np.min(samples[['time']]), decimals=3)
time_max = np.round(np.max(samples[['time']]), decimals=3)

PV_min = np.min(samples[['PV']])
PV_max = np.max(samples[['PV']])

temp_min = np.min(samples[['T']])
temp_max = np.max(samples[['T']])

samples[['time']] = np.round(samples[['time']], decimals=3)

index = 0

for i in range(100):
    index += int(len(samples)/100)

    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 6))
    fig, (ax1, ax3) = plt.subplots(ncols=1, nrows=2, figsize=(6, 6))
    ax1.set_ylabel('Y')
    ax1.plot(samples[['time']].iloc[:index], samples[['PODE']].iloc[:index], label='Fuel', color='m')
    ax1.plot(samples[['time']].iloc[:index], samples[['O2']].iloc[:index], label='O2', color='g')
    ax1.plot(samples[['time']].iloc[:index], samples[['H2O']].iloc[:index], label='H2O', color='b')
    ax1.plot(samples[['time']].iloc[:index], samples[['CO2']].iloc[:index], label='CO2', color='y')

    # ax2.plot(samples[['PV']].iloc[:index], samples[['PODE']].iloc[:index])
    # ax2.plot(samples[['PV']].iloc[:index], samples[['O2']].iloc[:index])
    # ax2.plot(samples[['PV']].iloc[:index], samples[['H2O']].iloc[:index])
    # ax2.plot(samples[['PV']].iloc[:index], samples[['CO2']].iloc[:index])

    ax3.set_ylabel('T')
    ax3.set_xlabel('time')
    ax3.plot(samples[['time']].iloc[:index], samples[['T']].iloc[:index], label='temperature', color='r')

    # ax4.set_xlabel('PV')
    # ax4.plot(samples[['PV']].iloc[:index], samples[['T']].iloc[:index], color='r')

    ax1.set_xlim(time_min[0], time_max[0])
    # ax2.set_xlim(PV_min[0], PV_max[0])
    ax3.set_xlim(time_min[0], time_max[0])
    # ax4.set_xlim(PV_min[0], PV_max[0])

    ax3.set_ylim(temp_min[0], temp_max[0])
    # ax4.set_ylim(temp_min[0], temp_max[0])

    # ax1.legend()

    plt.tight_layout()

    time = samples[['time']].iloc[index]
    path = Path(__file__).parents[2] / 'data/00001-post-processing/evolution/plot_{}.png'.format(i)
    plt.savefig(path)

    plt.show()
