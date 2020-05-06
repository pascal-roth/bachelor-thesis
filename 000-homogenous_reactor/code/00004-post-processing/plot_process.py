#######################################################################################################################
# This file includes the function to plot
#   - thermodynamic properities (temperature, pressure, volume)
#   - species development
#   - progress variable
#   - Heat release
#######################################################################################################################

# Import packages
import matplotlib.pyplot as plt
import numpy as np
import cantera as ct
from pathlib import Path
import pandas as pd


#######################################################################################################################
def loaddata_samples(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, scale, pode, t_start, t_end, t_step):
    path = Path(__file__).parents[2] / 'data/00002-reactor-OME/{}_PODE{}_{}_{:.0f}_{}_{}_{}/samples_{}'.format(
                                  mechanism[0], pode, equivalence_ratio, reactorPressure, t_start, t_end,
                                  t_step, reactorTemperature)
    data = pd.read_csv(path)

    data[['PV']] = data[['PV']] / np.amax(data[['PV']])
    data[['time']] = data[['time']] * 1.e+3
    if scale == 'PV':
        scale_name = 'PV (normalized)'
    elif scale == 'time':
        scale_name = 'time in ms'
    else:
        print('wrong scale parameter')

    return data, scale_name


def loaddata_delays(mechanism, equivalence_ratio, reactorPressure, pode, t_start, t_end, t_step):
    path = Path(__file__).parents[2] / 'data/00002-reactor-OME/{}_PODE{}_{}_{:.0f}_{}_{}_{}/delays'.format(
        mechanism[0], pode, equivalence_ratio, reactorPressure, t_start, t_end, t_step)
    data = pd.read_csv(path)
    return data


#######################################################################################################################
def plot_thermo(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, scale, pode, t_start, t_end, t_step):

    # get data
    samples, scale_name = loaddata_samples(mechanism, equivalence_ratio, reactorPressure, reactorTemperature,
                                           scale, pode, t_start, t_end, t_step)

    fig = plt.figure()
    samples[['P']] = samples[['P']] / ct.one_atm
    ax1 = samples.plot(scale, 'T', style='b-', ax=fig.add_subplot(221))
    ax1.set_xlabel(scale_name)
    ax1.set_ylabel('Temperature (K)')

    ax2 = samples.plot(scale, 'P', style='b-', ax=fig.add_subplot(222))
    ax2.set_xlabel(scale_name)
    ax2.set_ylabel('Pressure (Bar)')

    ax3 = samples.plot(scale, 'V', style='b-', ax=fig.add_subplot(223))
    ax3.set_xlabel(scale_name)
    ax3.set_ylabel('Volume (m$^3$)')

    plt.tight_layout()
    plt.figlegend(['$\Phi$ = {} \np = {} \n$T_0$ = {}'. format(equivalence_ratio, reactorPressure,
                                                               reactorTemperature)], loc='lower right')

    path = Path(__file__).parents[2] / 'data/00004-post-processing/{}_PODE{}_{}_{:.0f}_{}_{}_{}/plot_thermo_{}_{}.pdf'\
        .format(mechanism[0], pode, equivalence_ratio, reactorPressure, t_start, t_end, t_step, reactorTemperature,
                scale)
    plt.savefig(path)

    plt.show()


#######################################################################################################################
def plot_species(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, scale, pode, t_start, t_end, t_step):
    # get data
    samples, scale_name = loaddata_samples(mechanism, equivalence_ratio, reactorPressure, reactorTemperature,
                                           scale, pode, t_start, t_end, t_step)

#    samples.plot(scale, ['PODE', 'CO2', 'O2', 'CO', 'H2O', 'OH', 'H2O2', 'CH3', 'CH3O', 'CH2O', 'C2H2'],
#                 style=['b-', 'r-', 'g-', 'y-', 'c-', 'm-', 'k-', 'y-', 'r-', 'g-', 'c-'],
#                 label=['$Y_{pode_n}$', '$Y_{CO2}$', '$Y_{O2}$', '$Y_{CO}$', '$Y_{H2O}$', '$Y_{OH}$', '$Y_{H2O2}$',
#                        '$Y_{CH3}$', '$Y_{CH3O}$', '$Y_{CH2O}$', '$Y_{C2H2}$'])

    samples.plot(scale, ['PODE', 'CO2', 'O2', 'CO', 'H2O', 'CH2O'],
                 style=['b-', 'r-', 'g-', 'y-', 'k-', 'm-'],
                 label=['$Y_{pode_n}$', '$Y_{CO2}$', '$Y_{O2}$', '$Y_{CO}$', '$Y_{H2O}$', '$Y_{CH2O}$'])

    plt.legend(loc="upper right")
    plt.xlabel(scale_name)
    plt.ylabel('mass fraction value')
    plt.title('{} PODE{} $\\Phi$={:.1f} p={}bar $T_0$={:.0f}K'.format(mechanism[0], pode,equivalence_ratio,
                                                                      reactorPressure, reactorTemperature))

    path = Path(__file__).parents[2] / 'data/00004-post-processing/{}_PODE{}_{}_{:.0f}_{}_{}_{}/plot_species_{}_{}.pdf'\
        .format(mechanism[0], pode, equivalence_ratio, reactorPressure, t_start, t_end, t_step, reactorTemperature,
                scale)
    plt.savefig(path)

    plt.show()


#######################################################################################################################
def plot_HR(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, scale, pode, t_start, t_end, t_step):
    # get data
    samples, scale_name = loaddata_samples(mechanism, equivalence_ratio, reactorPressure, reactorTemperature,
                                           scale, pode, t_start, t_end, t_step)

    samples.plot(scale, 'Q', style='b-')
    plt.xlabel(scale_name)
    plt.ylabel('Heat release [W/m$^3$]')

    delays = loaddata_delays(mechanism, equivalence_ratio, reactorPressure, pode, t_start, t_end, t_step)
    result = np.where(delays == reactorTemperature)
    first = delays.iloc[result[0], 2]
    main = delays.iloc[result[0], 3]
    textstr = 'first={:.4f}ms \nmain={:.4f}ms'.format(first.iloc[0], main.iloc[0])
    plt.text(0.6 * np.amax(samples[[scale]]), 0.4 * np.amax(samples['Q']), textstr, fontsize=12)

    plt.title('{} PODE{} $\\Phi$={:.1f} p={}bar $T_0$={:.0f}K'.format(mechanism[0], pode, equivalence_ratio,
                                                                      reactorPressure, reactorTemperature))

    path = Path(__file__).parents[2] / 'data/00004-post-processing/{}_PODE{}_{}_{:.0f}_{}_{}_{}/plot_HR_{}_{}.pdf'.format(
        mechanism[0], pode, equivalence_ratio, reactorPressure, t_start, t_end, t_step, reactorTemperature, scale)
    plt.savefig(path)

    plt.show()


#######################################################################################################################
def plot_PV(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, pode, t_start, t_end, t_step):
    # get data
    path = Path(__file__).parents[2] / 'data/00002-reactor-OME/{}_PODE{}_{}_{:.0f}_{}_{}_{}/samples_{}'.format(
        mechanism[0], pode, equivalence_ratio, reactorPressure, t_start, t_end, t_step, reactorTemperature)
    data = pd.read_csv(path)

    data[['PV']] = data[['PV']] / np.amax(data[['PV']])
    data[['time']] = data[['time']] * 1.e+3
    data.plot('time', 'PV', style='b-')
    plt.xlabel('time in ms')
    plt.ylabel('PV (normalized)')

    plt.title('{} PODE{} $\\Phi$={:.1f} p={}bar $T_0$={:.0f}K'.format(mechanism[0], pode, equivalence_ratio,
                                                                      reactorPressure, reactorTemperature))

    delays = loaddata_delays(mechanism, equivalence_ratio, reactorPressure, pode, t_start, t_end, t_step)
    result = np.where(delays == reactorTemperature)
    first = delays.iloc[result[0], 2]
    main = delays.iloc[result[0], 3]
    textstr = 'first={:.4f}ms \nmain={:.4f}ms'.format(first.iloc[0], main.iloc[0])
    plt.text(0.6 * np.amax(data[['time']]) * 1.e+3, 0.05, textstr, fontsize=12)

    path = Path(__file__).parents[2] / 'data/00004-post-processing/{}_PODE{}_{}_{:.0f}_{}_{}_{}/plot_PV_{}.pdf'.format(
        mechanism[0], pode, equivalence_ratio, reactorPressure, t_start, t_end, t_step, reactorTemperature)
    plt.savefig(path)

    plt.show()


#######################################################################################################################
def plot_time_scale(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, pode, t_start, t_end, t_step, scale='time'):
    # get data
    samples, scale_name = loaddata_samples(mechanism, equivalence_ratio, reactorPressure, reactorTemperature,
                                           scale, pode, t_start, t_end, t_step)

    time = samples['time'].to_numpy()
    x = np.zeros((len(time)))

    for i in range(0, len(time), 1):
        x[i] = i

    plt.plot(x, time)
    plt.xlabel('Samples taken')
    plt.ylabel('Time [ms]')

    delays = loaddata_delays(mechanism, equivalence_ratio, reactorPressure, pode, t_start, t_end, t_step)
    result = np.where(delays == reactorTemperature)
    first = delays.iloc[result[0], 2]
    main = delays.iloc[result[0], 3]
    textstr = 'first={:.4f}ms \nmain={:.4f}ms'.format(first.iloc[0], main.iloc[0])
    plt.text(0.6 * np.amax(samples[['time']]), 0.4 * np.amax(samples['Q']), textstr, fontsize=12)

    plt.title('{} PODE{} $\\Phi$={:.1f} p={}bar $T_0$={:.0f}K'.format(mechanism[0], pode, equivalence_ratio,
                                                                      reactorPressure, reactorTemperature))

    path = Path(__file__).parents[2] / 'data/00004-post-processing/{}_PODE{}_{}_{:.0f}_{}_{}_{}/plot_time_scale_{}.pdf'.format(
        mechanism[0], pode, equivalence_ratio, reactorPressure, t_start, t_end, t_step, reactorTemperature)
    plt.savefig(path)

    plt.show()