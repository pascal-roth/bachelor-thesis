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

plt.style.use('stfs_2')


#######################################################################################################################
def loaddata_samples(mechanism, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature, scale, pode, category):
    path = Path(__file__).parents[2] / 'data/00000-reactor-OME/{}/{}_{}_samples.csv'.format(mechanism[0], nbr_run,
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

    # define the name of the x_axis
    if scale == 'PV':
        scale_name = '$Y_cloaddata_samples(mechanism, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature, pode, category)$ (normalized)'
    elif scale == 'time':
        scale_name = 'time in ms'
    else:
        print('wrong scale parameter')

    return data, scale_name


def create_text(mechanism, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature, pode, category):
    path = Path(__file__).parents[2] / 'data/00002-reactor-OME/{}/{}_{}_delays.csv'.format(mechanism[0], nbr_run,
                                                                                           category)
    data = pd.read_csv(path)

    # Select only the data needed for the plot
    data = data[data.pode == pode]
    data = data[data.phi == equivalence_ratio]
    data = data[data.P_0 == reactorPressure * ct.one_atm]

    if data.empty:
        print('WARNING:Entered parameter setting not capturad in data!')

    result = np.where(data == reactorTemperature)
    first = data.iloc[result[0], 5]
    main = data.iloc[result[0], 6]
    textstr = 'first IDT={:.4f}ms \nmain IDT ={:.4f}ms'.format(first.iloc[0], main.iloc[0])
    return textstr


#######################################################################################################################
def plot_thermo(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, scale, pode, nbr_run, category):
    # get data
    samples, scale_name = loaddata_samples(mechanism, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature,
                                           scale, pode, category)

    # Normalize the PV
    samples[['PV']] = samples[['PV']] / np.amax(samples[['PV']])

    fig = plt.figure()
    ax1 = samples.plot(scale, 'T', style='b-', ax=fig.add_subplot(221))
    ax1.set_xlabel(scale_name)
    ax1.set_ylabel('T [K]')

    ax2 = samples.plot(scale, 'P', style='b-', ax=fig.add_subplot(222))
    ax2.set_xlabel(scale_name)
    ax2.set_ylabel('P [Bar]')

    ax3 = samples.plot(scale, 'V', style='b-', ax=fig.add_subplot(223))
    ax3.set_xlabel(scale_name)
    ax3.set_ylabel('V [m$^3$]')

    plt.tight_layout()
    plt.figlegend(['$\Phi$ = {}\np = {}bar\n$T_0$ = {}K'.format(equivalence_ratio, reactorPressure,
                                                                reactorTemperature)], loc='lower right')
    plt.tight_layout()

    path = Path(__file__).parents[2] / 'data/00001-post-processing/{}/{}/plot_thermo_PODE{}_phi{}_p{:.0f}_T{}_{}.pdf'. \
        format(mechanism[0], nbr_run, pode, equivalence_ratio, reactorPressure, reactorTemperature, scale)
    plt.savefig(path)

    plt.show()


#######################################################################################################################
def plot_species(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, scale, pode, nbr_run, category):

    # get data
    samples, scale_name = loaddata_samples(mechanism, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature,
                                           scale, pode, category)

    # Normalize the PV
    samples[['PV']] = samples[['PV']] / np.amax(samples[['PV']])

    samples.plot(scale, ['PODE', 'CO2', 'O2', 'CO', 'H2O', 'CH2O'],
                 style=['b-', 'r-', 'g-', 'y-', 'k-', 'm-'],
                 label=['$Y_{pode_n}$', '$Y_{CO2}$', '$Y_{O2}$', '$Y_{CO}$', '$Y_{H2O}$', '$Y_{CH2O}$'], figsize=(9, 6))

    # samples.plot(scale, ['PODE', 'O2'],
    #              style=['b-', 'g-'],
    #              label=['$Y_{pode_n}$', '$Y_{CO2}$'], figsize=(9, 6))

    # samples.plot(scale, ['CO2', 'CO', 'H2O'],
    #              style=['r-', 'y-', 'k-'],
    #              label=['$Y_{CO2}$', '$Y_{CO}$', '$Y_{H2O}$'], figsize=(9, 6))

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=6, prop={'size': 14})
    plt.xlabel(scale_name)
    plt.ylabel('Y')
    plt.title('PODE{} $\\Phi$={:.2f} p={}bar $T_0$={:.0f}K'.format(pode, equivalence_ratio,
                                                                   reactorPressure, reactorTemperature))

    plt.tight_layout()

    path = Path(__file__).parents[2] / 'data/00001-post-processing/{}/{}/plot_species_PODE{}_phi{}_p{:.0f}_T{}_{}.pdf' \
        .format(mechanism[0], nbr_run, pode, equivalence_ratio, reactorPressure, reactorTemperature, scale)
    plt.savefig(path)

    plt.show()


#######################################################################################################################
def plot_HRR(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, scale, pode, nbr_run, category):
    # get data
    samples, scale_name = loaddata_samples(mechanism, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature,
                                           scale, pode, category)

    # Normalize the PV
    samples[['PV']] = samples[['PV']] / np.amax(samples[['PV']])

    samples.plot(scale, 'HRR', style='b-')
    plt.xlabel(scale_name)
    plt.ylabel('HRR [W/kg/m$^3$]')

    # textstr = create_text(mechanism, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature, pode, category)
    # plt.text(0.05 * np.amax(samples[[scale]]), 0.4 * np.amax(samples['HRR']), textstr, fontsize=12)

    plt.title('PODE{} $\\Phi$={:.2f} p={}bar $T_0$={:.0f}K'.format(pode, equivalence_ratio,
                                                                   reactorPressure, reactorTemperature))

    plt.tight_layout()

    path = Path(__file__).parents[2] / 'data/00001-post-processing/{}/{}/plot_HRR_PODE{}_phi{}_p{:.0f}_T{}_{}.pdf'. \
        format(mechanism[0], nbr_run, pode, equivalence_ratio, reactorPressure, reactorTemperature, scale)
    plt.savefig(path)

    plt.show()


#######################################################################################################################
def plot_PV(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, pode, nbr_run, category, scale='time'):
    # get data
    samples, scale_name = loaddata_samples(mechanism, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature,
                                           scale, pode, category)

    # plot PV as scatter plot
    samples.plot.scatter('time', 'PV', style='m-', figsize=(9, 6))
    plt.xlabel('time [ms]')
    plt.ylabel('$Y_c$')

    plt.title('{} PODE{} $\\Phi$={:.2f} p={}bar $T_0$={:.0f}K'.format(mechanism[0], pode, equivalence_ratio,
                                                                      reactorPressure, reactorTemperature))

    plt.tight_layout()

    path = Path(__file__).parents[2] / 'data/00001post-processing/{}/{}/plot_PV_scatter_PODE{}_phi{}_p{:.0f}_T{}.pdf'. \
        format(mechanism[0], nbr_run, pode, equivalence_ratio, reactorPressure, reactorTemperature)
    plt.savefig(path)
    
    plt.show()

    # plot PV as plot
    samples.plot('time', 'PV', style='m-', figsize=(9, 6))
    plt.xlabel('time [ms]')
    plt.ylabel('$Y_c$')

    plt.title('{} PODE{} $\\Phi$={:.2f} p={}bar $T_0$={:.0f}K'.format(mechanism[0], pode, equivalence_ratio,
                                                                      reactorPressure, reactorTemperature))

    plt.tight_layout()

    path = Path(__file__).parents[2] / 'data/00001-post-processing/{}/{}/plot_PV_process_PODE{}_phi{}_p{:.0f}_T{}.pdf'. \
        format(mechanism[0], nbr_run, pode, equivalence_ratio, reactorPressure, reactorTemperature)
    plt.savefig(path)
    
    plt.show()

    # show the species of the PV
    samples[['CH2O']] = samples[['CH2O']] * 0.5
    samples[['H2O']] = samples[['H2O']] * 0.5
    samples[['CO2']] = samples[['CO2']] * 0.05
    pode_t0 = samples[['PODE']].iloc[0]
    pode_t0_series = np.ones((len(samples), 1)) * pode_t0[0]
    samples[['PODE']] = (pode_t0_series - samples[['PODE']].to_numpy()) * 0.5

    samples.plot('time', ['PODE', 'H2O', 'CH2O', 'CO2', 'PV'],
                 style=['r-', 'b-', 'k-', '-y', 'm-'],
                 label=['$Y_{PODE}$', '$Y_{H2O}$', '$Y_{CH2O}$', '$Y_{CO2}$', '$Y_c$'],
                 figsize=(9, 6))

    # samples.plot('time', ['PODE', 'H2O', 'PV'],
    #              style=['r-', 'b-', 'm-'],
    #              label=['$Y_{PODE}$', '$Y_{H2O}$', 'PV'],
    #              figsize=(9, 6))

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=5, prop={'size': 14})
    plt.xlabel('time [ms]')
    plt.ylabel('Y')
    plt.title('PODE{} $\\Phi$={:.2f} p={}bar $T_0$={:.0f}K'.format(pode, equivalence_ratio,
                                                                   reactorPressure, reactorTemperature))

    plt.tight_layout()

    path = Path(__file__).parents[2] / 'data/00001-post-processing/{}/{}/plot_PV_species_PODE{}_phi{}_p{:.0f}_T{}.pdf' \
        .format(mechanism[0], nbr_run, pode, equivalence_ratio, reactorPressure, reactorTemperature)
    plt.savefig(path)

    plt.show()


#######################################################################################################################
def plot_time_scale(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, pode, nbr_run, category,
                    scale='time'):
    # get data
    samples, scale_name = loaddata_samples(mechanism, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature,
                                           scale, pode, category)

    time = samples['time'].to_numpy()
    x = np.zeros((len(time)))

    for i in range(0, len(time), 1):
        x[i] = i

    fig, ax1 = plt.subplots(figsize=(9, 6))

    color = 'tab:red'
    ax1.set_xlabel(scale_name)
    ax1.set_ylabel('$Y_c$', color=color)
    ax1.plot(time, samples[['PV']], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Samples taken', color=color)  # we already handled the x-label with ax1
    ax2.plot(time, x, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('PODE{} $\\Phi$={:.2f} p={}bar $T_0$={:.0f}K'.format(pode, equivalence_ratio,
                                                                   reactorPressure, reactorTemperature))

    plt.tight_layout()

    path = Path(__file__).parents[2] / 'data/00001-post-processing/{}/{}/plot_time_scale_PODE{}_phi{}_p{:.0f}_T{}.pdf'. \
        format(mechanism[0], nbr_run, pode, equivalence_ratio, reactorPressure, reactorTemperature)
    plt.savefig(path)
    
    plt.show()


#######################################################################################################################
def plot_T_and_HRR(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, scale, pode, nbr_run, category):
    # get data
    samples, scale_name = loaddata_samples(mechanism, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature,
                                           scale, pode, category)

    # Normalize the PV
    samples[['PV']] = samples[['PV']] / np.amax(samples[['PV']])

    fig, ax1 = plt.subplots(figsize=(9, 6))

    color = 'tab:red'
    ax1.set_xlabel(scale_name)
    ax1.set_ylabel('T [K]', color=color)
    ax1.plot(samples[[scale]], samples[['T']], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('HRR [W/kg/m$^3$]', color=color)  # we already handled the x-label with ax1
    ax2.plot(samples[[scale]], samples[['HRR']], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    path = Path(__file__).parents[2] / 'data/00001-post-processing/{}/{}/plot_T_HRR_PODE{}_phi{}_p{:.0f}_T{}_{}.pdf'. \
        format(mechanism[0], nbr_run, pode, equivalence_ratio, reactorPressure, reactorTemperature, scale)
    plt.savefig(path)

    plt.show()

#######################################################################################################################
def plot_T_and_P(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, scale, pode, nbr_run, category):
    # get data
    samples, scale_name = loaddata_samples(mechanism, nbr_run, equivalence_ratio, reactorPressure, reactorTemperature,
                                           scale, pode, category)

    # Normalize the PV
    samples[['PV']] = samples[['PV']] / np.amax(samples[['PV']])

    fig, ax1 = plt.subplots(figsize=(9, 6))

    color = 'tab:red'
    ax1.set_xlabel(scale_name)
    ax1.set_ylabel('T [K]', color=color)
    ax1.plot(samples[[scale]], samples[['T']], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('P [bar]', color=color)  # we already handled the x-label with ax1
    ax2.plot(samples[[scale]], samples[['P']], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    path = Path(__file__).parents[2] / 'data/00001-post-processing/{}/{}/plot_T_P_PODE{}_phi{}_p{:.0f}_T{}_{}.pdf'. \
        format(mechanism[0], nbr_run, pode, equivalence_ratio, reactorPressure, reactorTemperature, scale)
    plt.savefig(path)

    plt.show()