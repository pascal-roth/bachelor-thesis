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
from scipy.signal import find_peaks
from pathlib import Path


#######################################################################################################################
def loaddata_samples(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, scale, pode):
    path = Path(__file__).parents[2] / 'data/00002-reactor-OME/{}_PODE{}_{}_{:.0f}_650_1250_15/samples_{}.npy'.format(
                                  mechanism[0], pode, equivalence_ratio, reactorPressure, reactorTemperature)
    data = np.load(path)

    if scale == 'PV':
        x_axis = data[:, 1]
        x_axis = x_axis/np.amax(x_axis)
        scale_name = 'PV (normalized)'
    elif scale == 'time':
        x_axis = data[:, 0] * 1.e+3
        scale_name = 'time in ms'
    else:
        print('wrong scale parameter')

    data = data[:, 2:]

    return data, x_axis, scale_name


def loaddata_delays(mechanism, equivalence_ratio, reactorPressure, pode):
    path = Path(__file__).parents[2] / 'data/00002-reactor-OME/{}_PODE{}_{}_{:.0f}_650_1250_15/delays.npy'.format(
        mechanism[0], pode, equivalence_ratio, reactorPressure)
    data = np.load(path)
    return data


#######################################################################################################################
def plot_thermo(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, scale, pode):

    # get data
    samples, x_axis, scale_name = loaddata_samples(mechanism, equivalence_ratio, reactorPressure, reactorTemperature,
                                                   scale, pode)

    plt.clf()
    plt.subplot(2, 2, 1)
    h = plt.plot(x_axis, samples[:, 2], 'b-')
    plt.xlabel(scale_name)
    plt.ylabel('Temperature (K)')

    plt.subplot(2, 2, 2)
    plt.plot(x_axis, samples[:, 3] / 1e5, 'b-')
    plt.xlabel(scale_name)
    plt.ylabel('Pressure (Bar)')

    plt.subplot(2, 2, 3)
    plt.plot(x_axis, samples[:, 4], 'b-')
    plt.xlabel(scale_name)
    plt.ylabel('Volume (m$^3$)')

    plt.tight_layout()
    plt.figlegend(h, ['$\Phi$ = {} \n p = {} \n $T_0$ = {}'. format(equivalence_ratio, reactorPressure, reactorTemperature)], loc='lower right')

    path = Path(__file__).parents[2] / 'data/00004-post-processing/{}_PODE{}_{}_{:.0f}_650_1250_15/plot_thermo_{}_{}.pdf'.format(
            mechanism[0], pode, equivalence_ratio, reactorPressure, reactorTemperature, scale)
    plt.savefig(path)

    plt.show()


#######################################################################################################################
def plot_species(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, scale, pode):
    # get data
    samples, x_axis, scale_name = loaddata_samples(mechanism, equivalence_ratio, reactorPressure, reactorTemperature,
                                                   scale, pode)

    plt.plot(x_axis, samples[:, 5], 'b-', label='$Y_{pode_n}$')
    plt.plot(x_axis, samples[:, 6], 'r-', label='$Y_{CO2}$')
    plt.plot(x_axis, samples[:, 7], 'g-', label='$Y_{O2}$')
    plt.plot(x_axis, samples[:, 8], 'y-', label='$Y_{CO}$')
    plt.plot(x_axis, samples[:, 9], 'c-', label='$Y_{H20}$')
    plt.plot(x_axis, samples[:, 10], 'm-', label='$Y_{OH}$')
    plt.plot(x_axis, samples[:, 11], 'k-', label='$Y_{H2O2}$')

    plt.legend(loc="upper right")
    plt.xlabel(scale_name)
    plt.ylabel('mass fraction value')
    plt.title('$\\Phi$={:.1f} p={:.0f} bar $T_0$={:.0f}K'.format(equivalence_ratio, reactorPressure, reactorTemperature))

    path = Path(__file__).parents[2] / 'data/00004-post-processing/{}_PODE{}_{}_{:.0f}_650_1250_15/plot_species_{}_{}.pdf'.format(
        mechanism[0], pode, equivalence_ratio, reactorPressure, reactorTemperature, scale)
    plt.savefig(path)

    plt.show()


#######################################################################################################################
def plot_HR(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, scale, pode):
    # get data
    samples, x_axis, scale_name = loaddata_samples(mechanism, equivalence_ratio, reactorPressure, reactorTemperature,
                                                   scale, pode)

    plt.plot(x_axis, samples[:, 2], 'b-')
    plt.xlabel(scale_name)
    plt.ylabel('Heat release [W/m$^3$]')

    delays = loaddata_delays(mechanism, equivalence_ratio, reactorPressure, pode)
    result = np.where(delays == reactorTemperature)
    first = delays[result[0], 1]
    main = delays[result[0], 2]
    textstr = 'first={}ms \nmain={}ms'.format(first[0], main[0])
    plt.text(0.95 * np.amax(x_axis), 0.95 * np.amax(samples[:, 2]), textstr, fontsize=14, verticalalignment='top')

    plt.title('$\\Phi$={:.1f} p={:.0f} bar $T_0$={:.0f}K'.format(equivalence_ratio, reactorPressure, reactorTemperature))

    path = Path(__file__).parents[2] / 'data/00004-post-processing/{}_PODE{}_{}_{:.0f}_650_1250_15/plot_HR_{}_{}.pdf'.format(
        mechanism[0], pode, equivalence_ratio, reactorPressure, reactorTemperature, scale)
    plt.savefig(path)

    plt.show()


#######################################################################################################################
def plot_PV(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, pode):
    # get data
    path = Path(__file__).parents[2] / 'data/00002-reactor-OME/{}_PODE{}_{}_{:.0f}_650_1250_15/samples_{}.npy'.format(
        mechanism[0], pode, equivalence_ratio, reactorPressure, reactorTemperature)
    data = np.load(path)

    plt.plot(data[:, 0] * 1.e+3, data[:, 1] / np.amax(data[:, 1]), 'b-')
    plt.xlabel('time in ms')
    plt.ylabel('PV (normalized)')

    plt.title('$\\Phi$={:.1f} p={}bar $T_0$={:.0f}K'.format(equivalence_ratio, reactorPressure, reactorTemperature))

    path = Path(__file__).parents[2] / 'data/00004-post-processing/{}_PODE{}_{}_{:.0f}_650_1250_15/plot_PV_{}.pdf'.format(
        mechanism[0], pode, equivalence_ratio, reactorPressure, reactorTemperature)
    plt.savefig(path)

    plt.show()