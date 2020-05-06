#######################################################################################################################
# Plot the ignition delays from the reactor model and from experiments
#######################################################################################################################

# Import packages
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cantera as ct
import pandas as pd


# %% initialise dataloaders
def loaddata_delays(mechanism, equivalence_ratio, reactorPressure, pode, t_start, t_end, t_step):
    path = Path(__file__).parents[2] / 'data/00002-reactor-OME/{}_PODE{}_{}_{}_{}_{}_{}/delays'.format(
        mechanism[0], pode, equivalence_ratio, reactorPressure, t_start, t_end, t_step)
    data =  np.array(pd.read_csv(path))
    return data


def loaddata_exp(OME, reactorPressure, equivalence_ratio):
    path = Path(__file__).parents[2] / 'data/00004-post-processing/data_exp/Exp_{0}_{1}_{2}.csv'. \
        format(OME, reactorPressure, equivalence_ratio)
    data = np.array(pd.read_csv(path, delimiter=';', decimal=','))
    return data


# %% plot data
mechanism_all = np.array([['he_2018.xml'], ['cai_ome14_2019.xml'], ['sun_2017.xml']])


def plot_delays(pode, equivalence_ratio, reactorPressure, t_start, t_end, t_step):
    print('\nIgnition Delay plot started \n')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if pode <= 3:
        if (reactorPressure == 10 or reactorPressure == 20) and equivalence_ratio == 1.0:
            exp = loaddata_exp('OME' + str(pode), reactorPressure, equivalence_ratio)
            ax.semilogy(exp[:, 0], exp[:, 1], 'bx', label='exp_OME' + str(pode))

        he = np.flip(loaddata_delays(mechanism_all[0], equivalence_ratio, reactorPressure, pode, t_start, t_end, t_step), axis=0)
        cai = np.flip(loaddata_delays(mechanism_all[1], equivalence_ratio, reactorPressure, pode, t_start, t_end, t_step), axis=0)
        sun = np.flip(loaddata_delays(mechanism_all[2], equivalence_ratio, reactorPressure, pode, t_start, t_end, t_step), axis=0)

        ax.semilogy(1000 / he[:, 0], he[:, 2], 'r-', label='sim_he_2018')
        ax.semilogy(1000 / cai[:, 0], cai[:, 2], 'g-', label='sim_cai_2019')
        ax.semilogy(1000 / sun[:, 0], sun[:, 2], 'y-', label='sim_sun_2017')

    elif pode == 4:
        if reactorPressure == 10 and equivalence_ratio == 1.0:
            exp = loaddata_exp('OME4', reactorPressure, equivalence_ratio)
            ax.semilogy(exp[:, 0], exp[:, 1], 'bx', label='exp_OME4')

        cai = np.flip(loaddata_delays(mechanism_all[2, 0], equivalence_ratio, reactorPressure, pode, t_start, t_end, t_step))
        ax.semilogy(1000 / cai[:, 0], cai[:, 2], 'g-', label='sim_cai_2019')

    else:
        print('Entered PODE > 4 and not focus of this work')

    ax.set_ylabel('Ignition Delay (ms)')
    ax.set_xlabel(r'$\frac{1000}{T (K)}$', fontsize=18)

    # Add a second axis on top to plot the temperature for better readability
    ax2 = ax.twiny()
    ticks = ax.get_xticks()
    ax2.set_xticks(ticks)
    ax2.set_xticklabels((1000 / ticks).round(1))
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel(r'Temperature: $T(K)$')

    textstr = '$\\Phi$={:.1f}\np={:.0f}bar'.format(equivalence_ratio, reactorPressure)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top')

    ax.set_yscale('log')
    ax.legend(loc='lower right')

    path = Path(__file__).parents[2] / 'data/00004-post-processing/delays_{}_{}_PODE{}_{}_{}_{}.pdf'\
        .format(equivalence_ratio, reactorPressure, pode, t_start, t_end, t_step)
    plt.savefig(path)

    plt.show()