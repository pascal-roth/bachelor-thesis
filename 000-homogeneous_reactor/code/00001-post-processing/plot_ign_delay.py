#######################################################################################################################
# Plot the ignition delays from the reactor model and from experiments
#######################################################################################################################

# Import packages
import numpy as np
from pathlib import Path
import cantera as ct
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('stfs')


# %% initialise dataloaders
def loaddata_delays(mechanism, nbr_run, equivalence_ratio, reactorPressure, pode, category):
    path = Path(__file__).parents[2] / 'data/00000-reactor-OME/{}/{}_{}_delays.csv'.format(mechanism, nbr_run, category)
    data = pd.read_csv(path)

    # Select only the data needed for the plot
    data = data[data.pode == pode]
    data = data[data.phi == equivalence_ratio]
    data = data[data.P_0 == reactorPressure * ct.one_atm]

    data = np.array(data)
    return data[:, 4:]

# %% plot data
mechanism_all = np.array([['he_2018.xml'], ['cai_ome14_2019.xml'], ['sun_2017.xml']])


def plot_delays(mechanism, pode, equivalence_ratio, reactorPressure, nbr_run, category):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if pode <= 3:

        colors = ['r-', 'g-', 'y-']

        for i, mechanism_run in enumerate(mechanism):

            if len(mechanism) > 1:
                mechanism_run = mechanism_run[0]

            sim = np.flip(loaddata_delays(mechanism_run, nbr_run, equivalence_ratio, reactorPressure, pode, category), axis=0)
            ax.semilogy(1000 / sim[:, 0], sim[:, 2], colors[i], label=mechanism_run)

    elif pode == 4:
        cai = np.flip(loaddata_delays(mechanism_all[1], nbr_run, equivalence_ratio, reactorPressure, pode, category), axis=0)
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

    textstr = '$\\Phi$={:.2f}\np={:.0f}bar'.format(equivalence_ratio, reactorPressure)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top')

    ax.set_yscale('log')
    ax.legend(loc='lower right')

    path = Path(__file__).parents[2] / 'data/00001-post-processing/delays_PODE{}_phi{}_p{}.pdf'\
        .format(pode, equivalence_ratio, reactorPressure)
    plt.savefig(path)

    plt.show()