#######################################################################################################################
# Compare the different detailed mechanism
#######################################################################################################################

# Import packages
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cantera as ct

# %% Load arrays
mechanism_all = np.array([['he_2018.xml', 'DMM3'], ['cai_ome14_2019.xml', 'OME3'], ['sun_2017.xml', 'DMM3']])


def loaddata(author, mechanism, equivalence_ratio):
    path = Path(__file__).parents[2] / 'data/00002-reactor-OME/{0}_{2}/delays_{1}_{2}.npy'.format(author, mechanism,
                                                                                                  equivalence_ratio)
    data = np.load(path)
    return data


delays_he_1_0 = loaddata('He_2018', mechanism_all[0, 0], '1.0')
delays_he_1_5 = loaddata('He_2018', mechanism_all[0, 0], '1.5')

# delays_cai_1_0 = loaddata('Cai_2019', mechanism_all[1, 0], '1.0')
# delays_cai_1_5 = loaddata('Cai_2019', mechanism_all[1, 0], '1.5')

delays_sun_1_0 = loaddata('Sun_2017', mechanism_all[2, 0], '1.0')
delays_sun_1_5 = loaddata('Sun_2017', mechanism_all[2, 0], '1.5')

# %% Split data in 20bar and 25bar starting conditions


def split_data(data):
    n = 0
    for i in range(0, data.shape[0], 1):
        if data[i, 1] == ct.one_atm*20:
            n += 1
    data_20 = data[:n, :]
    data_25 = data[n:, :]
    return data_20, data_25


delays_he_1_0_20, delays_he_1_0_25 = split_data(delays_he_1_0)
delays_he_1_5_20, delays_he_1_5_25 = split_data(delays_he_1_5)

# delays_cai_1_0_20, delays_cai_1_0_25 = split_data(delays_cai_1_0)
# delays_cai_1_5_20, delays_cai_1_5_25 = split_data(delays_cai_1_5)

delays_sun_1_0_20, delays_sun_1_0_25 = split_data(delays_sun_1_0)
delays_sun_1_5_20, delays_sun_1_5_25 = split_data(delays_sun_1_5)

# %% Plot delays and their difference
def plot_delays(he, sun): #, cai):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogy(1000 / he[:, 2], he[:, 3] , 'bx-', label='he_2018')
#    ax.semilogy(1000 / cai[:, 2], cai[:, 4] , 'ro-', label='cai_2019')
    ax.semilogy(1000 / sun[:, 2], sun[:, 4], 'ro-', label='sun_2017')

    ax.set_ylabel('Ignition Delay (ms)')
    ax.set_xlabel(r'$\frac{1000}{T (K)}$', fontsize=18)

    # Add a second axis on top to plot the temperature for better readability
    ax2 = ax.twiny()
    ticks = ax.get_xticks()
    ax2.set_xticks(ticks)
    ax2.set_xticklabels((1000 / ticks).round(1))
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel(r'Temperature: $T(K)$')

    textstr = '$\\Phi$={:.2f}\np={:.0f}bar'.format(he[0, 0], he[0, 1] / 1.e+5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top')

    ax.set_yscale('log')
    ax.legend(loc='lower right')

    path = Path(__file__).parents[2] / 'data/00004-mechanism-comparison/plt_comparison_{}_{}.png'.format(he[0, 0],
                                                                                                         he[0, 1] / 
                                                                                                         1.e+5)
    plt.savefig(path)
    plt.show()

plot_delays(delays_he_1_0_20, delays_sun_1_0_20) #, delays_cai_1_0_20)
plot_delays(delays_he_1_5_20, delays_sun_1_5_20) #, delays_cai_1_5_20)
plot_delays(delays_he_1_0_25, delays_sun_1_0_25) #, delays_cai_1_0_25)
plot_delays(delays_he_1_5_25, delays_sun_1_5_25) #, delays_cai_1_5_25)

