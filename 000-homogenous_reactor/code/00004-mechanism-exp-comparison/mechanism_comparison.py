#######################################################################################################################
# Compare the different detailed mechanism between each other and with experimental data
#######################################################################################################################

# Import packages
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cantera as ct
import pandas as pd

# %% Load arrays
mechanism_all = np.array([['he_2018.xml', 'DMM3'], ['cai_ome14_2019.xml', 'OME3'], ['sun_2017.xml', 'DMM3']])


def loaddata_mech(author, mechanism, equivalence_ratio):
    path = Path(__file__).parents[2] / 'data/00002-reactor-OME/{0}_{2}/delays_{1}_{2}.npy'.format(author, mechanism,
                                                                                                  equivalence_ratio)
    data = np.load(path)
    return data


delays_he_1_0 = np.flip(loaddata_mech('He_2018', mechanism_all[0, 0], '1.0'), axis=0)
delays_he_1_5 = np.flip(loaddata_mech('He_2018', mechanism_all[0, 0], '1.5'), axis=0)

# delays_cai_1_0 = np.flip(loaddata_mech('Cai_2019', mechanism_all[1, 0], '1.0'), axis=0)
# delays_cai_1_5 = np.flip(loaddata_mech('Cai_2019', mechanism_all[1, 0], '1.5'), axis=0)

delays_sun_1_0 = np.flip(loaddata_mech('Sun_2017', mechanism_all[2, 0], '1.0'), axis=0)
delays_sun_1_5 = np.flip(loaddata_mech('Sun_2017', mechanism_all[2, 0], '1.5'), axis=0)


def loaddata_exp(OME, pressure, equivalence_ratio):
    path = Path(__file__).parents[2] / 'data/00004-mechanism-exp-comparison/Exp_{0}_{1}_{2}.csv'. \
        format(OME, pressure, equivalence_ratio)
    data = np.array(pd.read_csv(path, delimiter=';', decimal=','))
    return data


exp_OME1_20_1_0 = loaddata_exp('OME1', '20', '1_0')
exp_OME2_20_1_0 = loaddata_exp('OME2', '20', '1_0')
exp_OME3_20_1_0 = loaddata_exp('OME3', '20', '1_0')

exp_OME3_10_1_0 = loaddata_exp('OME3', '10', '1_0')
exp_OME4_10_1_0 = loaddata_exp('OME4', '10', '1_0')


def loaddata_sim(mechanism, equivalence_ratio):
    path = Path(__file__).parents[2] / 'data/00002-reactor-OME/Interval-650-1250/delays_{}_{}.npy' \
        .format(mechanism, equivalence_ratio)
    data = np.load(path)
    return data


sim_delays_he_1_0_20 = np.flip(loaddata_sim(mechanism_all[0, 0], '1.0'), axis=0)
# sim_delays_cai_1_0_20 = np.flip(loaddata_sim(mechanism_all[1, 0], '1.0'), axis=0)
sim_delays_sun_1_0_20 = np.flip(loaddata_sim(mechanism_all[2, 0], '1.0'), axis=0)


# %% Split data in 20bar and 25bar starting conditions


def split_data(data):
    n = 0
    for i in range(0, data.shape[0], 1):
        if data[i, 1] == ct.one_atm * 25:
            n += 1
    data_25 = data[:n, :]
    data_20 = data[n:, :]
    return data_20, data_25


delays_he_1_0_20, delays_he_1_0_25 = split_data(delays_he_1_0)
delays_he_1_5_20, delays_he_1_5_25 = split_data(delays_he_1_5)

# delays_cai_1_0_20, delays_cai_1_0_25 = split_data(delays_cai_1_0)
# delays_cai_1_5_20, delays_cai_1_5_25 = split_data(delays_cai_1_5)

delays_sun_1_0_20, delays_sun_1_0_25 = split_data(delays_sun_1_0)
delays_sun_1_5_20, delays_sun_1_5_25 = split_data(delays_sun_1_5)


# %% Plot delays and their difference
def plot_delays(he, sun):  # , cai):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogy(1000 / he[:, 2], he[:, 3], 'bx-', label='sim_he_2018')
    #    ax.semilogy(1000 / cai[:, 2], cai[:, 4] , 'ro-', label='sim_cai_2019')
    ax.semilogy(1000 / sun[:, 2], sun[:, 4], 'ro-', label='sim_sun_2017')

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

    path = Path(__file__).parents[2] / 'data/00004-mechanism-exp-comparison/plt_comparison_{}_{:.0f}.png' \
        .format(he[0, 0], he[0, 1] / 1.e+5)
    plt.savefig(path)
    plt.show()


plot_delays(delays_he_1_0_20, delays_sun_1_0_20)  # , delays_cai_1_0_20)
plot_delays(delays_he_1_5_20, delays_sun_1_5_20)  # , delays_cai_1_5_20)
plot_delays(delays_he_1_0_25, delays_sun_1_0_25)  # , delays_cai_1_0_25)
plot_delays(delays_he_1_5_25, delays_sun_1_5_25)  # , delays_cai_1_5_25)


# %% Plot delays with the experimental data
def plot_exp(he, sun):  # , cai):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.semilogy(1000 / he[:, 2], he[:, 3], 'bo-', label='sim_he_2018')
    #    ax.semilogy(1000 / cai[:, 2], cai[:, 4] , 'ro-', label='sim_cai_2019')
    ax.semilogy(1000 / sun[:, 2], sun[:, 4], 'ro-', label='sim_sun_2017')
    ax.semilogy(exp_OME3_20_1_0[:, 0], exp_OME3_20_1_0[:, 1], 'bx', label='exp_OME3')
    ax.set_yscale('log')
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

    path = Path(__file__).parents[2] / 'data/00004-mechanism-exp-comparison/plt_exp_comparison_{}_{:.0f}.png' \
        .format(he[0, 0], he[0, 1] / 1.e+5)
    plt.savefig(path)
    plt.show()


plot_exp(sim_delays_he_1_0_20, sim_delays_sun_1_0_20)  # , sim_delays_cai_1_0_20)
