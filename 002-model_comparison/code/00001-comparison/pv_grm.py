import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from fc_GRM import load_GRM
from fc_HR import load_samples


def plot_IDT(IDT_GRM_time):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    ax.semilogy(1000 / IDT_GRM_time[:, 0], IDT_GRM_time[:, 1], 'b-', label='GRM')

    ax.set_ylabel('IDT [ms]')
    ax.set_xlabel('1000/T [1/K]')

    # Add a second axis on top to plot the temperature for better readability
    ax2 = ax.twiny()
    ticks = ax.get_xticks()
    ax2.set_xticks(ticks)
    ax2.set_xticklabels((1000 / ticks).round(1))
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel('T [K]')

    ax.set_yscale('log')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=4,
              prop={'size': 14})

    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    # load grm samples
    samples_grm = load_GRM(pode=3, phi=1, pressure=40, temperature=1080)

    feature_select = {'pode': [3], 'phi': 1.0, 'P_0': [40], 'T_0': [1080]}
    features = ['time']
    labels = ['PV']
    samples_hr_time, samples_hr_pv = load_samples('cai', nbr_run='002', feature_select=feature_select,
                                                  features=features, labels=labels, select_data='include',
                                                  category='test')

    # plt.plot(samples_grm['time'], samples_grm['PV'], '-r', label='PV_GRM')
    # plt.plot(samples_hr_time * 1.e+3, samples_hr_pv, '-b', label='PV_HR')
    # plt.xlabel('time [ms]')
    # plt.ylabel('PV')
    #
    # plt.legend()
    # plt.show()

    feature_select = {'pode': [3], 'phi': 1.0, 'P_0': [40], 'T_0': [1160]}
    features = ['PV']
    labels = ['T']
    samples_hr_time, samples_hr_pv = load_samples('cai', nbr_run='002', feature_select=feature_select,
                                                  features=features, labels=labels, select_data='include',
                                                  category='test')

    # plt.plot(samples_grm['PV'], samples_grm['T'], '-r', label='PV_GRM')
    # plt.plot(samples_hr_time, samples_hr_pv, '-b', label='PV_HR')
    plt.xlim(xmin=0, xmax=0.05)
    # plt.xlabel('PV')
    # plt.ylabel('T')
    #
    # plt.legend()
    # plt.show()

    # labels = ['O2']
    # samples_hr_time, samples_hr_o2 = load_samples('cai', nbr_run='002', feature_select=feature_select,
    #                                               features=features, labels=labels, select_data='include',
    #                                               category='test')
    #
    # plt.plot(samples_grm['time'], samples_grm['O2'], '-r', label='Y_O2_GRM')
    plt.plot(samples_grm['time'], samples_grm['H2O'], '-b', label='Y_H2O_GRM')
    # plt.plot(samples_grm['time'], samples_grm['CO'], '-g', label='Y_CO_GRM')
    # plt.plot(samples_grm['time'], samples_grm['CO2'], '-y', label='Y_CO2_GRM')
    # plt.plot(samples_grm['time'], samples_grm['I1'], '-m', label='Y_I1_GRM')
    # plt.plot(samples_grm['time'], samples_grm['Y'], '-c', label='Y_Y_GRM')
    # plt.plot(samples_grm['time'], samples_grm['I2'], '-r', label='Y_I2_GRM')
    # plt.plot(samples_grm['time'], samples_grm['N2'], '-b', label='Y_N2_GRM')
    #
    # plt.plot(samples_hr_time * 1.e+3, samples_hr_o2, '-b', label='Y_O2_HR')
    #
    plt.xlabel('time [ms]')
    plt.ylabel('Y')

    plt.legend()
    plt.show()

    # IDT_GRM_time = np.zeros((15, 2))
    #
    # for i, temperature_run in enumerate(np.arange(680, 1240 + 40, 40)):
    #     samples_grm = load_GRM(pode=3, phi=1, pressure=40, temperature=temperature_run)
    #
    #     IDT_location = np.argmax(samples_grm['HRR'])
    #     IDT_time = samples_grm['time'].iloc[IDT_location]
    #
    #     IDT_GRM_time[i, :] = (temperature_run, IDT_time)
    #
    # plot_IDT(IDT_GRM_time)
