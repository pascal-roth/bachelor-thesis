import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from fc_GRM import load_GRM
from fc_HR import load_samples

if __name__ == '__main__':
    # load grm samples
    samples_grm = load_GRM(pode=3, phi=1, pressure=40, temperature=680)

    feature_select = {'pode': [3], 'phi': 1.0, 'P_0': [40], 'T_0': [680]}
    features = ['time']
    labels = ['PV']
    samples_hr_time, samples_hr_pv = load_samples('cai', nbr_run='002', feature_select=feature_select,
                                                  features=features, labels=labels, select_data='include',
                                                  category='test')

    plt.plot(samples_grm['time'], samples_grm['PV'], '-r', label='PV_GRM')
    # plt.plot(samples_hr_time * 1.e+3, samples_hr_pv, '-b', label='PV_HR')
    plt.xlabel('time [ms]')
    plt.ylabel('PV')

    plt.legend()
    plt.show()

    # labels = ['O2']
    # samples_hr_time, samples_hr_o2 = load_samples('cai', nbr_run='002', feature_select=feature_select,
    #                                               features=features, labels=labels, select_data='include',
    #                                               category='test')
    #
    # plt.plot(samples_grm['time'], samples_grm['O2'], '-r', label='Y_O2_GRM')
    # plt.plot(samples_grm['time'], samples_grm['H2O'], '-b', label='Y_H2O_GRM')
    # plt.plot(samples_grm['time'], samples_grm['CO'], '-g', label='Y_CO_GRM')
    # plt.plot(samples_grm['time'], samples_grm['CO2'], '-y', label='Y_CO2_GRM')
    # plt.plot(samples_grm['time'], samples_grm['I1'], '-m', label='Y_I1_GRM')
    # plt.plot(samples_grm['time'], samples_grm['Y'], '-c', label='Y_Y_GRM')
    # plt.plot(samples_grm['time'], samples_grm['I2'], '-r', label='Y_I2_GRM')
    # plt.plot(samples_grm['time'], samples_grm['N2'], '-b', label='Y_N2_GRM')
    #
    # plt.plot(samples_hr_time * 1.e+3, samples_hr_o2, '-b', label='Y_O2_HR')
    #
    # plt.xlabel('time [ms]')
    # plt.ylabel('Y')
    #
    # plt.legend()
    # plt.show()