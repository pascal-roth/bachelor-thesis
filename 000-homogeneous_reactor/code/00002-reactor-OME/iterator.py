#######################################################################################################################
# Iteration through different temperatures, pressures, ... of the homogeneous reactor
#######################################################################################################################

# Import packages
import argparse
import numpy as np
import cantera as ct
from Homogeneous_Reactor import homogeneous_reactor
from pathlib import Path
import os
import pandas as pd


# %% Collect arguments
parser = argparse.ArgumentParser(description="Run homogeneous reactor model")

parser.add_argument("-mech", "--mechanism_input", type=str, choices=['he', 'sun', 'cai'], default='he',
                    help="chose reaction mechanism")

parser.add_argument("-phi", "--equivalence_ratio", type=float, default='1.0',
                    help="chose equivalence ratio")

parser.add_argument("-p", "--pressure", type=int, default=20,
                    help="chose reactor pressure")

parser.add_argument("--pode", type=int, choices=[1, 2, 3, 4], default=3,
                    help="chose degree of polymerization")

parser.add_argument("-t_0", "--temperature_start", type=int, default=650,
                    help="chose staring temperature of simulation")

parser.add_argument("-t_end", "--temperature_end", type=int, default=1250,
                    help="chose staring temperature of simulation")

parser.add_argument("-t_step", "--temperature_step", type=int, default=15,
                    help="chose staring temperature of simulation")

parser.add_argument("-inf_print", "--information_print", type=bool, default=True,
                    help="chose if basic information are displayed")

args = parser.parse_args()
print(args)

# %% Rename arguments
if args.mechanism_input == 'he':
    mechanism = np.array(['he_2018.xml', 'DMM' + str(args.pode)])
    t_step = 1.e-6
elif args.mechanism_input == 'cai':
    mechanism = np.array(['cai_ome14_2019.xml', 'OME' + str(args.pode)])
    t_step = 1.e-6
elif args.mechanism_input == 'sun':
    mechanism = np.array(['sun_2017.xml', 'DMM' + str(args.pode)])
    t_step = 1.e-6

if args.pode == 1:
    mechanism[1] = 'CH3OCH2OCH3'
elif args.pode == 4 and not mechanism[0] == 'cai_ome14_2019.xml':
    print ('WARNING: This mechanism is not available for PODE4, will be downgraded to PODE3')
    mechanism[1] = 'DMM3'

if args.equivalence_ratio == 0.0:
    equivalence_ratio = np.array([0.5, 1.0, 1.5])
else:
    equivalence_ratio = np.array([args.equivalence_ratio])

if args.pressure == 0:
    reactorPressure = ct.one_atm * np.array([10, 20, 40])
else:
    reactorPressure = ct.one_atm * np.array([args.pressure])

reactorTemperature_start = args.temperature_start
reactorTemperature_end = args.temperature_end
reactorTemperature_step = args.temperature_step

# Define end time and time step
t_end = 0.010

# create an array for the different samples/ the ignition delays and decide if to save them
save_samples = True
save_delays = True


# %% Iterate between the parameter settings
if args.information_print is True:
    print('the used mechanism is {} with fuel {}'.format(mechanism[0], mechanism[1]))

for ii, equivalence_ratio_run in enumerate(equivalence_ratio):  # enumerate through different equivalence ratios

    for i, reactorPressure_run in enumerate(reactorPressure):  # enumerate through different pressures

        if not ((save_delays is False) and (save_samples is False)):
            path = Path(__file__).resolve()
            path_dir = path.parents[2] / 'data/00002-reactor-OME/{}_PODE{}_{}_{:.0f}_{}_{}_{}'.format(
                                mechanism[0], args.pode, equivalence_ratio_run, reactorPressure_run / ct.one_atm,
                                reactorTemperature_start, reactorTemperature_end, reactorTemperature_step)
            os.makedirs(path_dir)
            path_plt = path.parents[2] / 'data/00004-post-processing/{}_PODE{}_{}_{:.0f}_{}_{}_{}'.format(
                                mechanism[0], args.pode, equivalence_ratio_run, reactorPressure_run / ct.one_atm,
                                reactorTemperature_start, reactorTemperature_end, reactorTemperature_step)
            os.makedirs(path_plt)

        if save_delays is True:
            delays = np.zeros(
                (((reactorTemperature_end - reactorTemperature_start) // reactorTemperature_step), 3))
            n = 0

        for reactorTemperature in range(reactorTemperature_start, reactorTemperature_end, reactorTemperature_step):
            # start homogeneous reactor model with defined settings
            # values vector: 'time (s)', 'PV', 'equivalence_ratio', 'Q', 'T (K)', 'P1 (Bar)', 'V1 (m3)', 'YPODE',
            # 'YCO2', 'YO2', 'Y_CO', 'Y_H2O', 'Y_OH', 'Y_H2O2'
            values, first_ignition_delay, main_ignition_delay = homogeneous_reactor(mechanism,
                                                                                    equivalence_ratio_run,
                                                                                    reactorPressure_run,
                                                                                    reactorTemperature,
                                                                                    t_end, t_step)

            # save species development with the parameter setting
            if save_samples is True:
                path_sample = '{}/samples_{}.csv'.format(path_dir, reactorTemperature)
                values.to_csv(path_sample)

            # saving ignition delays for the parameter setting
            if save_delays is True and 0 < main_ignition_delay < t_end*1.e+3*0.99:
                delays[n] = (reactorTemperature, first_ignition_delay, main_ignition_delay)
                n += 1
            elif save_delays is True:  # cancelling rows if ignition didn't happened
                n_rows, _ = delays.shape
                delays = delays[:(n_rows - 1), :]

            # print information about parameter setting and ignition
            if args.information_print is True and 0 < main_ignition_delay < t_end*1.e+3:
                print('For settings: Phi={:.1f}, p={:.0f}bar, T={:.0f}K the delays are: first {:.5f}ms, '
                        'main {:.5f}ms'.format(equivalence_ratio_run, reactorPressure_run / ct.one_atm,
                                               reactorTemperature, first_ignition_delay, main_ignition_delay))

            elif args.information_print is True and main_ignition_delay is 0:
                print('For settings: Phi={:.1f}, p={:.0f}bar, T={:.0f}K ignition will happen after the '
                      'monitored interval'.format(equivalence_ratio_run, reactorPressure_run / ct.one_atm,
                                                  reactorTemperature))

            elif args.information_print is True and main_ignition_delay is t_end*1.e+3*0.99:
                print('For settings: Phi={:.1f}, p={:.0f}bar, T={:.0f}K ignition happens shortly after the end'
                      ' of the interval {}ms'.format(equivalence_ratio_run, reactorPressure_run / ct.one_atm,
                                                     reactorTemperature, t_end*1.e+3))

        if save_delays is True:
            path_delay = '{}/delays.csv'.format(path_dir)
            delays = pd.DataFrame(delays)
            delays.columns = ['T', 'first', 'main']
            delays.to_csv(path_delay)