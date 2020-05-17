#######################################################################################################################
# Iteration through different temperatures, pressures, ... of the homogeneous reactor
#######################################################################################################################

# Import packages
import argparse
import numpy as np
import cantera as ct
import pandas as pd
from Homogeneous_Reactor import homogeneous_reactor
from pre_process_fc import save_df
from pre_process_fc import create_path
from pre_process_fc import make_dir


# %% Collect arguments
parser = argparse.ArgumentParser(description="Run homogeneous reactor model")

parser.add_argument("-mech", "--mechanism_input", type=str, choices=['he', 'sun', 'cai'], default='he',
                    help="chose reaction mechanism")

parser.add_argument("-phi", "--equivalence_ratio", nargs='+', type=float, default='1.0',
                    help="chose equivalence ratio")

parser.add_argument("-p", "--pressure", nargs='+', type=int, default=20,
                    help="chose reactor pressure")

parser.add_argument("--pode", type=int, nargs='+', default=3,
                    help="chose degree of polymerization")

parser.add_argument("-t_0", "--temperature_start", type=int, default=650,
                    help="chose staring temperature of simulation")

parser.add_argument("-t_end", "--temperature_end", type=int, default=1250,
                    help="chose staring temperature of simulation")

parser.add_argument("-t_step", "--temperature_step", type=int, default=15,
                    help="chose staring temperature of simulation")

parser.add_argument("-nbr_run", "--number_run", type=str, default='001',
                    help="define a nbr to identify the started iterator run")

parser.add_argument("-inf_print", "--information_print", default=True, action='store_false',
                    help="chose if basic information are displayed")

parser.add_argument("--category", type=str, choices=['train', 'test', 'exp'], default='train',
                    help="chose if train or test data should be generated")

parser.add_argument("--O2", type=float,  default=0.21,
                    help="chose O2 ratio in air")

parser.add_argument("--N2", type=float,  default=0.79,
                    help="chose N2 ratio in air")

args = parser.parse_args()
if args.information_print is True:
    print('\n{}\n'.format(args))

# %% Rename arguments
reactorTemperature_start = args.temperature_start
reactorTemperature_end = args.temperature_end
reactorTemperature_step = args.temperature_step

# Define end time and time step
t_end = 0.010

# create an array for the different samples/ the ignition delays and decide if to save them
save_samples = True
save_delays = True
save_initials = True

#%% Create to save files
if not ((save_delays is False) and (save_samples is False) and (save_initials is False)):
    make_dir(args.mechanism_input, args.number_run, args.information_print)

if save_samples is True:
    typ = 'samples'
    size = 18
    samples, nnn = save_df(typ, args.category, args.mechanism_input, args.number_run, reactorTemperature_end,
                           reactorTemperature_start, reactorTemperature_step, args.equivalence_ratio, args.pressure,
                           args.pode, size)

if save_delays is True:
    typ = 'delays'
    size = 6
    delays, n = save_df(typ, args.category, args.mechanism_input, args.number_run, reactorTemperature_end,
                        reactorTemperature_start, reactorTemperature_step, args.equivalence_ratio, args.pressure,
                        args.pode, size)

if save_initials is True:
    typ = 'initials'
    size = 7
    initials, nn = save_df(typ, args.category, args.mechanism_input, args.number_run, reactorTemperature_end,
                           reactorTemperature_start, reactorTemperature_step, args.equivalence_ratio, args.pressure,
                           args.pode, size)


# %% Iterate between the parameter settings
for iii, pode_run in enumerate(args.pode):

    if args.mechanism_input == 'he':
        mechanism = np.array(['he_2018.xml', 'DMM' + str(pode_run)])
        t_step = 1.e-6
    elif args.mechanism_input == 'cai':
        mechanism = np.array(['cai_ome14_2019.xml', 'OME' + str(pode_run)])
        t_step = 1.e-6
    elif args.mechanism_input == 'sun':
        mechanism = np.array(['sun_2017.xml', 'DMM' + str(pode_run)])
        t_step = 1.e-6

    if pode_run == 1:
        mechanism[1] = 'CH3OCH2OCH3'
    elif args.pode == 4 and not mechanism[0] == 'cai_ome14_2019.xml':
        print('WARNING: This mechanism is not available for PODE4, will be downgraded to PODE3')
        mechanism[1] = 'DMM3'

    if args.information_print is True:
        print('the used mechanism is {} with fuel {}\n'.format(mechanism[0], mechanism[1]))

    for ii, equivalence_ratio_run in enumerate(args.equivalence_ratio):  # enumerate through different equivalence ratios
        if args.information_print is True:
            print('Equivalence ratio: {}'.format(equivalence_ratio_run))

        for i, reactorPressure_run in enumerate(args.pressure):  # enumerate through different pressures
            if args.information_print is True:
                print('\nReactor Pressure: {}'.format(reactorPressure_run))

            reactorPressure_run = np.array(reactorPressure_run) * ct.one_atm

            for reactorTemperature in range(reactorTemperature_start, reactorTemperature_end, reactorTemperature_step):
                # start homogeneous reactor model with defined settings
                # values vector: ['time', 'PV', 'phi', 'Q', 'T', 'P', 'V', 'U', 'PODE', 'CO2', 'O2', 'CO', 'H2O', 'H2',
                # 'CH2O']
                values, first_ignition_delay, main_ignition_delay, s_0, g_0, v_0 = homogeneous_reactor\
                    (mechanism, equivalence_ratio_run, reactorPressure_run, reactorTemperature, t_end, t_step, pode_run,
                     args.O2, args.N2)


                # saving ignition delays for the parameter setting
                if save_delays is True and 0 < main_ignition_delay < t_end*1.e+3*0.99:
                    delays[n, :] = (pode_run, equivalence_ratio_run, reactorPressure_run, reactorTemperature,
                                    first_ignition_delay, main_ignition_delay)
                    n += 1
                elif save_delays is True:  # cancelling rows if ignition didn't happened
                    n_rows, _ = delays.shape
                    delays = delays[:(n_rows - 1), :]

                # combine dataframes of different reactorTemperatures
                if save_samples is True:
                    n_samples_run = len(values)
                    samples[nnn:(nnn+n_samples_run), :] = values
                    n_samples = len(samples)
                    samples = samples[:(n_samples-(12500-n_samples_run)), :]
                    nnn += n_samples_run

                # initial values to calculate the starting temperature of the reactor
                if save_initials is True:
                    initials[nn, :] = [pode_run, equivalence_ratio_run, reactorPressure_run, reactorTemperature, s_0,
                                       g_0, v_0]
                    nn += 1

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
    path_dir, _ = create_path(args.mechanism_input, args.number_run)
    path_delay = '{}/{}_{}_delays.csv'.format(path_dir, args.number_run, args.category)
    delays = pd.DataFrame(delays)
    delays.columns = ['pode', 'phi', 'P_0', 'T_0', 'first', 'main']
    delays.set_index(['pode', 'phi', 'P_0'])
    delays.to_csv(path_delay)

# save species development with the parameter setting
if save_samples is True:
    path_dir, _ = create_path(args.mechanism_input, args.number_run)
    path_sample = '{}/{}_{}_samples.csv'.format(path_dir, args.number_run, args.category)
    samples = pd.DataFrame(samples)
    samples.columns = ['pode', 'phi', 'P_0', 'T_0', 'time', 'PV', 'Q', 'T', 'P', 'V', 'U', 'PODE', 'CO2', 'O2', 'CO',
                       'H2O', 'H2', 'CH2O']
    samples = samples.set_index(['pode', 'phi', 'P_0', 'T_0'])
    samples.to_csv(path_sample)

if save_initials is True:
    path_dir, _ = create_path(args.mechanism_input, args.number_run)
    path_initials = '{}/{}_{}_initials.csv'.format(path_dir, args.number_run, args.category)
    initials = pd.DataFrame(initials)
    initials.columns = ['pode', 'phi', 'P_0', 'T_0', 's', 'g', 'v']
    initials.set_index(['pode', 'phi', 'P_0'])
    initials.to_csv(path_initials)
