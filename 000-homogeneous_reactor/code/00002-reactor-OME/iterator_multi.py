#######################################################################################################################
# Iteration through different temperatures, pressures, ... of the homogeneous reactor
#######################################################################################################################

# Import packages
import argparse
import itertools
import numpy as np
import cantera as ct
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from Homogeneous_Reactor_Multi import homogeneous_reactor, init_process
from pre_process_fc import save_df
from pre_process_fc import create_path
from pre_process_fc import make_dir

# %% Collect arguments
parser = argparse.ArgumentParser(description="Run homogeneous reactor model")

parser.add_argument("-mech", "--mechanism_input", type=str, choices=['he', 'sun', 'cai'], default='cai',
                    help="chose reaction mechanism")

parser.add_argument("--phi_0", type=float, default=1.0,
                    help="chose staring phi of simulation")

parser.add_argument("--phi_end", type=float, default=1.5,
                    help="chose end phi of simulation")

parser.add_argument("--phi_step", type=float, default=0.5,
                    help="chose step size of phi of simulation")

parser.add_argument("--p_0", type=int, default=20,
                    help="chose staring pressure of simulation")

parser.add_argument("--p_end", type=int, default=40,
                    help="chose end pressure of simulation")

parser.add_argument("--p_step", type=int, default=10,
                    help="chose step size of pressure of simulation")

parser.add_argument("--pode", type=int, nargs='+', default=[3],
                    help="chose degree of polymerization")

parser.add_argument("-t_0", "--temperature_start", type=int, default=950,
                    help="chose staring temperature of simulation")

parser.add_argument("-t_end", "--temperature_end", type=int, default=1250,
                    help="chose end temperature of simulation")

parser.add_argument("-t_step", "--temperature_step", type=int, default=30,
                    help="chose step size temperature of simulation")

parser.add_argument("-nbr_run", "--number_run", type=str, default='000',
                    help="define a nbr to identify the started iterator run")

parser.add_argument("-inf_print", "--information_print", default=True, action='store_false',
                    help="chose if basic information are displayed")

parser.add_argument("--category", type=str, choices=['train', 'test', 'exp'], default='train',
                    help="chose if train or test data should be generated")

parser.add_argument("--O2", type=float, default=0.21,
                    help="chose O2 ratio in air")

parser.add_argument("--N2", type=float, default=0.79,
                    help="chose N2 ratio in air")

parser.add_argument("--NCPU", type=int, default=6,
                    help="chose nbr of available CPU cores")

parser.add_argument("-comp", "--comparison", default=False, action='store_true',
                    help="select PV if normal data or data for comparison with GRM")

args = parser.parse_args()
if args.information_print:
    print('\n{}\n'.format(args))

if args.comparison:
    print('\nPV special for comparison with GRM of Haspel et al., normal mode contains better PV!\n')

#%% Define end time and time step
if args.category == 'exp':
    t_end = 0.100
    t_step = 1.e-5
    # create an array for the different samples/ the ignition delays and decide if to save them
    save_samples = False
    save_delays = True
else:
    t_end = 0.013
    t_step = 1.e-6
    # create an array for the different samples/ the ignition delays and decide if to save them
    save_samples = True
    save_delays = True

# %% Load enthalpy of formation
path = Path(__file__).resolve()
path_h = path.parents[2] / 'data/00002-reactor-OME/enthalpies_of_formation.csv'
h0 = pd.read_csv(path_h)
h0_mass = h0[['h0_mass']]
h0_mass = h0_mass.to_numpy()

# %% Create to save files
if not ((save_delays is False) and (save_samples is False)):
    make_dir(args.mechanism_input, args.number_run, args.information_print)

if save_samples:
    typ = 'samples'
    samples, nn = save_df(typ, args.category, args.mechanism_input, args.number_run, args.temperature_end,
                          args.temperature_start, args.temperature_step, args.phi_end, args.phi_0, args.phi_step,
                          args.p_end, args.p_0, args.p_step, args.pode, size=19)

if save_delays:
    typ = 'delays'
    delays, n = save_df(typ, args.category, args.mechanism_input, args.number_run, args.temperature_end,
                        args.temperature_start, args.temperature_step, args.phi_end, args.phi_0, args.phi_step,
                        args.p_end, args.p_0, args.p_step, args.pode, size=6)

# %% Iterate between the parameter settings
for iii, pode_run in enumerate(args.pode):

    if args.mechanism_input == 'he':
        mechanism = np.array(['he_2018.xml', 'DMM' + str(pode_run)])
    elif args.mechanism_input == 'cai':
        mechanism = np.array(['cai_ome14_2019.xml', 'OME' + str(pode_run)])
    elif args.mechanism_input == 'sun':
        mechanism = np.array(['sun_2017.xml', 'DMM' + str(pode_run)])

    if pode_run == 1:
        mechanism[1] = 'CH3OCH2OCH3'
    elif args.pode == 4 and not mechanism[0] == 'cai_ome14_2019.xml':
        print('WARNING: This mechanism is not available for PODE4, will be downgraded to PODE3')
        mechanism[1] = 'DMM3'

    if args.information_print is True:
        print('the used mechanism is {} with fuel {}\n'.format(mechanism[0], mechanism[1]))

    for ii, equivalence_ratio_run in enumerate(np.arange(args.phi_0, args.phi_end + args.phi_step, args.phi_step)):
        # enumerate through different equivalence ratios
        if args.information_print is True:
            print('Equivalence ratio: {}'.format(equivalence_ratio_run))

        for reactorPressure_run in range(args.p_0, args.p_end + args.p_step, args.p_step):
            # enumerate through different pressures
            if args.information_print is True:
                print('\nReactor Pressure: {}'.format(reactorPressure_run))

            reactorPressure_run = np.array(reactorPressure_run) * ct.one_atm

            pool = mp.Pool(processes=args.NCPU, initializer=init_process, initargs=(mechanism, ))

            values = pool.map(homogeneous_reactor, zip(itertools.repeat(mechanism), itertools.repeat(equivalence_ratio_run),
                                                       itertools.repeat(reactorPressure_run),
                                                       np.arange(args.temperature_start, args.temperature_end +
                                                       args.temperature_step, args.temperature_step),
                                                       itertools.repeat(t_end), itertools.repeat(t_step),
                                                       itertools.repeat(pode_run), itertools.repeat(args.O2),
                                                       itertools.repeat(args.N2), itertools.repeat(args.comparison)))

            for i in range(len(values)):
                # separate the list of all temperatures into the single ones
                values_run = values[i]

                # separate list in samples, first IDT and main IDT
                samples_run = values_run[0]
                first_ignition_delay = values_run[1]
                main_ignition_delay = values_run[2]

                # saving ignition delays for the parameter setting
                if save_delays is True and 0 < main_ignition_delay < t_end * 1.e+3 * 0.99:
                    delays[n, :] = (pode_run, equivalence_ratio_run, reactorPressure_run, samples_run[0, 3],
                                    first_ignition_delay, main_ignition_delay)
                    n += 1
                elif save_delays is True:  # cancelling rows if ignition didn't happened
                    n_rows, _ = delays.shape
                    delays = delays[:(n_rows - 1), :]

                # combine dataframes of different reactorTemperatures
                if save_samples is True:
                    n_samples_run = len(samples_run)
                    samples[nn:(nn + n_samples_run), :] = samples_run
                    n_samples = len(samples)
                    samples = samples[:(n_samples - (25000 - n_samples_run)), :]
                    nn += n_samples_run

            pool.close()


if save_delays is True:
    print('Reactor solved! Saving delays ...')
    path_dir, _ = create_path(args.mechanism_input, args.number_run)
    path_delay = '{}/{}_{}_delays.csv'.format(path_dir, args.number_run, args.category)
    delays = pd.DataFrame(delays)
    delays.columns = ['pode', 'phi', 'P_0', 'T_0', 'first', 'main']
    delays.set_index(['pode', 'phi', 'P_0'])
    delays.to_csv(path_delay)

# save species development with the parameter setting
if save_samples is True:
    print('Delays saved! Saving samples ...')
    path_dir, _ = create_path(args.mechanism_input, args.number_run)
    #    path_dir = '/media/pascal/TOSHIBA EXT/BA'
    path_sample = '{}/{}_{}_samples.csv'.format(path_dir, args.number_run, args.category)
    samples = pd.DataFrame(samples)
    samples.columns = ['pode', 'phi', 'P_0', 'T_0', 'H', 'Z', 'time', 'PV', 'HRR', 'T', 'P', 'V', 'PODE',
                       'CO2', 'O2', 'CO', 'H2O', 'H2', 'CH2O']
    samples = samples.set_index(['pode', 'phi', 'P_0', 'T_0'])
    samples.to_csv(path_sample)

print('DONE!')
