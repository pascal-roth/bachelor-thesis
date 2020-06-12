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
from pre_process_fc import save_df
from pre_process_fc import create_path
from pre_process_fc import make_dir

# %% Collect arguments
parser = argparse.ArgumentParser(description="Run homogeneous reactor model")

parser.add_argument("-mech", "--mechanism_input", type=str, choices=['he', 'sun', 'cai'], default='cai',
                    help="chose reaction mechanism")

parser.add_argument("--phi_0", type=float, default=0.5,
                    help="chose staring phi of simulation")

parser.add_argument("--phi_end", type=float, default=1.5,
                    help="chose end phi of simulation")

parser.add_argument("--phi_step", type=float, default=0.5,
                    help="chose step size of phi of simulation")

parser.add_argument("--p_0", type=int, default=10,
                    help="chose staring pressure of simulation")

parser.add_argument("--p_end", type=int, default=40,
                    help="chose end pressure of simulation")

parser.add_argument("--p_step", type=int, default=10,
                    help="chose step size of pressure of simulation")

parser.add_argument("--pode", type=int, nargs='+', default=3,
                    help="chose degree of polymerization")

parser.add_argument("-t_0", "--temperature_start", type=int, default=650,
                    help="chose staring temperature of simulation")

parser.add_argument("-t_end", "--temperature_end", type=int, default=1250,
                    help="chose end temperature of simulation")

parser.add_argument("-t_step", "--temperature_step", type=int, default=30,
                    help="chose step size temperature of simulation")

parser.add_argument("-nbr_run", "--number_run", type=str, default='002',
                    help="define a nbr to identify the started iterator run")

parser.add_argument("-inf_print", "--information_print", default=True, action='store_false',
                    help="chose if basic information are displayed")

parser.add_argument("--category", type=str, choices=['train', 'test', 'exp'], default='train',
                    help="chose if train or test data should be generated")

parser.add_argument("--O2", type=float, default=0.21,
                    help="chose O2 ratio in air")

parser.add_argument("--N2", type=float, default=0.79,
                    help="chose N2 ratio in air")

parser.add_argument("--NCPU", type=int, default=20,
                    help="chose nbr of available CPU cores")

args = parser.parse_args()
if args.information_print is True:
    print('\n{}\n'.format(args))


# Suppress warnings
ct.suppress_thermo_warnings()

# Define if inforamtion should be printed
information_print = True

# Change the RPV calculation method
PV_p = np.array(['H2O', 'CO2', 'CH2O'])

# Create a global variable for pode_mulit
pode_multi = {}


#%% functions to calculate the mixture fraction variable
def beta(gas, components, weights):
    for i in range(len(components)):
        if i == 0:
            beta = weights[i] * gas.elemental_mole_fraction(components[i])
        else:
            beta = beta + weights[i] * gas.elemental_mole_fraction(components[i])

    return beta


def mixture_frac(pode, mechanism, O2, N2, equivalence_ratio, reactorPressure, reactorTemperature):     # create the mixture fraction variable for the run

    Z_components = ['C', 'O', 'H']
    Z_weights = [2, -1, 0.5]

    pode.TPX = reactorTemperature, reactorPressure, 'O2:{} N2:{}'.format(O2, N2)
    beta_oxidizer = beta(pode, Z_components, Z_weights)

    pode.TPX = reactorTemperature, reactorPressure, '{}:1.0'.format(mechanism[1])
    beta_fuel = beta(pode, Z_components, Z_weights)

    pode.TP = reactorTemperature, reactorPressure
    pode.set_equivalence_ratio(equivalence_ratio, mechanism[1], 'O2:{} N2:{}'.format(O2, N2))
    beta_run = beta(pode, Z_components, Z_weights)

    Z = (beta_run - beta_oxidizer) / (beta_fuel - beta_oxidizer)

    return Z


def init_process(mechanism):
    mech = mechanism[0]
    pode_multi[mech] = ct.Solution(mech)
    pode_multi[mech].transport_model = 'Multi'


# %% Homogeneous reactor simulation
def homogeneous_reactor(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, t_end, t_step,
                        pode_nbr, O2, N2):

    # calculate mixture fraction
    Z = mixture_frac(pode_multi[mechanism[0]], mechanism, O2, N2, equivalence_ratio, reactorPressure, reactorTemperature)
#    Z = 0
    # Create Reactor
    pode_multi[mechanism[0]].TP = reactorTemperature, reactorPressure
    pode_multi[mechanism[0]].set_equivalence_ratio(equivalence_ratio, mechanism[1], 'O2:{} N2:{}'.format(O2, N2))

    # if information_print is True:
    #     print(pode())

    r1 = ct.IdealGasReactor(contents=pode_multi[mechanism[0]], name='homogeneous_reactor')
    sim = ct.ReactorNet([r1])
    #    sim.atol = 1.e-14  # standard: 1e-15
    #    sim.rtol = 1.e-10  # standard: 1e-09
    sim.max_err_test_fails = 10

    #  Solution of reaction
    time = 0.0
    n_samples = 12500
    n = 0
    samples_after_ignition = 300
    stop_criterion = False

    values = np.zeros((n_samples, 19))

    while time < t_end:
        # calculate grad to define step size and stop_criterion
        if n <= 1:
            grad_PV = np.zeros((3))
            grad_T = np.zeros((3))
        else:
            grad_PV = np.gradient(values[:(n + 1), 7])
            grad_T = np.gradient(values[:(n + 1), 9])

        #  gradient from 2 time steps earlier, because np.gradient would otherwise take zeros into account
        if grad_PV[n - 2] > 1.e-6:
            time += t_step / 100
        else:
            time += t_step

        # Initialize a break condition so that after the ignition, samples are not taken for an unnecessary long time
        if r1.thermo.T > 1.25 * reactorTemperature and grad_T[n - 2] < 1.e-7 and stop_criterion is False:
            t_end = time + samples_after_ignition * t_step
            stop_criterion = True

        # Calculate the reactor parameters for the point in time
        sim.advance(time)

        # Calculate the PV
        PV = r1.Y[pode_multi[mechanism[0]].species_index(PV_p[0])] / pode_multi[mechanism[0]].molecular_weights[pode_multi[mechanism[0]].species_index(PV_p[0])] + \
             r1.Y[pode_multi[mechanism[0]].species_index(PV_p[1])] / pode_multi[mechanism[0]].molecular_weights[pode_multi[mechanism[0]].species_index(PV_p[1])] * 0.15 + \
             r1.Y[pode_multi[mechanism[0]].species_index(PV_p[2])] / pode_multi[mechanism[0]].molecular_weights[pode_multi[mechanism[0]].species_index(PV_p[2])] * 1.5

        Q = - np.sum(r1.thermo.net_production_rates * r1.thermo.partial_molar_enthalpies)
        # Net production rates for each species. [kmol/m^3/s] for bulk phases or [kmol/m^2/s] for surface phases.
        # partial_molar_enthalpies: Array of species partial molar enthalpies[J / kmol]

        # Summarize all values to be saved in an array
        values[n] = (pode_nbr, equivalence_ratio, reactorPressure, reactorTemperature, r1.thermo.enthalpy_mass, Z,
                     time, PV, Q, r1.thermo.T, r1.thermo.P, r1.volume, r1.Y[pode_multi[mechanism[0]].species_index(mechanism[1])],
                     r1.Y[pode_multi[mechanism[0]].species_index('CO2')], r1.Y[pode_multi[mechanism[0]].species_index('O2')],
                     r1.Y[pode_multi[mechanism[0]].species_index('CO')], r1.Y[pode_multi[mechanism[0]].species_index('H2O')],
                     r1.Y[pode_multi[mechanism[0]].species_index('H2')], r1.Y[pode_multi[mechanism[0]].species_index('CH2O')])

        n += 1

        if n == n_samples and time < t_end:
            print('WARNING: maximum nbr of samples: {} taken and {} not reached'.format(n_samples, t_end))
            break

    values = values[:n, :]

    # ignition delay times
    from scipy.signal import find_peaks

    max_Q = np.argmax(values[:, 8])
    peaks, _ = find_peaks(values[:, 8], prominence=values[max_Q, 8] / 100)  # define minimum height

    if peaks.any() and values[max_Q, 9] > (reactorTemperature * 1.15):
        first_ignition_delay = values[peaks[0], 6] * 1.e+3
        main_ignition_delay = values[max_Q, 6] * 1.e+3

    else:
        first_ignition_delay = 0
        main_ignition_delay = 0

    # print information about parameter setting and ignition
    if information_print is True and 0 < main_ignition_delay < t_end * 1.e+3:
        print('For settings: Phi={:2.2e}, p={:.0f}bar, T={:2.2e}K the delays are: first {:6.5e}ms, '
              'main {:6.5e}ms'.format(equivalence_ratio, reactorPressure / ct.one_atm,
                                     reactorTemperature, first_ignition_delay, main_ignition_delay))

    elif information_print is True and main_ignition_delay is 0:
        print('For settings: Phi={:2.2e}, p={:.0f}bar, T={:2.2e}K ignition will happen after the '
              'monitored interval'.format(equivalence_ratio, reactorPressure / ct.one_atm,
                                          reactorTemperature))

    elif information_print is True and main_ignition_delay is t_end * 1.e+3 * 0.99:
        print('For settings: Phi={:2.2e}, p={:.0f}bar, T={:2.2e}K \tignition happens shortly after the end'
              ' of the interval {}ms'.format(equivalence_ratio, reactorPressure / ct.one_atm,
                                             reactorTemperature, t_end * 1.e+3))

    return values, first_ignition_delay, main_ignition_delay


if __name__ == '__main__':
    #%% Define end time and time step
    if args.category == 'exp':
        t_end = 0.100
        t_step = 1.e-5
        # create an array for the different samples/ the ignition delays and decide if to save them
        save_samples = False
        save_delays = True
    else:
        t_end = 0.010
        t_step = 1.e-6
        # create an array for the different samples/ the ignition delays and decide if to save them
        save_samples = True
        save_delays = True

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

                values = [pool.apply(homogeneous_reactor, args=(mechanism, equivalence_ratio_run, reactorPressure_run,
                                                                reactorTemperature, t_end, t_step, pode_run, args.O2,
                                                                args.N2)) for
                          reactorTemperature in range(args.temperature_start, args.temperature_end +
                                                      args.temperature_step, args.temperature_step)]

                # values = pool.map(homogeneous_reactor, zip(itertools.repeat(mechanism), itertools.repeat(equivalence_ratio_run),
                #                                             itertools.repeat(reactorPressure_run),
                #                                             np.arange(args.temperature_start, args.temperature_end +
                #                                             args.temperature_step, args.temperature_step),
                #                                             itertools.repeat(t_end), itertools.repeat(t_step),
                #                                             itertools.repeat(pode_run), itertools.repeat(args.O2),
                #                                             itertools.repeat(args.N2)))

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
                        samples = samples[:(n_samples - (12500 - n_samples_run)), :]
                        nn += n_samples_run

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
        #    path_dir = '/media/pascal/TOSHIBA EXT/BA'
        path_sample = '{}/{}_{}_samples.csv'.format(path_dir, args.number_run, args.category)
        samples = pd.DataFrame(samples)
        samples.columns = ['pode', 'phi', 'P_0', 'T_0', 'H', 'Z', 'time', 'PV', 'Q', 'T', 'P', 'V', 'PODE',
                           'CO2', 'O2', 'CO', 'H2O', 'H2', 'CH2O']
        samples = samples.set_index(['pode', 'phi', 'P_0', 'T_0'])
        samples.to_csv(path_sample)
