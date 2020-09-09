#!/home/pascal/anaconda3/envs/BA/bin/python

#######################################################################################################################
# Iteration through different temperatures, pressures, ... of the homogeneous reactor
#######################################################################################################################

# Import packages
import argparse
import numpy as np
import cantera as ct


# %% Collect arguments
parser = argparse.ArgumentParser(description="Run homogeneous reactor model")

parser.add_argument("-plt", "--plot", type=str, choices=['ign_delay', 'thermo', 'species', 'HRR', 'PV', 'time_scale',
                                                         'T+HRR', 'T+P'],
                    default='PV', help="chose which plot to create")

parser.add_argument("-mech", "--mechanism_input", type=str, choices=['he', 'sun', 'cai', 'all'], default='cai',
                    help="chose reaction mechanism")

parser.add_argument("--pode", type=int, choices=[1, 2, 3, 4], default=3,
                    help="chose degree of polymerization")

parser.add_argument("-phi", "--equivalence_ratio", type=float, default='1.5',
                    help="chose equivalence ratio")

parser.add_argument("-p", "--pressure", type=int, default=40,
                    help="chose reactor pressure")

parser.add_argument("-x", "--scale", type=str, choices=['PV', 'time'], default='time',
                    help="chose if plotted over time or PV")

parser.add_argument("-t", "--temperature", type=int, default='950',
                    help="chose starting temperature")

parser.add_argument("-nbr_run", "--number_run", type=str, default='000',
                    help="define a nbr to identify the started iterator run")

parser.add_argument("--category", type=str, choices=['train', 'test', 'exp'], default='train',
                    help="chose if train or test data should be generated")

args = parser.parse_args()
print('\n{}\n'.format(args))

# %% Rename mechanism
mechanism_all = np.array([['he_2018.xml'], ['cai_ome14_2019.xml'], ['sun_2017.xml']])

if args.mechanism_input == 'he':
    mechanism = mechanism_all[0]
elif args.mechanism_input == 'cai':
    mechanism = mechanism_all[1]
elif args.mechanism_input == 'sun':
    mechanism = mechanism_all[2]
else:
    mechanism = mechanism_all

# %% Call plot function

if args.plot == 'ign_delay':
    from plot_ign_delay import plot_delays
    plot_delays(mechanism, args.pode, args.equivalence_ratio, args.pressure, args.number_run, args.category)
elif args.plot == 'thermo':
    from plot_process import plot_thermo
    plot_thermo(mechanism, args.equivalence_ratio, args.pressure, args.temperature, args.scale, args.pode,
                args.number_run, args.category)
elif args.plot == 'species':
    from plot_process import plot_species
    plot_species(mechanism, args.equivalence_ratio, args.pressure, args.temperature, args.scale, args.pode,
                 args.number_run, args.category)
elif args.plot == 'HRR':
    from plot_process import plot_HRR
    plot_HRR(mechanism, args.equivalence_ratio, args.pressure, args.temperature, args.scale, args.pode,
             args.number_run, args.category)
elif args.plot == 'PV':
    from plot_process import plot_PV
    plot_PV(mechanism, args.equivalence_ratio, args.pressure, args.temperature, args.pode, args.number_run,
            args.category)
elif args.plot == 'time_scale':
    from plot_process import plot_time_scale
    plot_time_scale(mechanism, args.equivalence_ratio, args.pressure, args.temperature, args.pode, args.number_run,
                    args.category)
elif args.plot == 'T+HRR':
    from plot_process import plot_T_and_HRR
    plot_T_and_HRR(mechanism, args.equivalence_ratio, args.pressure, args.temperature, args.scale, args.pode,
                   args.number_run, args.category)
elif args.plot == 'T+P':
    from plot_process import plot_T_and_P
    plot_T_and_P(mechanism, args.equivalence_ratio, args.pressure, args.temperature, args.scale, args.pode,
                 args.number_run, args.category)