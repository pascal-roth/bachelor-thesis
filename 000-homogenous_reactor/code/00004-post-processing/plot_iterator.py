#######################################################################################################################
# Iteration through different temperatures, pressures, ... of the homogeneous reactor
#######################################################################################################################

# Import packages
import argparse
import numpy as np
import cantera as ct


# %% Collect arguments
parser = argparse.ArgumentParser(description="Run homogeneous reactor model")

parser.add_argument("-plt", "--plot", type=str, choices=['ign_delays', 'thermo', 'species', 'HR', 'PV'], default='HR',
                    help="chose which plot to create")

parser.add_argument("-mech", "--mechanism_input", type=str, choices=['he', 'sun', 'cai', 'all'], default='he',
                    help="chose reaction mechanism")

parser.add_argument("--pode", type=int, choices=[1, 2, 3, 4], default=3,
                    help="chose degree of polymerization")

parser.add_argument("-phi", "--equivalence_ratio", type=float, choices=[0.5, 1.0, 1.5, 0.0], default='1.0',
                    help="chose equivalence ratio")

parser.add_argument("-p", "--pressure", type=int, choices=[10, 20, 40, 0], default=20,
                    help="chose reactor pressure")

parser.add_argument("-x", "--scale", type=str, choices=['PV', 'time'], default='PV',
                    help="chose if plotted over time or PV")

parser.add_argument("-t_0", "--temperature", type=int, default='950',
                    help="chose starting temperature")

args = parser.parse_args()
print(args)

# %% Rename mechanism
mechanism_all = np.array([['he_2018.xml'], ['cai_ome14_2019.xml'], ['sun_2017.xml']])

if args.mechanism_input == 'he':
    mechanism = mechanism_all[0]
elif args.mechanism_input == 'cai':
    mechanism = mechanism_all[1]
elif args.mechanism_input == 'sun':
    mechanism = mechanism_all[2]

# %% Call plot function

if args.plot is 'ign_delay':
    from plot_ign_delay import plot_delays
    plot_delays(args.pode, args.equivalence_ratio, args.pressure)
elif args.plot == 'thermo':
    from plot_process import plot_thermo
    plot_thermo(mechanism, args.equivalence_ratio, args.pressure, args.temperature, args.scale, args.pode)
elif args.plot == 'species':
    from plot_process import plot_species
    plot_species(mechanism, args.equivalence_ratio, args.pressure, args.temperature, args.scale, args.pode)
elif args.plot == 'HR':
    from plot_process import plot_HR
    plot_HR(mechanism, args.equivalence_ratio, args.pressure, args.temperature, args.scale, args.pode)
elif args.plot == 'PV':
    from plot_process import plot_PV
    plot_PV(mechanism, args.equivalence_ratio, args.pressure, args.temperature, args.pode)