### Implementation constant volume, fixed mass reactor

# Information source
# Cantera Website
# - https://cantera.org/science/reactors.html
# - https://cantera.github.io/docs/sphinx/html/cython/thermo.html
# - https://cantera.org/documentation/docs-2.4/sphinx/html/cython/zerodim.html
# - https://cantera.org/examples/python/reactors/reactor2.py.html (Example of two reactors with a piston and with heat
# loss to the environment

# %% Import Packages
import cantera as ct
import numpy as np

# Suppress warnings
ct.suppress_thermo_warnings()

# Define if inforamtion should be printed
information_print = True

# Create a global variable for pode_mulit
pode_multi = {}


# %% functions to calculate the mixture fraction variable
def beta(gas, components, weights):
    """
    calculate the coupling function

    :parameter
    :param gas:                             cantera solution object
    :param components:  - list of str -     list of element str
    :param weights:     - array-            array with weights for each element

    :returns
    :return beta:       - float -           value of coupling function
    """

    for i in range(len(components)):
        if i == 0:
            beta = weights[i] * gas.elemental_mole_fraction(components[i])
        else:
            beta = beta + weights[i] * gas.elemental_mole_fraction(components[i])

    return beta


def mixture_frac(pode, mechanism, O2, N2, equivalence_ratio, reactorPressure,
                 reactorTemperature):  # create the mixture fraction variable for the run
    """
    calc the mixture fraction variable

    :parameter
    :param pode:                            cantera solution object
    :param mechanism:           - array -   mechanism name and fuel species
    :param O2:                  - float -   O2 percentage in air mixture
    :param N2:                  - float -   N2 percentage in air mixture
    :param equivalence_ratio:   - float -   equivalence_ratio of mixture
    :param reactorPressure:     - int -     initial pressure of reactor
    :param reactorTemperature:  - int -     initial temperature of reactor

    :returns
    :return Z:                  - float -   mixture fraction variable
    """

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
    """
    function to initialize a cantera solution object as global variable so that in can be used in multiprocessinhg

    :parameter
    :param mechanism:               - array -           mechanism name and fuel species
    """
    mech = mechanism[0]
    pode_multi[mech] = ct.Solution(mech)
    pode_multi[mech].transport_model = 'Multi'


# %% Homogeneous reactor simulation
def homogeneous_reactor(args_reactor):
    """
    Constant volume and fixed mass homogeneous reactor model to solve the detailed mechanism in time and extract
    combustion properties

    :parameter (all included in args_reactor)
    :param mechanism:               - array -           mechanism name and fuel species
    :param equivalence_ratio:       - float -           equivalence_ratio of mixture
    :param reactorPressure:         - int -             initial pressure of reactor
    :param reactorTemperature:      - int -             initial temperature of reactor
    :param t_end:                   - float -           time until the combustion process should be solved
    :param t_step:                  - float -           time step to discretize samples
    :param pode_nbr:                - int -             degree of ploymerization of used fuel
    :param O2:                      - float -           O2 percentage in air mixture
    :param N2:                      - float -           N2 percentage in air mixture
    :param h0_mass:                 - pd dataframe -    enthalpy of formation of used fuel

    :returns:
    :return values:                 - np array -        array with initial conditions, thermodynamic properties and
                                                        species development
    :return first_ignition_delay:   - float -           first ignition delay of combustion
    :return main_ignition_delay:    - float -           main ignition delay of combustion
    """
    # get arguments
    mechanism, equivalence_ratio, reactorPressure, reactorTemperature, t_end, t_step, pode_nbr, O2, N2, comparison = \
        args_reactor

    pode = pode_multi[mechanism[0]]

    # calculate mixture fraction
    Z = mixture_frac(pode, mechanism, O2, N2, equivalence_ratio, reactorPressure, reactorTemperature)

    # Create Reactor
    pode.TP = reactorTemperature, reactorPressure
    pode.set_equivalence_ratio(equivalence_ratio, mechanism[1], 'O2:{} N2:{}'.format(O2, N2))

    # if information_print is True:
    #     print(pode())

    r1 = ct.IdealGasReactor(contents=pode, name='homogeneous_reactor')
    sim = ct.ReactorNet([r1])
    #    sim.atol = 1.e-14  # standard: 1e-15
    #    sim.rtol = 1.e-10  # standard: 1e-09
    sim.max_err_test_fails = 10

    #  Solution of reaction
    time = 0.0
    n_samples = 25000
    n = 0
    samples_after_ignition = 200
    stop_criterion = False

    values = np.zeros((n_samples, 19))

    # Parameters of PV
    PV_p = np.array(['H2O', 'CH2O', mechanism[1], 'CO2'])
    OME3_0 = r1.Y[pode.species_index(mechanism[1])]

    # cantera enthalpy is sum of abs enthalpy and the enthalpy generated by pressure increase (Poinsot Eq. 1.60)
    H = r1.thermo.enthalpy_mass

    while time < t_end:
        # calculate grad to define step size and stop_criterion
        if n <= 1:
            grad_PV = np.zeros((3))
            grad_T = np.zeros((3))
        else:
            grad_PV = np.gradient(values[:(n + 1), 7], values[:(n + 1), 6])
            grad_T = np.gradient(values[:(n + 1), 9])

        # gradient from 2 time steps earlier, because np.gradient would otherwise take zeros into account
        if np.abs(grad_PV[n - 2]) > 25000:
            time += t_step / 10000
        elif np.abs(grad_PV[n - 2]) > 10000:
            time += t_step / 1000
        elif np.abs(grad_PV[n - 2]) > 1000:
            time += t_step / 100
        elif np.abs(grad_PV[n - 2]) > 100:
            time += t_step / 10
        else:
            time += t_step

        # Initialize a break condition so that after the ignition, samples are not taken for an unnecessary long time
        if r1.thermo.T > 1.25 * reactorTemperature and grad_T[n - 2] < 1.e-7 and stop_criterion is False:
            t_end = time + samples_after_ignition * t_step
            stop_criterion = True

        # Calculate the reactor parameters for the point in time
        sim.advance(time)

        # Calculate the PV
        if comparison:
            PV = r1.Y[pode.species_index(PV_p[0])] * 0.25 + (- r1.Y[pode.species_index(PV_p[2])] + OME3_0) * 0.5 + \
                 r1.Y[pode.species_index(PV_p[3])] * 0.05
        else:
            PV = r1.Y[pode.species_index(PV_p[0])] * 0.5 + r1.Y[pode.species_index(PV_p[1])] * 0.5 + \
                 (- r1.Y[pode.species_index(PV_p[2])] + OME3_0) * 0.5 + r1.Y[pode.species_index(PV_p[3])] * 0.05

        HRR = - np.sum(r1.thermo.net_production_rates * r1.thermo.partial_molar_enthalpies / r1.mass)
        # Net production rates for each species. [kmol/m^3/s] for bulk phases or [kmol/m^2/s] for surface phases.
        # partial_molar_enthalpies: Array of species partial molar enthalpies[J / kmol]

        # Summarize all values to be saved in an array
        values[n] = (pode_nbr, equivalence_ratio, reactorPressure, reactorTemperature, H, Z, time, PV, HRR, r1.thermo.T,
                     r1.thermo.P, r1.volume, r1.Y[pode.species_index(mechanism[1])],
                     r1.Y[pode.species_index('CO2')], r1.Y[pode.species_index('O2')],
                     r1.Y[pode.species_index('CO')], r1.Y[pode.species_index('H2O')],
                     r1.Y[pode.species_index('H2')], r1.Y[pode.species_index('CH2O')])

        n += 1

        if n == n_samples and time < t_end:
            print('WARNING: maximum nbr of samples: {} taken and {} not reached'.format(n_samples, t_end))
            break

    values = values[:n, :]

    # ignition delay times
    from scipy.signal import find_peaks

    max_HRR = np.argmax(values[:, 8])
    peaks, _ = find_peaks(values[:, 8], prominence=values[max_HRR, 8] / 100)  # define minimum height

    if peaks.any() and values[max_HRR, 9] > (reactorTemperature * 1.15):
        first_ignition_delay = values[peaks[0], 6] * 1.e+3
        main_ignition_delay = values[max_HRR, 6] * 1.e+3

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
