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
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt

# Suppress warnings
ct.suppress_thermo_warnings()

# Define if inforamtion should be printed
information_print = False

# Change the RPV calculation method
PV_p = np.array(['H2O', 'CO2', 'CH2O'])

if information_print is True:
    print('The parameters for the reaction progress variable are: {}'.format(PV_p))


mechanism_input = 'cai'
pode_run = 3
reactorTemperature = 740
N2 = 0.79
O2 = 0.21
category = 'train'
number_run = '002'
mechanism_input = 'cai'
reactorPressure_run = 20
equivalence_ratio_run = 1.0
information_print = True
t_end = 0.010
t_step = 1.e-6

if mechanism_input == 'he':
    mechanism = np.array(['he_2018.xml', 'DMM' + str(pode_run)])
elif mechanism_input == 'cai':
    mechanism = np.array(['cai_ome14_2019.xml', 'OME' + str(pode_run)])
elif mechanism_input == 'sun':
    mechanism = np.array(['sun_2017.xml', 'DMM' + str(pode_run)])

#%% functions to calculate the mixture fraction variable
def beta(gas, components, weights):
    for i in range(len(components)):
        if i == 0:
            beta = weights[i] * gas.elemental_mole_fraction(components[i])
        else:
            beta = beta + weights[i] * gas.elemental_mole_fraction(components[i])

    return beta


def mixture_frac(pode, mechanism, O2, N2, equivalence_ratio, r1):     # create the mixture fraction variable for the run

    Z_components = ['C', 'O', 'H']
    Z_weights = [2, -1, 0.5]

    pode.X = {'O2': O2, 'N2': N2}
    beta_oxidizer = beta(pode, Z_components, Z_weights)

    pode.X = {mechanism[1]: 1.0}
    beta_fuel = beta(pode, Z_components, Z_weights)

    pode.set_equivalence_ratio(equivalence_ratio, mechanism[1], 'O2:{} N2:{}'.format(O2, N2))
    beta_run = beta(r1.thermo, Z_components, Z_weights)

    Z = (beta_run - beta_oxidizer) / (beta_fuel - beta_oxidizer)

    return Z


# %% Homogeneous reactor simulation
def homogeneous_reactor(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, t_end, t_step, pode_nbr, O2, N2):
    #  Fuel mixture
    pode = ct.Solution(mechanism[0])
    pode.TP = reactorTemperature, reactorPressure

    # calculate mixture fraction

    # Create Reactor
    pode.set_equivalence_ratio(equivalence_ratio, mechanism[1], 'O2:{} N2:{}'.format(O2, N2))

    if information_print is True:
        print(pode())

    r1 = ct.IdealGasReactor(contents=pode, name='homogeneous_reactor')
    sim = ct.ReactorNet([r1])
    #    sim.atol = 1.e-14  # standard: 1e-15
    #    sim.rtol = 1.e-10  # standard: 1e-09
    sim.max_err_test_fails = 50

    if information_print is True:
        print('finished setup, begin solution...')

    #  Solution of reaction
    time = 0.0
    n_samples = 12500
    n = 0
    samples_after_ignition = 300
    stop_criterion = False

    values = np.zeros((n_samples, 20))

    while time < t_end:
        # calculate grad to define step size and stop_criterion
        if n <= 1:
            grad_PV = np.zeros((3))
            grad_T = np.zeros((3))
        else:
            grad_PV = np.gradient(values[:(n + 1), 8])
            grad_T = np.gradient(values[:(n + 1), 10])

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
        PV = r1.Y[pode.species_index(PV_p[0])] / pode.molecular_weights[pode.species_index(PV_p[0])] + \
             r1.Y[pode.species_index(PV_p[1])] / pode.molecular_weights[pode.species_index(PV_p[1])] * 0.15 + \
             r1.Y[pode.species_index(PV_p[2])] / pode.molecular_weights[pode.species_index(PV_p[2])] * 1.5

        Q = - np.sum(r1.thermo.net_production_rates * r1.thermo.partial_molar_enthalpies)
        # Net production rates for each species. [kmol/m^3/s] for bulk phases or [kmol/m^2/s] for surface phases.
        # partial_molar_enthalpies: Array of species partial molar enthalpies[J / kmol]

        # Calculate the internal energy as characterization of the thermodynamical state
        r1.thermo.basis = 'mass'
        if n == 0:
            s_0 = r1.thermo.s
            g_0 = r1.thermo.g
            v_0 = r1.thermo.v

            if information_print is True:
                print('The initial conditions for the temperature of {}K are: \n'
                      'entropy s: {:.3f} [J/kgK] free gibbs energy: {:.3f} [J/kg] volume: {:.3f} [m³/kg]'.format(
                       reactorTemperature, s_0, g_0, v_0))

        state = r1.get_state()
        internal_energy = state[2]

        Z = mixture_frac(pode, mechanism, O2, N2, equivalence_ratio, r1)

        # Summarize all values to be saved in an array
        values[n] = (pode_nbr, equivalence_ratio, reactorPressure, reactorTemperature, internal_energy, r1.thermo.h, Z,
                     time, PV, Q, r1.thermo.T, r1.thermo.P, r1.volume, r1.Y[pode.species_index(mechanism[1])],
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

    max_Q = np.argmax(values[:, 9])
    peaks, _ = find_peaks(values[:, 9], prominence=values[max_Q, 9] / 100)  # define minimum height

    if peaks.any() and values[max_Q, 10] > (reactorTemperature * 1.15):
        first_ignition_delay = values[peaks[0], 7] * 1.e+3
        main_ignition_delay = values[max_Q, 7] * 1.e+3

        if information_print is True:
            print('The first stage ignition delay is {:.3f} ms'.format(first_ignition_delay))
            print('The second/ main stage ignition delay: is {:.3f} ms'.format(main_ignition_delay))

    else:
        first_ignition_delay = 0
        main_ignition_delay = 0

        if information_print is True:
            print('No ignition delay')

    return values, first_ignition_delay, main_ignition_delay, s_0, g_0, v_0


values, first_ignition_delay, main_ignition_delay, s_0, g_0, v_0 = homogeneous_reactor \
    (mechanism, equivalence_ratio_run, reactorPressure_run, reactorTemperature, t_end, t_step, pode_run,
     O2, N2)

#%%
plt.plot(values[:, 7] * 1.e3, values[:, 6], 'r-', label='Mixture Fraction')
plt.legend()
plt.title('PODE{} $\\Phi$={:.1f} p={}bar $T_0$={:.0f}K'.format(pode_run, equivalence_ratio_run,
                                                               reactorPressure_run, reactorTemperature))
plt.xlabel('time [ms]')
plt.ylabel('Z')
plt.show()