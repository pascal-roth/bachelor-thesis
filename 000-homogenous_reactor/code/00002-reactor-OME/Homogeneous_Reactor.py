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


# %% Homogeneous reactor simulation
def homogeneous_reactor(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, t_end, t_step):
    #  Fuel mixture
    pome = ct.Solution(mechanism[0])
    pome.TP = reactorTemperature, reactorPressure
    pome.set_equivalence_ratio(equivalence_ratio, mechanism[1], 'O2:0.21 N2:0.79')

    if information_print is True:
        print(pome())

    # Create Reactor
    r1 = ct.Reactor(contents=pome, name='homogeneous_reactor')
    sim = ct.ReactorNet([r1])
    #    sim.atol = 1.e-14  # standard: 1e-15
    #    sim.rtol = 1.e-10  # standard: 1e-09
    sim.max_err_test_fails = 10

    if information_print is True:
        print('finished setup, begin solution...')

    #  Solution of reaction
    time = 0.0
    n_samples = 12500
    n = 0

    values = np.zeros((n_samples, 18))

    while time < t_end:
        if n == n_samples:
            print('WARNING: {} samples taken and {} not reached'.format(n_samples, t_end))
            break

        # calculate grad to define step size
        if n <= 1:
            grad_run = np.zeros((3))
        else:
            grad_run = np.gradient(values[:(n + 1), 1])

        #  gradient from 2 time steps earlier, because np.gradient would otherwise take zeros into account
        if grad_run[n - 2] > 1.e-6:
            time += t_step / 100
        else:
            time += t_step

        # Calculate the reactor parameters for the point in time
        sim.advance(time)

        PV = r1.Y[pome.species_index(PV_p[0])] / pome.molecular_weights[pome.species_index(PV_p[0])] + \
             r1.Y[pome.species_index(PV_p[1])] / pome.molecular_weights[pome.species_index(PV_p[1])] * 0.15 + \
             r1.Y[pome.species_index(PV_p[2])] / pome.molecular_weights[pome.species_index(PV_p[2])] * 1.5

        Q = - np.sum(r1.thermo.net_production_rates * r1.thermo.partial_molar_enthalpies)
        # Net production rates for each species. [kmol/m^3/s] for bulk phases or [kmol/m^2/s] for surface phases.
        # partial_molar_enthalpies: Array of species partial molar enthalpies[J / kmol]

        values[n] = (time, PV, equivalence_ratio, Q, r1.thermo.T, r1.thermo.P, r1.volume,
                     r1.Y[pome.species_index(mechanism[1])],
                     r1.Y[pome.species_index('CO2')], r1.Y[pome.species_index('O2')],
                     r1.Y[pome.species_index('CO')], r1.Y[pome.species_index('H2O')],
                     r1.Y[pome.species_index('OH')], r1.Y[pome.species_index('H2O2')],
                     r1.Y[pome.species_index('CH3')], r1.Y[pome.species_index('CH3O')],
                     r1.Y[pome.species_index('CH2O')], r1.Y[pome.species_index('C2H2')])

        n += 1

    values = values[:n, :]

    # ignition delay times
    from scipy.signal import find_peaks

    max_Q = np.argmax(values[:, 3])
    peaks, _ = find_peaks(values[:, 3], prominence=values[max_Q, 3] / 100)  # define minimum height

    if peaks.any():
        first_ignition_delay = values[peaks[0], 0] * 1.e+3
        main_ignition_delay = values[max_Q, 0] * 1.e+3

        if information_print is True:
            print('The first stage ignition delay is {:.3f} ms'.format(first_ignition_delay))
            print('The second/ main stage ignition delay: is {:.3f} ms'.format(main_ignition_delay))

    else:
        first_ignition_delay = 0
        main_ignition_delay = 0

        if information_print is True:
            print('No ignition delay')

    # Convert to panda array
    values = pd.DataFrame(values)
    values.columns = ['time', 'PV', 'phi', 'Q', 'T', 'P', 'V', 'PODE', 'CO2', 'O2', 'CO', 'H2O', 'OH', 'H2O2', 'CH3',
                      'CH3O', 'CH2O', 'C2H2']
    return values, first_ignition_delay, main_ignition_delay
