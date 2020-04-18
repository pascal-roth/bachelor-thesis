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
import os
import csv
import matplotlib.pyplot as plt

# Suppress warnings
ct.suppress_thermo_warnings()

# Define if inforamtion should be printed
information_print = False

# Define if results should be saved in csv data
csv_save = False

# Change the RPV calculation method
RPV_param = np.array(['O', 'C', 'CO2', 'CO'])

if information_print is True:
    print('The parameters for the reaction progress variable are: {}'.format(RPV_param))


# %% Homogeneous reactor simulation
def homogeneous_reactor(mechanism, equivalence_ratio, reactorPressure, reactorTemperature, t_end, t_step):
    #  Fuel mixture
    pome = ct.Solution(mechanism[0])
    pome.TP = reactorTemperature, reactorPressure
    pome.set_equivalence_ratio(equivalence_ratio, mechanism[1], 'O2:0.21 N2:0.79')
    nbr_species = pome.n_species

    if information_print is True:
        print(pome())

    # Create Reactor
    r1 = ct.Reactor(contents=pome, name='homogeneous_reactor')
    sim = ct.ReactorNet([r1])

    if information_print is True:
        print('finished setup, begin solution...')

    #  Solution of reaction
    time = 0.0
    n_steps = int(t_end / t_step)

    values = np.zeros((n_steps, 13))
    RPV = np.zeros(n_steps)
    production_rate = np.zeros((n_steps, nbr_species))
    molar_enthalpies = np.zeros((n_steps, nbr_species))

    for n in range(n_steps):
        time += t_step
        sim.advance(time)

        production_rate[n, :] = r1.thermo.net_production_rates
        molar_enthalpies[n, :] = r1.thermo.partial_molar_enthalpies

        values[n] = (time, equivalence_ratio, 0, r1.thermo.T, r1.thermo.P, r1.volume, r1.Y[pome.species_index(mechanism[1])],
                     r1.Y[pome.species_index('CO2')], r1.Y[pome.species_index('O2')],
                     r1.Y[pome.species_index('CO')], r1.Y[pome.species_index('H2O')],
                     r1.Y[pome.species_index('OH')], r1.Y[pome.species_index('H2O2')])

        # Calculate tabulated chemistry variables
        # In the literature two different definitions (with and without dividing through the molecular mass
#        RPV[n] = r1.Y[pome.species_index(RPV_param[0])] / pome.molecular_weights[pome.species_index(RPV_param[0])] + \
#                 r1.Y[pome.species_index(RPV_param[1])] / pome.molecular_weights[pome.species_index(RPV_param[1])] + \
#                 r1.Y[pome.species_index(RPV_param[2])] / pome.molecular_weights[pome.species_index(RPV_param[2])] + \
        RPV[n] = r1.Y[pome.species_index(RPV_param[0])] + \
                 r1.Y[pome.species_index(RPV_param[1])] + \
                 r1.Y[pome.species_index(RPV_param[2])] #+ \
#                 r1.Y[pome.species_index(RPV_param[3])]

    # heat release rate [W/m^3] and ignition delays
    from scipy.signal import find_peaks

    values[:, 2] = - np.sum(production_rate * molar_enthalpies, axis=1)
    # Net production rates for each species. [kmol/m^3/s] for bulk phases or [kmol/m^2/s] for surface phases.
    # partial_molar_enthalpies: Array of species partial molar enthalpies[J / kmol]

    # add Q to values vector
    max_Q = np.argmax(values[:, 2])
    peaks, _ = find_peaks(values[:, 2], prominence=values[max_Q, 2] / 100)  # define minimum height

    if peaks.any():
        first_ignition_delay = peaks[0] * t_step * 1.e+3
        main_ignition_delay = max_Q * t_step * 1.e+3

        if information_print is True:
            print('The first stage ignition delay is {:.3f} ms'.format(first_ignition_delay))
            print('The second/ main stage ignition delay: is {:.3f} ms'.format(main_ignition_delay))

    else:
        first_ignition_delay = 0
        main_ignition_delay = 0

        if information_print is True:
            print('No ignition delay')

    return values, first_ignition_delay, main_ignition_delay, RPV