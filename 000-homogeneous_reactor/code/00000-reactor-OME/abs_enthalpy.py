########################################################################################################################
# Calculate the initial temperature at every point in time
########################################################################################################################
# %% Import Packages
import cantera as ct
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use('stfs_2')

# Suppress warnings
ct.suppress_thermo_warnings()


def enthalpy_of_formation():
    pode = ct.Solution('cai_ome14_2019.xml')
    species = pode.species_names
    enthalpies = np.zeros(len(species))

    for i in range(len(species)):
        pode.TPY = 298.15, ct.one_atm, '{}:1.0'.format(species[i])
        enthalpies[i] = pode.enthalpy_mole

    return enthalpies


# %% Homogeneous reactor simulation
mechanism = np.array(['cai_ome14_2019.xml', 'OME3'])

equivalence_ratio = 1.0
reactorPressure = 40 * ct.one_atm
reactorTemperature = 950

t_end = 0.010
t_step = 1.e-6
pode = 3

# Load enthalpy of formation
h_formation_species = enthalpy_of_formation()

#  Fuel mixture
pode = ct.Solution(mechanism[0])
pode.TP = reactorTemperature, reactorPressure
pode.set_equivalence_ratio(equivalence_ratio, mechanism[1], 'O2:0.21 N2:0.79')

# Create Reactor
r1 = ct.Reactor(contents=pode, name='homogeneous_reactor')
sim = ct.ReactorNet([r1])
sim.max_err_test_fails = 10

#  Solution of reaction
time = 0.0
n_samples = 12500
n = 0

# Parameters of PV
PV_p = np.array(['H2O', 'CH2O', mechanism[1], 'CO2'])
OME3_0 = r1.Y[pode.species_index(mechanism[1])]

values = np.zeros((n_samples, 6))

while time < t_end:
    if n == n_samples:
        print('WARNING: {} samples taken and {} not reached'.format(n_samples, t_end))
        break

    # calculate grad to define step size
    if n <= 1:
        grad_run = np.zeros((3))
        grad_P = np.zeros((3))
    else:
        grad_run = np.gradient(values[:(n + 1), 1])

    #  gradient from 2 time steps earlier, because np.gradient would otherwise take zeros into account
    if grad_run[n - 2] > 1.e-6:
        time += t_step / 100
    else:
        time += t_step

    # Calculate the reactor parameters for the point in time
    sim.advance(time)

    # calculate the PV
    PV = r1.Y[pode.species_index(PV_p[0])] * 0.5 + r1.Y[pode.species_index(PV_p[1])] * 0.5 + \
         (- r1.Y[pode.species_index(PV_p[2])] + OME3_0) * 0.5 + r1.Y[pode.species_index(PV_p[3])] * 0.05

    states = r1.get_state()
    r1.thermo.basis = 'mass'

    h_formation = 0
    for i in range(len(h_formation_species)):
        h_formation = h_formation + r1.Y[i] * h_formation_species[i]

    # keep track of the enthalpy over
    values[n] = (time,
                 PV,
                 h_formation,
                 r1.thermo.enthalpy_mole,
                 r1.thermo.enthalpy_mass,
                 r1.thermo.P)

    n += 1

values = values[:n, :]

enthalpy_pressure_adjusted = np.gradient(values[:, 4]) - (np.gradient(values[:, 5]) / np.mean(r1.thermo.density))

# %%
plt.plot(values[:, 0] * 1.e+3, values[:, 4], label='abs enthalpy')
plt.xlabel('time [ms]')
plt.ylabel('H [J/kg]')
plt.legend()
plt.show()

# %%
plt.plot(values[:, 0] * 1.e+3, enthalpy_pressure_adjusted, label='enthalpy pressure adjusted')

plt.xlabel('time [ms]')
plt.ylabel('DH/Dt [J/kg/s]')
plt.legend()
plt.show()
