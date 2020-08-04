########################################################################################################################
# Calculate the initial temperature at every point in time
########################################################################################################################
# %% Import Packages
import cantera as ct
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use('stfs')

# Suppress warnings
ct.suppress_thermo_warnings()

# %% Homogeneous reactor simulation
mechanism = np.array(['cai_ome14_2019.xml', 'OME3'])
PV_p = np.array(['H2O', 'CO2', 'CH2O'])

equivalence_ratio = 1.0
reactorPressure = 20 * ct.one_atm
reactorTemperature = 650

t_end = 0.010
t_step = 1.e-6
pode = 3

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

values = np.zeros((n_samples, 10))
h_major_species = np.zeros((n_samples, 6))
abs_energy = np.zeros((n_samples, 1))

# Load enthalpy of formation
path = Path(__file__).resolve()
path_h = path.parents[2] / 'data/00002-reactor-OME/enthalpies_of_formation.csv'
h0 = pd.read_csv(path_h)
h0_mass = h0[['h0_mass']]
h0_mass = h0_mass.to_numpy()
h0_mole = h0[['h0_mole']]
h0_mole = h0_mole.to_numpy()

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
        grad_P = np.gradient(values[:(n+1), 8])

    #  gradient from 2 time steps earlier, because np.gradient would otherwise take zeros into account
    if grad_run[n - 2] > 1.e-6:
        time += t_step / 100
    else:
        time += t_step

    # Calculate the reactor parameters for the point in time
    sim.advance(time)

    # calculate the PV
    PV = r1.Y[pode.species_index(PV_p[0])] / pode.molecular_weights[pode.species_index(PV_p[0])] + \
         r1.Y[pode.species_index(PV_p[1])] / pode.molecular_weights[pode.species_index(PV_p[1])] * 0.15 + \
         r1.Y[pode.species_index(PV_p[2])] / pode.molecular_weights[pode.species_index(PV_p[2])] * 1.5

    states = r1.get_state()
    r1.thermo.basis = 'mass'

    h_formation = 0
    for i in range(len(h0_mass)):
        h_formation = h_formation + r1.Y[i] * h0_mass[i]

    # keep track of the enthalpy over
    values[n] = (time,
                 PV,
                 r1.thermo.enthalpy_mole + (np.sum(h0_mole * r1.thermo.X)),
                 r1.thermo.enthalpy_mass + h_formation,
                 np.sum(h0_mole * r1.thermo.X),
                 h_formation,
                 r1.thermo.enthalpy_mole,
                 r1.thermo.enthalpy_mass,
                 r1.thermo.P,
                 grad_P[n-2]) # damit ich es abziehen kann, müsste ich es eig noch umrechnen, (P * V / M), aber dann noch kleiner

    h_major_species[n] = (r1.Y[pode.species_index('CO2')] * h0_mass[pode.species_index('CO2')],
                          r1.Y[pode.species_index('O2')] * h0_mass[pode.species_index('O2')],
                          r1.Y[pode.species_index('CO')] * h0_mass[pode.species_index('CO')],
                          r1.Y[pode.species_index('H2O')] * h0_mass[pode.species_index('H2O')],
                          r1.Y[pode.species_index('OME3')] * h0_mass[pode.species_index('OME3')],
                          r1.Y[pode.species_index('N2')] * h0_mass[pode.species_index('N2')])

    n += 1

values = values[:n, :]
h_major_species = h_major_species[:n, :]
abs_energy = abs_energy[:n, :]
r1.thermo.basis = 'mass'
abs_energy = np.gradient(values[:, 7]) - (np.gradient(values[:, 8]) / np.mean(r1.thermo.density))

title = '{} PODE{} $\\Phi$={:.1f} p={}bar $T_0$={:.0f}K'.format(mechanism[0], pode, equivalence_ratio,
                                                                reactorPressure / ct.one_atm, reactorTemperature)

# %%
#plt.plot(values[:, 0] * 1.e+3, values[:, 3], label='H abs mass')
#plt.plot(values[:, 0] * 1.e+3, values[:, 5], label='H formation mass')
plt.plot(values[:, 0] * 1.e+3, values[:, 7], label='H mass')
# plt.title(title)
plt.xlabel('time [ms]')
plt.ylabel('H [J/kg]')
plt.legend()
plt.show()

# %%
plt.plot(values[:, 0] * 1.e+3, values[:, 9], label='grad P')

plt.xlabel('time [ms]')
plt.ylabel('P/s [Pa/s]')
plt.legend()
plt.show()

# %%
plt.plot(values[:, 0] * 1.e+3, abs_energy, label='mass minus pressure')

plt.xlabel('time [ms]')
plt.ylabel('DH/Dt [J/kg/s]')
plt.legend()
plt.show()

# %%
plt.plot(values[:, 0] * 1.e+3, h_major_species[:, 0], label='CO2')
plt.plot(values[:, 0] * 1.e+3, h_major_species[:, 1], label='CO')
plt.plot(values[:, 0] * 1.e+3, h_major_species[:, 2], label='O2')
plt.plot(values[:, 0] * 1.e+3, h_major_species[:, 3], label='H2O')
plt.plot(values[:, 0] * 1.e+3, h_major_species[:, 4], label='OME3')
plt.plot(values[:, 0] * 1.e+3, h_major_species[:, 5], label='N2')
plt.xlabel('time [ms]')
plt.ylabel('H [J/kg]')
plt.legend()
plt.show()

# %%
plt.plot(values[:, 0] * 1.e+3, values[:, 5] / 3.5 + values[:, 7], label='H cp')
# plt.title(title)

plt.xlabel('time [ms]')
plt.ylabel('H [J/kmol]')
plt.legend()
plt.show()