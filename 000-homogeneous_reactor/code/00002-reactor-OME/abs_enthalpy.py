########################################################################################################################
# Calculate the initial temperature at every point in time
########################################################################################################################
# %% Import Packages
import cantera as ct
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

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
pome = ct.Solution(mechanism[0])
pome.TP = reactorTemperature, reactorPressure
pome.set_equivalence_ratio(equivalence_ratio, mechanism[1], 'O2:0.21 N2:0.79')

# Create Reactor
r1 = ct.IdealGasReactor(contents=pome, name='homogeneous_reactor')
sim = ct.ReactorNet([r1])
sim.max_err_test_fails = 10

#  Solution of reaction
time = 0.0
n_samples = 12500
n = 0

values = np.zeros((n_samples, 10))

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
    PV = r1.Y[pome.species_index(PV_p[0])] / pome.molecular_weights[pome.species_index(PV_p[0])] + \
         r1.Y[pome.species_index(PV_p[1])] / pome.molecular_weights[pome.species_index(PV_p[1])] * 0.15 + \
         r1.Y[pome.species_index(PV_p[2])] / pome.molecular_weights[pome.species_index(PV_p[2])] * 1.5

    states = r1.get_state()
    r1.thermo.basis = 'mass'

    # keep track of the enthalpy over
    values[n] = (time,
                 PV,
                 r1.thermo.enthalpy_mole - (np.sum(h0_mole * r1.thermo.X)),
                 r1.thermo.enthalpy_mass - (np.sum(h0_mass * r1.thermo.Y)),
                 np.sum(h0_mole * r1.thermo.X),
                 np.sum(h0_mass * r1.thermo.Y),
                 r1.thermo.enthalpy_mole,
                 r1.thermo.enthalpy_mass,
                 np.sum(r1.thermo.T * r1.thermo.s + r1.thermo.g - r1.thermo.P * r1.thermo.v) * states[0],
                 np.sum(r1.thermo.partial_molar_cp * r1.thermo.X * r1.thermo.T))

    n += 1

values = values[:n, :]

title = '{} PODE{} $\\Phi$={:.1f} p={}bar $T_0$={:.0f}K'.format(mechanism[0], pode, equivalence_ratio,
                                                                reactorPressure / ct.one_atm, reactorTemperature)

# %%
plt.plot(values[:, 0] * 1.e+3, values[:, 2], label='H abs mole')
plt.title(title)
plt.xlabel('time [ms]')
plt.ylabel('H [J/kmol]')
plt.legend()
plt.show()

# %%
plt.plot(values[:, 0] * 1.e+3, values[:, 3], label='H abs mass')
plt.title(title)
plt.xlabel('time [ms]')
plt.ylabel('H [J/kg]')
plt.legend()
plt.show()

# %%
plt.plot(values[:, 0] * 1.e+3, values[:, 4], label='H formation mole')
plt.title(title)
plt.xlabel('time [ms]')
plt.ylabel('H [J/kmol]')
plt.legend()
plt.show()

# %%
plt.plot(values[:, 0] * 1.e+3, values[:, 5], label='H formation mass')
plt.title(title)
plt.xlabel('time [ms]')
plt.ylabel('H [J/kg]')
plt.legend()
plt.show()

# %%
plt.plot(values[:, 0] * 1.e+3, values[:, 6], label='H  mole')
plt.title(title)
plt.xlabel('time [ms]')
plt.ylabel('H [J/kmol]')
plt.legend()
plt.show()

# %%
plt.plot(values[:, 0] * 1.e+3, values[:, 7], label='H  mass')
plt.title(title)
plt.xlabel('time [ms]')
plt.ylabel('H [J/kg]')
plt.legend()
plt.show()

# %%
plt.plot(values[:, 0] * 1.e+3, values[:, 8], label='H cp')
# plt.title(title)
plt.xlabel('time [ms]')
plt.ylabel('H [J/kmol]')
plt.legend()
plt.show()