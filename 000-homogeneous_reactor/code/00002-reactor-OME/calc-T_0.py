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

# Change the RPV calculation method
PV_p = np.array(['H2O', 'CO2', 'CH2O'])

# %% Homogeneous reactor simulation
mechanism = np.array(['cai_ome14_2019.xml', 'OME3'])
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

values = np.zeros((n_samples, 17))

path = Path(__file__).resolve()
path_h = path.parents[2] / 'data/00002-reactor-OME/enthalpies_of_formation.csv'
h0 = pd.read_csv(path_h)
h0_mass = h0[['h0_mass']]
h0_mass = h0_mass.to_numpy()
h0_mole = h0[['h0_mole']]
h0_mole = h0_mole.to_numpy()

enthalpy_cp = 0

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

    state = r1.get_state()

    r1.thermo.basis = 'mass'
    u_thermo = r1.thermo.u
    U_thermo = u_thermo * state[0]
    u_self = (r1.thermo.T * r1.thermo.s + r1.thermo.g - r1.thermo.P * r1.thermo.v) * state[0]

    if n == 0:
        s_0 = r1.thermo.s
        g_0 = r1.thermo.g
        v_0 = r1.thermo.v
        H_ref = np.sum(r1.thermo.delta_enthalpy)
        print('The initial conditions for the temperature of {}K are: \n'
              'entropy s: {:.3f} [J/kgK] free gibbs energy: {:.3f} [J/kg] volume: {:.3f} [m³/kg]'.format(
               reactorTemperature, s_0, g_0, v_0))

    T_start = (state[2] / state[0] - g_0 + reactorPressure * v_0) / s_0

    # different ways to calculate the enthalpy or internal energy
    u_state = state[2]
    h_thermo = r1.thermo.enthalpy_mass * state[0]

    T = (state[2] / state[0] - r1.thermo.g + r1.thermo.P * r1.thermo.v) / r1.thermo.s

    species_enthalpies = r1.thermo.partial_molar_enthalpies * r1.thermo.concentrations

    enthalpy_cp = np.sum(r1.thermo.partial_molar_cp * r1.thermo.concentrations * r1.thermo.T)

    formation_enthalpy_mass = np.sum(h0_mass * r1.thermo.Y * state[0])
    formation_enthalpy_mole = np.sum(h0_mole * r1.thermo.concentrations * r1.volume)

    values[n] = (time,
                 r1.thermo.enthalpy_mass - np.sum(h0_mass*r1.thermo.Y)/pome.n_species,
                 np.sum(r1.thermo.delta_enthalpy),
                 r1.thermo.cp_mass * state[0] * r1.thermo.T,
                 r1.thermo.T,
                 r1.thermo.P,
                 np.sum(r1.thermo.delta_standard_enthalpy),
                 np.sum(species_enthalpies),
                 h_thermo,
                 np.sum(r1.thermo.standard_enthalpies_RT) * r1.thermo.T * ct.gas_constant,
                 enthalpy_cp,
                 np.sum(r1.thermo.standard_enthalpies_RT * r1.thermo.Y * state[0] / pome.molecular_weights) * r1.thermo.T * ct.gas_constant,
                 np.sum(r1.thermo.partial_molar_cp * r1.thermo.concentrations * r1.thermo.T),
                 U_thermo + r1.thermo.P * r1.volume,
                 formation_enthalpy_mass,
                 formation_enthalpy_mole,
                 np.sum(r1.thermo.standard_enthalpies_RT * r1.thermo.concentrations) * r1.thermo.T * ct.gas_constant)

    n += 1

values = values[:n, :]

title = '{} PODE{} $\\Phi$={:.1f} p={}bar $T_0$={:.0f}K'.format(mechanism[0], pode, equivalence_ratio,
                                                                reactorPressure / ct.one_atm, reactorTemperature)
# %% enthalpy
# plt.plot(values[:, 0], values[:, 8], label='h_thermo')
plt.plot(values[:, 0], values[:, 2], label='delta_enthalpy')
plt.title(title)
plt.xlabel('time [ms]')
plt.ylabel('h [J]')
plt.legend()
plt.show()
##
# %%

plt.plot(values[:, 0], values[:, 15] / values[:, 14], label='sum')
plt.title(title)
plt.xlabel('time [ms]')
plt.ylabel('h [J]')
plt.legend()
plt.show()
#
# # %%
# plt.plot(values[:, 0], values[:, 9], label='standard enthalpy RT')
# plt.title(title)
# plt.xlabel('time [ms]')
# plt.ylabel('h [J]')
# plt.legend()
# plt.show()
#
#%%
plt.plot(values[:, 0], values[:, 10], label='enthalpy_cp')
plt.title(title)
plt.xlabel('time [ms]')
plt.ylabel('h [J]')
plt.legend()
plt.show()

# %%
plt.plot(values[:, 0], values[:, 3], label='cp mass')
plt.title(title)
plt.xlabel('time [ms]')
plt.ylabel('h [J]')
plt.legend()
plt.show()
#
# %%
plt.plot(values[:, 0], values[:, 16], label='standard enthalpy RT concent')
plt.title(title)
plt.xlabel('time [ms]')
plt.ylabel('h [J]')
plt.legend()
plt.show()

# %%
plt.plot(values[:, 0], values[:, 11], label='standard enthalpy RT mass')
plt.title(title)
plt.xlabel('time [ms]')
plt.ylabel('h [J]')
plt.legend()
plt.show()

# %% internal energy
plt.plot(values[:, 0] * 1.e+3, values[:, 14], label='h0 mass')
# plt.title(title)
plt.xlabel('time [ms]')
plt.ylabel('H [J]')
plt.legend()
plt.show()

# %%
plt.plot(values[:, 0] * 1.e+3, values[:, 15], label='h0 mole')
# plt.title(title)
plt.xlabel('time [ms]')
plt.ylabel('H [J]')
plt.legend()
plt.show()
#
# # %% Temperature comparison to validate formula
# plt.plot(values[:, 0] * 1.e+3, values[:, 4], label='temperature from cantera thermo')
# plt.plot(values[:, 0] * 1.e+3, values[:, 11], label='T_u_state')
# plt.title(title)
# plt.xlabel('time [ms]')
# plt.ylabel('T [K]')
# plt.legend()
# plt.show()
#
# # %% Plot the starting temperature calculated with the internal energy value at every point in time
# plt.plot(values[:, 0] * 1.e+3, values[:, 9], label='T_start')
# plt.title(title)
# plt.xlabel('time [ms]')
# plt.ylabel('T [K]')
# plt.legend()
# plt.show()
#
# # %% Plot difference
# T_diff = values[:, 9] - reactorTemperature
# plt.plot(values[:, 0] * 1.e+3, T_diff, label='T_diff')
# plt.title(title)
# plt.xlabel('time [ms]')
# plt.ylabel('T [K]')
# plt.show()
# print('The maximal difference between actual and calculated initial temperature is {}K'.format(np.amax(T_diff)))
