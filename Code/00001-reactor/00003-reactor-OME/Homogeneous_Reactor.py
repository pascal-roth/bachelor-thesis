### Implementation constant pressure, fixed mass reactor

# Information source
# Cantera Website
# - https://cantera.org/science/reactors.html
# - https://cantera.github.io/docs/sphinx/html/cython/thermo.html
# - https://cantera.org/documentation/docs-2.4/sphinx/html/cython/zerodim.html
# - https://cantera.org/examples/python/reactors/reactor2.py.html (Example of two reactors with a piston and with heat
# loss to the enviornment

# Import Packages
import cantera as ct
import numpy as np
import sys
import os
import csv

#%% Create POMEn mix

# POMDMEn has following chemical formula: CH_3O(CH_2O)_nCH_3
# thermodynamic properities:
ideal_gas = True        # Assumption that POME has a constant heat capacity
T_low = 300             # Minimum temperature [K] at which the parameterization is valid
T_high = 1000           # Maximum temperature [K] at which the parameterization is valid
P_ref = 1               # Reference pressure [Pa] for the parameterization
# coeffs = 0              # https://cantera.org/documentation/docs-2.4/sphinx/html/cython/thermo.html

pome1 = ct.Species('POME1', "C:3, H:8, O:2")   # here the single components, where can I define the structure?????

if ideal_gas is True:
    pome1.thermo = ct.ConstantCp(T_low, T_high, P_ref) # , coeffs)
#    tran = ct.GasTransportData()
#    tran.set_customary_units('nonlinear', 3.75, 141.40, 0.0, 2.60, 13.00)
#    pome1.transport = tran
#    gas = ct.Solution(thermo='IdealGas', species=[pome1])

print(pome1())

#%% Surrounding gas/ air for POMEn mixture
gas = ct.Solution('gri30.xml')
air = gas

gas.TP = 500, ct.one_atm    # Temperature and pressure of surrounding gas/ air
# equivalent ratio can be set, for stochiometric ratio is equal to 1.0
gas.set_equivalence_ratio(1.0, 'pome1', 'O2:0.21, N2:0.79')

print(gas())

#%% Mixture
# mix_1 = ct.Mixture([(gas, 1.0), (pome1, 0.1)])
# print(mix_1.species_names)

#%% Create Reactor Segments
env = ct.Reservoir(contents=air)

r1 = ct.ConstPressureReactor(content=gas, energy='on', name='homogeneous_reactor')
#%% Create Reactor-Walls
A_w1 = 1.0              # Area of piston
K_w1 = 0.5e-4           # Wall expansion rate parameter [m/s/Pa]
U_w1 = 100.0            # Overall heat transfer coefficient [W/m^2]

w1 = ct.Wall(r1, env, A=A_w1, K=K_w1, U=U_w1)

sim = ct.ReactorNet([r1])

print('finished setup, begin solution...')

#%% Solution of reaction
time = 0.0
n_steps = 5000
outfile = open('reactor.csv', 'w')
csvfile = csv.writer(outfile)
csvfile.writerow(['time (s)','T1 (K)','P1 (Bar)','V1 (m3)'])

states1 = ct.SolutionArray(gas, extra=['t', 'V'])

for n in range(n_steps):
    time += 4.e-4
    print(n, time)
    sim.advance(time)
    states1.append(r1.thermo.state, t=time, V=r1.volume)
    csvfile.writerow([time, r1.thermo.T, r1.thermo.P, r1.volume])

outfile.close()
print('Output written to file piston.csv')
print('Directory: '+os.getcwd())

# %% Polt results
plot_res = True

if plot_res is True:
    import matplotlib.pyplot as plt

    plt.clf()
    plt.subplot(2, 2, 1)
    h = plt.plot(states1.t, states1.T, 'g-')
    # plt.legend(['Reactor 1'])
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')

    plt.subplot(2, 2, 2)
    plt.plot(states1.t, states1.P / 1e5, 'g-')
    # plt.legend(['Reactor 1'])
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (Bar)')

    plt.subplot(2, 2, 3)
    plt.plot(states1.t, states1.V, 'g-')
    # plt.legend(['Reactor 1'])
    plt.xlabel('Time (s)')
    plt.ylabel('Volume (m$^3$)')

    plt.figlegend(h, ['Reactor 1'], loc='lower right')
    plt.tight_layout()
    plt.show()