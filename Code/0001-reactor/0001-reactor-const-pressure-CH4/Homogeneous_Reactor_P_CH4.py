### Implementation constant pressure, fixed mass reactor ###

# Information source
# Cantera Website
# - https://cantera.org/science/reactors.html
# - https://cantera.github.io/docs/sphinx/html/cython/thermo.html
# - https://cantera.org/documentation/docs-2.4/sphinx/html/cython/zerodim.html
# - https://cantera.org/examples/python/reactors/reactor2.py.html (Example of two reactors with a piston and with heat
# loss to the environment

# Import Packages
import cantera as ct
import sys
import os
import csv

# %% Surrounding gas/ air for POMEn mixture
gas = ct.Solution('gri30.xml')
air = gas

reactorTemperature = 1000
reactorPressure = ct.one_atm

gas.TP = reactorTemperature, reactorPressure

# equivalent ratio can be set, stoichiometry ratio is equal to 1.0
gas.set_equivalence_ratio(0.5, 'CH4', 'O2:0.21, N2:0.79')

print(gas())

# %% Create Reactor Segments
env = ct.Reservoir(contents=air)

r1 = ct.ConstPressureReactor(contents=gas, energy='on', name='homogeneous_reactor')
# %% Create Reactor-Walls
A_w1 = 1.0              # Area of piston
K_w1 = 0.5e-4           # Wall expansion rate parameter [m/s/Pa]
U_w1 = 500.0            # Overall heat transfer coefficient [W/m^2]

w1 = ct.Wall(r1, env, A=A_w1, K=K_w1, U=U_w1)

sim = ct.ReactorNet([r1])

print('finished setup, begin solution...')

# %% Solution of reaction
time = 0.0
n_steps = 5000
outfile = open('reactor.csv', 'w')
csvfile = csv.writer(outfile)
csvfile.writerow(['time (s)', 'T1 (K)', 'P1 (Bar)', 'V1 (m3)'])

states1 = ct.SolutionArray(gas, extra=['t', 'V'])

for n in range(n_steps):
    time += 4.e-4
    print(n, time)
    sim.advance(time)
    states1.append(r1.thermo.state, t=time, V=r1.volume)
    csvfile.writerow([time, r1.thermo.T, r1.thermo.P, r1.volume])

outfile.close()
print('Output written to file reactor.csv')
print('Directory: ' + os.getcwd())

# %% Polt results
plot_res = True

if plot_res is True:
    import matplotlib.pyplot as plt

    plt.clf()
    plt.subplot(2, 2, 1)
    h = plt.plot(states1.t, states1.T, 'b-')
    # plt.legend(['Reactor 1'])
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')

    plt.subplot(2, 2, 2)
    plt.plot(states1.t, states1.P / 1e5, 'b-')
    # plt.legend(['Reactor 1'])
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (Bar)')

    plt.subplot(2, 2, 3)
    plt.plot(states1.t, states1.V, 'b-')
    # plt.legend(['Reactor 1'])
    plt.xlabel('Time (s)')
    plt.ylabel('Volume (m$^3$)')

    plt.figlegend(h, ['Reactor 1'], loc='lower right')
    plt.tight_layout()
    plt.show()