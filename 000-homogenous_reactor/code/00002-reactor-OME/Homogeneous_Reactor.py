### Implementation constant pressure, fixed mass reactor

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
# %% Homogeneous reactor simulation
def homogeneous_reactor(equivalence_ratio, reactorPressure, reactorTemperature, air_O2, air_N2, process_plot):
    information_print = False
    csv_save = False

    #  Surrounding gas/ air for POMEn mixture
    pome = ct.Solution('he_2018.xml')
    air = ct.Solution('gri30.xml')  # not necessary, but makes it more realistic
    pome.TP = reactorTemperature, reactorPressure

    pome.set_equivalence_ratio(equivalence_ratio, 'CH3OCH2OCH3', 'O2:%.0f N2:%.0f' % (air_O2, air_N2))

    if information_print is True:
        print(pome())

    # Create Reactor Segments
    env = ct.Reservoir(contents=air)

    r1 = ct.Reactor(contents=pome, energy='on', name='homogeneous_reactor')
    r1.chemistry_enabled = True
    # Create Reactor-Walls
    A_w1 = 1.0  # Area of piston --> default value
    K_w1 = 0.0  # Wall expansion rate parameter [m/s/Pa] --> not movable
    U_w1 = 0.0  # Overall heat transfer coefficient [W/m^2] --> no heat loss

    w1 = ct.Wall(r1, env, A=A_w1, K=K_w1, U=U_w1)
    sim = ct.ReactorNet([r1])

    if information_print is True:
        print('finished setup, begin solution...')

    #  Solution of reaction
    time = 0.0
    n_steps = 500
    timestep = 2.e-4

    if csv_save is True:
        outfile = open('reactor.csv', 'w')
        csvfile = csv.writer(outfile)
        csvfile.writerow(['time (s)', 'T1 (K)', 'P1 (Bar)', 'V1 (m3)', 'YPOME', 'YCO2', 'YO2', 'Y_CO', 'Y_H2O', 'Y_OH',
                          'Y_H2O2'])

    states1 = ct.SolutionArray(pome, extra=['t', 'V', 'Y_pome', 'Y_CO2', 'Y_O2', 'Y_CO', 'Y_H2O', 'Y_OH', 'Y_H2O2'])

    values = np.zeros((n_steps, 11))
    RPV = np.zeros(n_steps)

    for n in range(n_steps):
        time += timestep
        #    print(n, time)
        sim.advance(time)

        states1.append(r1.thermo.state, t=time, V=r1.volume, Y_pome=r1.Y[pome.species_index('CH3OCH2OCH3')],
                       Y_CO2=r1.Y[pome.species_index('CO2')], Y_O2=r1.Y[pome.species_index('O2')],
                       Y_CO=r1.Y[pome.species_index('CO')], Y_H2O=r1.Y[pome.species_index('H2O')],
                       Y_OH=r1.Y[pome.species_index('OH')], Y_H2O2=r1.Y[pome.species_index('H2O2')])

        if csv_save is True:
            csvfile.writerow([time, r1.thermo.T, r1.thermo.P, r1.volume, r1.Y[pome.species_index('CH3OCH2OCH3')],
                              r1.Y[pome.species_index('CO2')], r1.Y[pome.species_index('O2')],
                              r1.Y[pome.species_index('CO')], r1.Y[pome.species_index('H2O')],
                              r1.Y[pome.species_index('OH')], r1.Y[pome.species_index('H2O2')]])

        values[n] = (time, r1.thermo.T, r1.thermo.P, r1.volume, r1.Y[pome.species_index('CH3OCH2OCH3')],
                     r1.Y[pome.species_index('CO2')], r1.Y[pome.species_index('O2')],
                     r1.Y[pome.species_index('CO')], r1.Y[pome.species_index('H2O')],
                     r1.Y[pome.species_index('OH')], r1.Y[pome.species_index('H2O2')])

        # Calculate tabulated chemistry variables
        RPV[n] = r1.Y[pome.species_index('CO2')] / pome.molecular_weights[pome.species_index('CO2')] + \
                 r1.Y[pome.species_index('CO')] / pome.molecular_weights[pome.species_index('CO')] + \
                 r1.Y[pome.species_index('H2O')] / pome.molecular_weights[pome.species_index('H2O')]

    # heat release rate [W/m^3] and ignition delays
    from scipy.signal import find_peaks

    Q = - np.sum(states1.net_production_rates * states1.partial_molar_enthalpies, axis=1)
    # Net production rates for each species. [kmol/m^3/s] for bulk phases or [kmol/m^2/s] for surface phases.
    # partial_molar_enthalpies: Array of species partial molar enthalpies[J / kmol]

    peaks, _ = find_peaks(Q, height=0) #1.2 * Q[10])  # define minimum height to prevent showing delays if no ignition happens.

    if peaks.any():
        first_ignition_delay = peaks[0] * timestep * 1.e+3
        main_ignition_delay = np.argmax(Q) * timestep * 1.e+3

        if information_print is True:
            print('The first stage ignition delay is %.1f ms' % first_ignition_delay)
            print('The second/ main stage ignition delay: is %.1f ms' % main_ignition_delay)

    else:
        first_ignition_delay = 0
        main_ignition_delay = 0

        if information_print is True:
            print('No ignition delay')

    if csv_save is True:
        outfile.close()
        print('Output written to file reactor.csv')
        print('Directory: ' + os.getcwd())

    # %% Plot results
    if peaks.any():
        if np.argmax(Q) > 100:
            x1 = int(round(0.9*peaks[0], 0))
            x2 = int(round(1.1*np.argmax(Q), 0))
        else:
            x1 = int(round(0.8*peaks[0], 0))
            x2 = int(round(1.2*np.argmax(Q), 0))
    else:
        x1 = 0
        x2 = 0

    title_plot = ('$\\theta$=%.2f p=%.0f bar $T_{start}$=%.1f K O2:N2 = %.0f : %.0f' % (
        equivalence_ratio, reactorPressure / 1.e+5, reactorTemperature, air_O2, air_N2))

    if process_plot[0] is True:
        plt.clf()
        plt.subplot(2, 2, 1)
        h = plt.plot(states1.t[x1:x2], states1.T[x1:x2], 'b-')
        # plt.legend(['Reactor 1'])
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (K)')

        plt.subplot(2, 2, 2)
        plt.plot(states1.t[x1:x2], states1.P[x1:x2] / 1e5, 'b-')
        # plt.legend(['Reactor 1'])
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (Bar)')

        plt.subplot(2, 2, 3)
        plt.plot(states1.t[x1:x2], states1.V[x1:x2], 'b-')
        # plt.legend(['Reactor 1'])
        plt.xlabel('Time (s)')
        plt.ylabel('Volume (m$^3$)')

        plt.figlegend(h, ['Reactor 1'], loc='lower right')
        plt.show()

    if process_plot[1] is True:
        plt.plot(states1.t[x1:x2], states1.Y_pome[x1:x2], 'b-', label='$Y_{pome}$')
        plt.plot(states1.t[x1:x2], states1.Y_CO2[x1:x2], 'r-', label='$Y_{CO2}$')
        plt.plot(states1.t[x1:x2], states1.Y_O2[x1:x2], 'g-', label='$Y_{O2}$')
        plt.plot(states1.t[x1:x2], states1.Y_CO[x1:x2], 'y-', label='$Y_{CO}$')
        plt.plot(states1.t[x1:x2], states1.Y_H2O[x1:x2], 'c-', label='$Y_{H20}$')
        plt.plot(states1.t[x1:x2], states1.Y_OH[x1:x2], 'm-', label='$Y_{OH}$')
        plt.plot(states1.t[x1:x2], states1.Y_H2O2[x1:x2], 'k-', label='$Y_{H2O2}$')

        plt.legend(loc="upper right")
        plt.xlabel('Time (s)')
        plt.ylabel('mass fraction value')
        plt.title(title_plot)
        plt.show()

    if process_plot[2] is True:
        plt.plot(states1.t[x1:x2], Q[x1:x2], 'b-')
        # plt.legend(['Reactor 1'])
        plt.xlabel('Time (s)')
        plt.ylabel('Heat release [W/m$^3$]')

        textstr = '$t_{first}$=%.1f ms\n$t_{main}$=%.1f ms' % (first_ignition_delay, main_ignition_delay)
        plt.text(states1.t[x1], 0.95 * np.amax(Q), textstr, fontsize=14, verticalalignment='top')

        plt.title(title_plot)
        plt.show()

    if process_plot[3] is True:
        plt.plot(states1.t[x1:x2], RPV[x1:x2], 'b-')
        plt.xlabel('Time (s)')
        plt.ylabel('RPV')

        plt.title(title_plot)
        plt.show()

    return values, Q, first_ignition_delay, main_ignition_delay, RPV