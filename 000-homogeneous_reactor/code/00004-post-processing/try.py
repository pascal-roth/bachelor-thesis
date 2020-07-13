# %% Import Packages
import cantera as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

mechanism = mechanism = np.array(['cai_ome14_2019.xml', 'OME3'])
equivalence_ratio = 0.5
reactorPressure = 10 * ct.one_atm
reactorTemperature = 650
t_end = 0.013
t_step = 1.e-6
pode_nbr = 3
O2 = 0.21
N2 = 0.79
information_print = True

ct.suppress_thermo_warnings()

PV_p = np.array(['H2O', 'CO2', 'CH2O'])

path = Path(__file__).resolve()
path_h = path.parents[2] / 'data/00002-reactor-OME/enthalpies_of_formation.csv'
h0 = pd.read_csv(path_h)
h0_mass = h0[['h0_mass']]
h0_mass = h0_mass.to_numpy()

pode = ct.Solution(mechanism[0])

Z=0

# Create Reactor
pode.TP = reactorTemperature, reactorPressure
pode.set_equivalence_ratio(equivalence_ratio, mechanism[1], 'O2:{} N2:{}'.format(O2, N2))

r1 = ct.IdealGasReactor(contents=pode, name='homogeneous_reactor')
sim = ct.ReactorNet([r1])
sim.max_err_test_fails = 10

#  Solution of reaction
time = 0.0
n_samples = 13000
n = 0
samples_after_ignition = 300
stop_criterion = False

values = np.zeros((n_samples, 19))

# calculation of abs enthalpy not fixed --> assume enthatly at t_0 as constant
H = r1.thermo.enthalpy_mass - (np.sum(h0_mass * r1.thermo.Y))
OME3_0 = r1.Y[pode.species_index('OME3')]

while time < t_end:
    # calculate grad to define step size and stop_criterion
    if n <= 1:
        grad_PV = np.zeros((3))
        grad_T = np.zeros((3))
    else:
        grad_PV = np.gradient(values[:(n + 1), 7])
        grad_T = np.gradient(values[:(n + 1), 9])

    #  gradient from 2 time steps earlier, because np.gradient would otherwise take zeros into account
    if grad_PV[n - 2] > 1.e-3:
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
    PV = r1.Y[pode.species_index(PV_p[0])] + \
         r1.Y[pode.species_index(PV_p[2])] * 1.5 - \
         r1.Y[pode.species_index('OME3')] * 0.25 + OME3_0 * 0.25

    Q = - np.sum(r1.thermo.net_production_rates * r1.thermo.partial_molar_enthalpies)

    # Summarize all values to be saved in an array
    values[n] = (pode_nbr, equivalence_ratio, reactorPressure, reactorTemperature, H, Z, time, PV, Q, r1.thermo.T,
                 r1.thermo.P, r1.volume, r1.Y[pode.species_index(mechanism[1])],
                 r1.Y[pode.species_index('CO2')], r1.Y[pode.species_index('O2')],
                 r1.Y[pode.species_index('CO')], r1.Y[pode.species_index('H2O')],
                 r1.Y[pode.species_index('H2')], r1.Y[pode.species_index('CH2O')])

    n += 1

    if n == n_samples and time < t_end:
        print('WARNING: maximum nbr of samples: {} taken and only {:.4f}s reached'.format(n_samples, time))
        break

values = values[:n, :]

# ignition delay times
from scipy.signal import find_peaks

max_Q = np.argmax(values[:, 8])
peaks, _ = find_peaks(values[:, 8], prominence=values[max_Q, 8] / 100)  # define minimum height

if peaks.any() and values[max_Q, 9] > (reactorTemperature * 1.15):
    first_ignition_delay = values[peaks[0], 6] * 1.e+3
    main_ignition_delay = values[max_Q, 6] * 1.e+3

else:
    first_ignition_delay = 0
    main_ignition_delay = 0

# print information about parameter setting and ignition
if information_print is True and 0 < main_ignition_delay < t_end * 1.e+3:
        print('For settings: Phi={:2.2e}, p={:.0f}bar, T={:2.2e}K the delays are: first {:6.5e}ms, '
              'main {:6.5e}ms'.format(equivalence_ratio, reactorPressure / ct.one_atm,
                                      reactorTemperature, first_ignition_delay, main_ignition_delay))

elif information_print is True and main_ignition_delay is 0:
    print('For settings: Phi={:2.2e}, p={:.0f}bar, T={:2.2e}K ignition will happen after the '
          'monitored interval'.format(equivalence_ratio, reactorPressure / ct.one_atm,
                                      reactorTemperature))

elif information_print is True and main_ignition_delay is t_end * 1.e+3 * 0.99:
    print('For settings: Phi={:2.2e}, p={:.0f}bar, T={:2.2e}K \tignition happens shortly after the end'
          ' of the interval {}ms'.format(equivalence_ratio, reactorPressure / ct.one_atm,
                                         reactorTemperature, t_end * 1.e+3))

samples = pd.DataFrame(values)
samples.columns = ['pode', 'phi', 'P_0', 'T_0', 'H', 'Z', 'time', 'PV', 'Q', 'T', 'P', 'V',  'PODE', 'CO2',
                   'O2', 'CO', 'H2O', 'H2', 'CH2O']

samples.plot('time', 'PV', style='m-')
plt.xlabel('time in ms')
plt.ylabel('PV')

plt.title('{} PODE{} $\\Phi$={:.1f} p={}bar $T_0$={:.0f}K'.format(mechanism[0], pode_nbr, equivalence_ratio,
                                                                  reactorPressure, reactorTemperature))

plt.show()

samples[['CO2']] = samples[['CO2']] * 0.15
samples[['CH2O']] = samples[['CH2O']] * 1.5
samples[['PODE']] = samples[['PODE']] * 0.25
samples[['H2O']] = samples[['H2O']]

samples.plot('time', ['CO2', 'H2O', 'CH2O', 'PV', 'PODE'],
             style=['r-', 'b-', 'k-', 'm-', 'y'],
             label=['$Y_{CO2}$', '$Y_{H2O}$', '$Y_{CH2O}$', 'PV', '$Y_{PODE3}'])

plt.legend(loc="upper right")
plt.xlabel('time in ms')
plt.ylabel('mass fraction / molecular weight')
plt.title('{} PODE{} $\\Phi$={:.1f} p={}bar $T_0$={:.0f}K'.format(mechanism[0], pode_nbr, equivalence_ratio,
                                                                  reactorPressure, reactorTemperature))

plt.show()

#%%

samples.plot('time', 'Q',
             style='r-',
             label='Q')

plt.legend(loc="upper right")
plt.xlabel('time in ms')
plt.ylabel('heat release')
plt.title('{} PODE{} $\\Phi$={:.1f} p={}bar $T_0$={:.0f}K'.format(mechanism[0], pode_nbr, equivalence_ratio,
                                                                  reactorPressure, reactorTemperature))

plt.show()