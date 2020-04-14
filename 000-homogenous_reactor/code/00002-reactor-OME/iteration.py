#######################################################################################################################
# Iteration through different temperatures, pressures, ... of the homogeneous reactor
#######################################################################################################################

# Import packages
from Homogeneous_Reactor import homogeneous_reactor
from plot_ign_delays import plot_ign
import cantera as ct
import numpy as np

# %% Run homogeneous reactor model
iteration = True
information_print = True

iii = 0  # variable to track the first run of the reactor model for the samples

# Plotting the process of: thermodynamic properties (T, p, V), species development, heat release, tabulated chemistry
# progress variable
process_plot = [False, False, False, False]
# Plotting the ignition delay for teh different settings:
plot_ign_delays = True

if iteration is True:
    # Initialise different configurations
    equivalence_ratio = [0.5, 1.0, 1.5]
    reactorPressure = ct.one_atm * np.array([10.0, 15.0])
    air_O2 = 1.0
    air_N2 = [8, 15.0, 20.0]
    # Define temperature interval and step size
    reactorTemperature_start = 640
    reactorTemperature_end = 865
    reactorTemperature_step = 15

    for ii, equivalence_ratio_run in enumerate(equivalence_ratio):  # enumerate through different equivalence ratios
        if information_print is True:
            print(ii, equivalence_ratio_run)

        for i, reactorPressure_run in enumerate(reactorPressure):  # enumerate through different pressures
            if information_print is True:
                print(i, reactorPressure_run)

            ign_delay_run = np.zeros(((reactorTemperature_end-reactorTemperature_start)//reactorTemperature_step, 5))
            n = 0

            for reactorTemperature in range(reactorTemperature_start, reactorTemperature_end, reactorTemperature_step):
                samples_run = []
                # start homogeneous reactor model with defined settings
                # values vector: 'time (s)', 'T1 (K)', 'P1 (Bar)', 'V1 (m3)', 'YPOME', 'YCO2', 'YO2', 'Y_CO', 'Y_H2O',
                # 'Y_OH', 'Y_H2O2'
                values, Q, first_ignition_delay, main_ignition_delay, RPV = homogeneous_reactor(equivalence_ratio_run,
                                                                                                reactorPressure_run,
                                                                                                reactorTemperature,
                                                                                                air_O2, air_N2[ii],
                                                                                                process_plot)

                # create training data
                ones = np.ones((values.shape[0], 1))
                samples_run = np.concatenate((equivalence_ratio_run * ones, air_N2[ii] * ones, air_O2 * ones, values),
                                             axis=1)

                if iii is 0:
                    iii += 1
                    print(iii)
                    samples = samples_run
                else:
                    samples = np.vstack((samples, samples_run))

                # saving ignition delays
                if first_ignition_delay != 0 and main_ignition_delay != 0:
                    ign_delay_run[n] = (equivalence_ratio_run, reactorPressure_run, reactorTemperature,
                                        first_ignition_delay, main_ignition_delay)
                    n += 1

                    if information_print is True:
                        print('For settings: $\\theta$=%.1f, p=%.0fbar, T=%.0fK the delays are: first %.0fms, main'
                              ' %.0fms' % (equivalence_ratio_run, reactorPressure_run / 1.e+5, reactorTemperature,
                                           first_ignition_delay, main_ignition_delay))

                else:  # cancelling rows if ignition didn't happened
                    n_rows, _ = ign_delay_run.shape
                    ign_delay_run = ign_delay_run[:n_rows-1, :]

                    if information_print is True:
                        print('at T=%.0f K no ignition happened in the first 100ms' % (reactorTemperature))

            if ii is 0:
                ign_delay = ign_delay_run
            else:
                ign_delay = np.vstack((ign_delay, ign_delay_run))

            if plot_ign_delays is True:
                plot_ign(ign_delay_run)

else:
    # create samples just for one parameter setting to have a closer look at specific behavior
    equivalence_ratio = 1.0
    reactorPressure = 10 * ct.one_atm
    reactorTemperature = 800.0
    air_O2 = 1.0
    air_N2 = 15.0

    values, Q, first_ignition_delay, main_ignition_delay, RPV = homogeneous_reactor(equivalence_ratio, reactorPressure,
                                                                                    reactorTemperature, air_O2, air_N2,
                                                                                    process_plot)

    # show ignition delays
    if information_print is True:
        print('For settings: $\\theta$=%.0f, p=%.0fbar, T=%.0fK the delays are: first %.0fms, main  %.0fms'
              % (equivalence_ratio, reactorPressure, reactorTemperature, first_ignition_delay, main_ignition_delay))

    # create training data
    ones = np.ones((values.shape[0], 1))
    samples = np.concatenate((equivalence_ratio * ones, air_N2 * ones, air_O2 * ones, values), axis=1)

#%% Save samples as numpy array

np.save('reactor_samples', samples)
np.save('heat_release', Q)
np.save('ignition_delays', ign_delay)