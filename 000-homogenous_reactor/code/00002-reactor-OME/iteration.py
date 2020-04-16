#######################################################################################################################
# Iteration through different temperatures, pressures, ... of the homogeneous reactor
#######################################################################################################################

# Import packages
from Homogeneous_Reactor import homogeneous_reactor
from plot_ign_delays import plot_ign
from plot_process import plot_process_f
import cantera as ct
import numpy as np
from pathlib import Path

# %% Run homogeneous reactor model
iteration = False
information_print = True

# create an array for the different samples/ the ignition delays and decide if to save them
save_samples = True
save_delays = True

# Plot decisions
plot_axes = [True, True]  # Decide if process variables are plotted over time and/or over RPV
process_plot = [True, True, True, True]  # thermodynamic properties (T, p, V), species development, heat release, RPV
plot_ign_delays = True  # Decide if plotting ignition delay over temperature for different settings

# define the used mechanism
mechanism_all = np.array([['he_2018.xml', 'DMM3'], ['cai_ome14_2019.xml', 'OME3'], ['sun_2017.xml', 'DMM3']])
mechanism = mechanism_all[0, :]
if information_print is True:
    print('the used mechanism is {}'.format(mechanism[0]))

# Define end time and time step
t_end = 0.001
t_step = 1.e-9

# %% set parameters
if iteration is True:
    # Initialize different configurations
    equivalence_ratio = [0.5, 1.0, 1.5]
    reactorPressure = ct.one_atm * np.array([20.0, 25.0])
    # Define temperature interval and step size
    reactorTemperature_start = 640
    reactorTemperature_end = 865
    reactorTemperature_step = 15
else:
    # Initialize the one configuration to test
    equivalence_ratio = np.array([1.0])
    reactorPressure = ct.one_atm * np.array([20])
    # initialize the start temperature
    reactorTemperature_start = 800
    reactorTemperature_end = reactorTemperature_start + 1  # don't change
    reactorTemperature_step = 1  # don't change

# %% Iterate between the parameter settings
if save_delays is True:
    ign_delay_run = np.zeros((((reactorTemperature_end - reactorTemperature_start) // reactorTemperature_step) *
                              len(equivalence_ratio) * len(reactorPressure), 5))
    n = 0
    nn = 0

for ii, equivalence_ratio_run in enumerate(equivalence_ratio):  # enumerate through different equivalence ratios

    for i, reactorPressure_run in enumerate(reactorPressure):  # enumerate through different pressures

        for reactorTemperature in range(reactorTemperature_start, reactorTemperature_end, reactorTemperature_step):
            # start homogeneous reactor model with defined settings
            # values vector: 'time (s)', 'equivalence_ratio', 'Q', 'T1r (K)', 'P1 (Bar)', 'V1 (m3)', 'YPOME', 'YCO2',
            # 'YO2', 'Y_CO', 'Y_H2O', 'Y_OH', 'Y_H2O2'
            values, first_ignition_delay, main_ignition_delay, RPV = homogeneous_reactor(mechanism,
                                                                                         equivalence_ratio_run,
                                                                                         reactorPressure_run,
                                                                                         reactorTemperature,
                                                                                         t_end, t_step)
            # Plot the process variables
            plot_process_f(values, first_ignition_delay, main_ignition_delay, RPV, process_plot,
                           equivalence_ratio_run, reactorPressure_run, reactorTemperature,
                           time=plot_axes[0], pRPV=plot_axes[1])

            # save species development with the parameter setting
            if save_samples is True:
                # path = Path(__file__).parents[2] / 'data/00002-reactor-OME/samples_{}_{}_{}_{}.npy'.format(
                #            mechanism[0], equivalence_ratio_run, reactorPressure_run / 1.e+5, reactorTemperature)
                path = ('/media/pascal/DATA/000-Homogeneous-Reactor/samples_{}_{}_{}_{}.npy'.format(
                    mechanism[0], equivalence_ratio_run, reactorPressure_run / 1.e+5, reactorTemperature))
                np.save(path, values)

            # saving ignition delays for the parameter setting
            if save_delays is True and 0 < main_ignition_delay < t_end*1.e+3*0.99:
                ign_delay_run[n] = (equivalence_ratio_run, reactorPressure_run, reactorTemperature,
                                    first_ignition_delay, main_ignition_delay)
                n += 1
            elif save_delays is True:  # cancelling rows if ignition didn't happened
                n_rows, _ = ign_delay_run.shape
                ign_delay_run = ign_delay_run[:(n_rows - 1), :]

            # print information about parameter setting and ignition
            if information_print is True and 0 < main_ignition_delay < t_end*1.e+3:
                print('For settings: $\\Phi$={:.1f}, p={:.0f}bar, T={:.0f}K the delays are: first {:.3f}ms, '
                      'main {:.3f}ms'.format(equivalence_ratio_run, reactorPressure_run / 1.e+5,
                                            reactorTemperature, first_ignition_delay, main_ignition_delay))
            elif information_print is True and main_ignition_delay is 0:
                print('For settings: $\\Phi$={:.1f}, p={:.0f}bar, T={:.0f}K no ignition will happen in engine relevant '
                      'time'.format(equivalence_ratio_run, reactorPressure_run / 1.e+5, reactorTemperature))
            elif information_print is True and main_ignition_delay is t_end*1.e+3*0.99:
                print('For settings: $\\Phi$={:.1f}, p={:.0f}bar, T={:.0f}K ignition happens after the end of the '
                      'interval {}ms'.format(equivalence_ratio_run, reactorPressure_run / 1.e+5, reactorTemperature,
                                             t_end*1.e+3))

        if plot_ign_delays is True and iteration is True:
            plot_ign(ign_delay_run[nn:n, :])
            nn = n

if save_delays is True:
    # path = Path(__file__).parents[2] / 'data/00002-reactor-OME/samples_{}.npy'.format(mechanism[0])
    path = ('/media/pascal/DATA/000-Homogeneous-Reactor/delays_{}.npy'.format(mechanism[0]))
    np.save(path, ign_delay_run)
