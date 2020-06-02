#######################################################################################################################
# Iteration through different temperatures, pressures, ... of the homogeneous reactor
#######################################################################################################################

# Import packages
import argparse
import numpy as np
import cantera as ct
import pandas as pd
from Homogeneous_Reactor import homogeneous_reactor
from pre_process_fc import save_df
from pre_process_fc import create_path
from pre_process_fc import make_dir



# %% Rename arguments
reactorTemperature_start = 740
reactorTemperature_end = 740
reactorTemperature_step = 30
N2 = 0.79
O2 = 0.21
category = 'train'
number_run = '002'
mechanism_input = 'cai'
p_0 = 10
p_end = 10
p_step = 10
phi_0 = 1.0
phi_end = 1.0
phi_step = 1.0
information_print = True
pode = [3]

# Define end time and time step
if category == 'exp':
    t_end = 0.020
    t_step = 5.e-6
else:
    t_end = 0.010
    t_step = 1.e-6

# create an array for the different samples/ the ignition delays and decide if to save them
save_samples = True
save_delays = True
save_initials = True

#%% Create to save files
if not ((save_delays is False) and (save_samples is False) and (save_initials is False)):
    make_dir(mechanism_input, number_run, information_print)

if save_samples is True:
    typ = 'samples'
    samples, nnn = save_df(typ, category, mechanism_input, number_run, reactorTemperature_end,
                           reactorTemperature_start, reactorTemperature_step, phi_end, phi_0, phi_step,
                           p_end, p_0, p_step, pode, size=20)

if save_delays is True:
    typ = 'delays'
    delays, n = save_df(typ, category, mechanism_input, number_run, reactorTemperature_end,
                        reactorTemperature_start, reactorTemperature_step, phi_end, phi_0, phi_step,
                        p_end, p_0, p_step, pode, size=6)

if save_initials is True:
    typ = 'initials'
    initials, nn = save_df(typ, category, mechanism_input, number_run, reactorTemperature_end,
                           reactorTemperature_start, reactorTemperature_step, phi_end, phi_0, phi_step,
                           p_end, p_0, p_step, pode, size=7)


# %% Iterate between the parameter settings
for iii, pode_run in enumerate(pode):

    if mechanism_input == 'he':
        mechanism = np.array(['he_2018.xml', 'DMM' + str(pode_run)])
    elif mechanism_input == 'cai':
        mechanism = np.array(['cai_ome14_2019.xml', 'OME' + str(pode_run)])
    elif mechanism_input == 'sun':
        mechanism = np.array(['sun_2017.xml', 'DMM' + str(pode_run)])

    if pode_run == 1:
        mechanism[1] = 'CH3OCH2OCH3'
    elif pode == 4 and not mechanism[0] == 'cai_ome14_2019.xml':
        print('WARNING: This mechanism is not available for PODE4, will be downgraded to PODE3')
        mechanism[1] = 'DMM3'

    if information_print is True:
        print('the used mechanism is {} with fuel {}\n'.format(mechanism[0], mechanism[1]))

    for ii, equivalence_ratio_run in enumerate(np.arange(phi_0, phi_end + phi_step, phi_step)):
        # enumerate through different equivalence ratios
        if information_print is True:
            print('Equivalence ratio: {}'.format(equivalence_ratio_run))

        for reactorPressure_run in range(p_0, p_end + p_step, p_step):
            # enumerate through different pressures
            if information_print is True:
                print('\nReactor Pressure: {}'.format(reactorPressure_run))

            reactorPressure_run = np.array(reactorPressure_run) * ct.one_atm

            for reactorTemperature in range(reactorTemperature_start, reactorTemperature_end + reactorTemperature_step,
                                            reactorTemperature_step):
                # start homogeneous reactor model with defined settings
                # values vector: ['time', 'PV', 'phi', 'Q', 'T', 'P', 'V', 'U', 'PODE', 'CO2', 'O2', 'CO', 'H2O', 'H2',
                # 'CH2O']
                values, first_ignition_delay, main_ignition_delay, s_0, g_0, v_0 = homogeneous_reactor\
                    (mechanism, equivalence_ratio_run, reactorPressure_run, reactorTemperature, t_end, t_step, pode_run,
                     O2, N2)


                # saving ignition delays for the parameter setting
                if save_delays is True and 0 < main_ignition_delay < t_end*1.e+3*0.99:
                    delays[n, :] = (pode_run, equivalence_ratio_run, reactorPressure_run, reactorTemperature,
                                    first_ignition_delay, main_ignition_delay)
                    n += 1
                elif save_delays is True:  # cancelling rows if ignition didn't happened
                    n_rows, _ = delays.shape
                    delays = delays[:(n_rows - 1), :]

                # combine dataframes of different reactorTemperatures
                if save_samples is True:
                    n_samples_run = len(values)
                    samples[nnn:(nnn+n_samples_run), :] = values
                    n_samples = len(samples)
                    samples = samples[:(n_samples-(12500-n_samples_run)), :]
                    nnn += n_samples_run

                # initial values to calculate the starting temperature of the reactor
                if save_initials is True:
                    initials[nn, :] = [pode_run, equivalence_ratio_run, reactorPressure_run, reactorTemperature, s_0,
                                       g_0, v_0]
                    nn += 1

                # print information about parameter setting and ignition
                if information_print is True and 0 < main_ignition_delay < t_end*1.e+3:
                    print('For settings: Phi={:.2f}, p={:.0f}bar, T={:.0f}K the delays are: first {:.5f}ms, '
                          'main {:.5f}ms'.format(equivalence_ratio_run, reactorPressure_run / ct.one_atm,
                                                 reactorTemperature, first_ignition_delay, main_ignition_delay))

                elif information_print is True and main_ignition_delay is 0:
                    print('For settings: Phi={:.2f}, p={:.0f}bar, T={:.0f}K ignition will happen after the '
                          'monitored interval'.format(equivalence_ratio_run, reactorPressure_run / ct.one_atm,
                                                      reactorTemperature))

                elif information_print is True and main_ignition_delay is t_end*1.e+3*0.99:
                    print('For settings: Phi={:.2f}, p={:.0f}bar, T={:.0f}K ignition happens shortly after the end'
                          ' of the interval {}ms'.format(equivalence_ratio_run, reactorPressure_run / ct.one_atm,
                                                         reactorTemperature, t_end*1.e+3))

if save_delays is True:
    path_dir, _ = create_path(mechanism_input, number_run)
    path_delay = '{}/{}_{}_delays.csv'.format(path_dir, number_run, category)
    delays = pd.DataFrame(delays)
    delays.columns = ['pode', 'phi', 'P_0', 'T_0', 'first', 'main']
    delays.set_index(['pode', 'phi', 'P_0'])
    delays.to_csv(path_delay)

# save species development with the parameter setting
if save_samples is True:
    path_dir, _ = create_path(mechanism_input, number_run)
#    path_dir = '/media/pascal/TOSHIBA EXT/BA'
    path_sample = '{}/{}_{}_samples.csv'.format(path_dir, number_run, category)
    samples = pd.DataFrame(samples)
    samples.columns = ['pode', 'phi', 'P_0', 'T_0', 'U', 'H', 'Z', 'time', 'PV', 'Q', 'T', 'P', 'V',  'PODE', 'CO2',
                       'O2', 'CO', 'H2O', 'H2', 'CH2O']
    samples = samples.set_index(['pode', 'phi', 'P_0', 'T_0'])
    samples.to_csv(path_sample)

if save_initials is True:
    path_dir, _ = create_path(mechanism_input, number_run)
    path_initials = '{}/{}_{}_initials.csv'.format(path_dir, number_run, category)
    initials = pd.DataFrame(initials)
    initials.columns = ['pode', 'phi', 'P_0', 'T_0', 's', 'g', 'v']
    initials.set_index(['pode', 'phi', 'P_0'])
    initials.to_csv(path_initials)
