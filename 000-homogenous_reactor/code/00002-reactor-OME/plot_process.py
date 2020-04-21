#######################################################################################################################
# Plot important process parameters like species, heat release, temperature, pressure and the reaction progress variable
#######################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def plot_process_f(values, first_ignition_delay, main_ignition_delay, RPV, process_plot, equivalence_ratio,
                   reactorPressure, reactorTemperature, time, pRPV):
    # find plot interval

    max_Q = np.argmax(values[:, 2])
    peaks, _ = find_peaks(values[:, 2], prominence=values[max_Q, 2]/100)  # define minimum height

    if peaks.any():
        if max_Q > 100:
            x1 = int(round(0.999 * peaks[0], 0))
            x2 = int(round(1.001 * max_Q, 0))
        else:
            x1 = int(round(0.8 * peaks[0], 0))
            x2 = int(round(1.2 * max_Q, 0))
    else:
        x1 = 0
        x2 = values.shape[0]

    # create a title for the plots
    title_plot = ('$\\Phi$={:.2f} p={:.0f} bar $T_0$={:.1f}K'.format(equivalence_ratio, reactorPressure / 1.e+5,
                                                                     reactorTemperature))

    if process_plot[0] is True and time is True:
        plt.clf()
        plt.subplot(2, 2, 1)
        h = plt.plot(values[x1:x2, 0] * 1.e+6, values[x1:x2, 3], 'b-')
        plt.xlabel('Time ($\mu$ s)')
        plt.ylabel('Temperature (K)')

        plt.subplot(2, 2, 2)
        plt.plot(values[x1:x2, 0] * 1.e+6, values[x1:x2, 4] / 1e5, 'b-')
        plt.xlabel('Time ($\mu$ s)')
        plt.ylabel('Pressure (Bar)')

        plt.subplot(2, 2, 3)
        plt.plot(values[x1:x2, 0] * 1.e+6, values[x1:x2, 5], 'b-')
        plt.xlabel('Time ($\mu$ s)')
        plt.ylabel('Volume (m$^3$)')
        
        plt.tight_layout()
        plt.figlegend(h, ['Reactor 1'], loc='lower right')
        plt.savefig('/media/pascal/DATA/000-Homogeneous-Reactor/plt_he_2018_{:.1f}_{:.0f}_{:.0f}_TPV_time.png'.format
                    (equivalence_ratio, reactorPressure / 1.e+5, reactorTemperature))
        plt.show()

    if process_plot[0] is True and pRPV is True:
        plt.clf()
        plt.subplot(2, 2, 1)
        h = plt.plot(RPV[x1:x2], values[x1:x2, 3], 'b-')
        plt.xlabel('RPV')
        plt.ylabel('Temperature (K)')

        plt.subplot(2, 2, 2)
        plt.plot(RPV[x1:x2], values[x1:x2, 4] / 1e5, 'b-')
        plt.xlabel('RPV')
        plt.ylabel('Pressure (Bar)')

        plt.subplot(2, 2, 3)
        plt.plot(RPV[x1:x2], values[x1:x2, 5], 'b-')
        plt.xlabel('PRV')
        plt.ylabel('Volume (m$^3$)')

        plt.tight_layout()
        plt.figlegend(h, ['Reactor 1'], loc='lower right')
        plt.savefig('/media/pascal/DATA/000-Homogeneous-Reactor/plt_he_2018_{:.1f}_{:.0f}_{:.0f}_TPV_pRPV.png'.format
                    (equivalence_ratio, reactorPressure / 1.e+5, reactorTemperature))
        plt.show()

    if process_plot[1] is True and time is True:
        plt.plot(values[x1:x2, 0] * 1.e+3, values[x1:x2, 6],  'b-', label='$Y_{pode_n}$')
        plt.plot(values[x1:x2, 0] * 1.e+3, values[x1:x2, 7],  'r-', label='$Y_{CO2}$')
        plt.plot(values[x1:x2, 0] * 1.e+3, values[x1:x2, 8],  'g-', label='$Y_{O2}$')
        plt.plot(values[x1:x2, 0] * 1.e+3, values[x1:x2, 9],  'y-', label='$Y_{CO}$')
        plt.plot(values[x1:x2, 0] * 1.e+3, values[x1:x2, 10], 'c-', label='$Y_{H20}$')
        plt.plot(values[x1:x2, 0] * 1.e+3, values[x1:x2, 11], 'm-', label='$Y_{OH}$')
        plt.plot(values[x1:x2, 0] * 1.e+3, values[x1:x2, 12], 'k-', label='$Y_{H2O2}$')

        plt.legend(loc="upper right")
        plt.xlabel('Time (ms)')
        plt.ylabel('mass fraction value')
        plt.title(title_plot)
        plt.savefig(
            '/media/pascal/DATA/000-Homogeneous-Reactor/plt_he_2018_{:.1f}_{:.0f}_{:.0f}_species_time.png'.format
            (equivalence_ratio, reactorPressure / 1.e+5, reactorTemperature))
        plt.show()

    if process_plot[1] is True and pRPV is True:
        plt.plot(RPV[x1:x2], values[x1:x2, 6],  'b-', label='$Y_{pode_n}$')
        plt.plot(RPV[x1:x2], values[x1:x2, 7],  'r-', label='$Y_{CO2}$')
        plt.plot(RPV[x1:x2], values[x1:x2, 8],  'g-', label='$Y_{O2}$')
        plt.plot(RPV[x1:x2], values[x1:x2, 9],  'y-', label='$Y_{CO}$')
        plt.plot(RPV[x1:x2], values[x1:x2, 10], 'c-', label='$Y_{H20}$')
        plt.plot(RPV[x1:x2], values[x1:x2, 11], 'm-', label='$Y_{OH}$')
        plt.plot(RPV[x1:x2], values[x1:x2, 12], 'k-', label='$Y_{H2O2}$')

        plt.legend(loc="upper right")
        plt.xlabel('RPV')
        plt.ylabel('mass fraction value')
        plt.title(title_plot)
        plt.savefig(
            '/media/pascal/DATA/000-Homogeneous-Reactor/plt_he_2018_{:.1f}_{:.0f}_{:.0f}_species_pRPV.png'.format
            (equivalence_ratio, reactorPressure / 1.e+5, reactorTemperature))
        plt.show()

    if process_plot[2] is True and time is True:
        plt.plot(values[x1:x2, 0] * 1.e+3, values[x1:x2, 2], 'b-')
        # plt.legend(['Reactor 1'])
        plt.xlabel('Time (ms)')
        plt.ylabel('Heat release [W/m$^3$]')

        textstr = '$t_1$={:.3f} ms\n$t_2$={:.3f} ms'.format(first_ignition_delay, main_ignition_delay)
        plt.text(values[x1, 0] * 1.e+3, 0.95 * values[max_Q, 2], textstr, fontsize=14, verticalalignment='top')

        plt.title(title_plot)
        plt.savefig('/media/pascal/DATA/000-Homogeneous-Reactor/plt_he_2018_{:.1f}_{:.0f}_{:.0f}_heat_time.png'.format
                    (equivalence_ratio, reactorPressure / 1.e+5, reactorTemperature))
        plt.show()

    if process_plot[2] is True and pRPV is True:
        plt.plot(RPV[x1:x2], values[x1:x2, 2], 'b-')
        # plt.legend(['Reactor 1'])
        plt.xlabel('RPV')
        plt.ylabel('Heat release [W/m$^3$]')

        if peaks.any():
            textstr = '$RPV_1$={:.3f} \n$RPV_2$={:.3f} '.format(RPV[peaks[0]], RPV[max_Q])
            plt.text(RPV[x1], 0.95 * values[max_Q, 2], textstr, fontsize=14, verticalalignment='top')

        plt.title(title_plot)
        plt.savefig('/media/pascal/DATA/000-Homogeneous-Reactor/plt_he_2018_{:.1f}_{:.0f}_{:.0f}_heat_pRPV.png'.format
                    (equivalence_ratio, reactorPressure / 1.e+5, reactorTemperature))
        plt.show()

    if process_plot[3] is True:
        plt.plot(values[x1:x2, 0] * 1.e+3, RPV[x1:x2], 'b-')
        plt.xlabel('Time (ms)')
        plt.ylabel('RPV')

        plt.title(title_plot)
        plt.savefig('/media/pascal/DATA/000-Homogeneous-Reactor/plt_he_2018_{:.1f}_{:.0f}_{:.0f}_RPV.png'.format
                    (equivalence_ratio, reactorPressure / 1.e+5, reactorTemperature))
        plt.show()

#        plt.plot(values[:, 0] * 1.e+3, RPV[:], 'b-')
#        plt.xlabel('Time (ms)')
#        plt.ylabel('RPV')
#
#        plt.title(title_plot)
#        plt.savefig('/media/pascal/DATA/000-Homogeneous-Reactor/plt_he_2018_{:.1f}_{:.0f}_{:.0f}_RPV_long.png'.format
#                    (equivalence_ratio, reactorPressure / 1.e+5, reactorTemperature))
#        plt.show()
