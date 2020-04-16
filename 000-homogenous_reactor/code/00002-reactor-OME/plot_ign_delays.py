########################################################################################################################
# Function to plot the ignition delays in regard to the temperature
########################################################################################################################

def plot_ign(ign_delay_run):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogy(1000 / ign_delay_run[:, 2], ign_delay_run[:, 3] * 1.e-3, 'bx-', label='first')
    ax.semilogy(1000 / ign_delay_run[:, 2], ign_delay_run[:, 4] * 1.e-3, 'ro-', label='main')
    ax.set_ylabel('Ignition Delay (s)')
    ax.set_xlabel(r'$\frac{1000}{T (K)}$', fontsize=18)

    # Add a second axis on top to plot the temperature for better readability
    ax2 = ax.twiny()
    ticks = ax.get_xticks()
    ax2.set_xticks(ticks)
    ax2.set_xticklabels((1000 / ticks).round(1))
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel(r'Temperature: $T(K)$')

    textstr = '$\\Phi$={:.2f}\np={:.0f}bar' .format(ign_delay_run[0, 0], ign_delay_run[0, 1] / 1.e+5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top')

    ax.legend(loc='lower right')
    print('Ignition Delay Plot finished')
    plt.show()
