#######################################################################################################################
# Plot the comparison between all available experiment datas and the data generated by the different mechanisms
#######################################################################################################################

# Import packages
import numpy as np
from pathlib import Path
import cantera as ct
import pandas as pd
import argparse
import matplotlib.pyplot as plt
plt.style.use('stfs_2')


# %% initialise dataloaders
def loaddata_sim(mechanism, nbr_run, equivalence_ratio, reactorPressure, pode):
    path = Path(__file__).resolve()
    path = path.parents[2] / 'data/00002-reactor-OME/{}/{}_exp_delays.csv'.format(mechanism[0], nbr_run)
    data = pd.read_csv(path)

    # Select only the data needed for the plot
    data = data[data.pode == pode]
    data = data[data.phi == equivalence_ratio]
    data = data[data.P_0 == reactorPressure * ct.one_atm]

    data = np.array(data)
    return data[:, 4:]


def loaddata_exp(OME, reactorPressure, equivalence_ratio):
    path = Path(__file__).resolve()
    path = path.parents[2] / 'data/00004-post-processing/data_exp/Exp_{}_{}_{}.csv'.format(OME, equivalence_ratio, reactorPressure)
    data = np.array(pd.read_csv(path, delimiter=',', decimal='.'))
    return data

# %% get input arguments
parser = argparse.ArgumentParser(description="Create exp-mech plots")

parser.add_argument("-mech", "--mechanism_input", type=str, choices=['he', 'sun', 'cai', 'all'], default='he',
                    help="chose reaction mechanism")

parser.add_argument("--pode", type=int, choices=[1, 2, 3, 4], default=3,
                    help="chose degree of polymerization")

parser.add_argument("-phi", "--equivalence_ratio", type=float, default='1.0',
                    help="chose equivalence ratio")

parser.add_argument("-p", "--pressure", type=int, default=20,
                    help="chose reactor pressure")

parser.add_argument("-nbr_run", "--number_run", type=str, default='000',
                    help="define a nbr to identify the started iterator run")

args = parser.parse_args()


mechanism_all = np.array([['he_2018.xml'], ['cai_ome14_2019.xml'], ['sun_2017.xml']])

pode = args.pode
equivalence_ratio = args.equivalence_ratio
reactorPressure = args.pressure
nbr_run = args.number_run

# %% plot data

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)

if pode <= 3:
    exp = loaddata_exp('OME' + str(pode), reactorPressure, equivalence_ratio)

    if reactorPressure == 10 and equivalence_ratio == 1.0 and nbr_run == '002':
        exp = loaddata_exp('PODE' + str(pode), reactorPressure, equivalence_ratio)
#        ax.semilogy(exp[:, 0], exp[:, 1], 'cx', label='exp_he_PODE' + str(pode))

    he = np.flip(loaddata_sim(mechanism_all[0], nbr_run, equivalence_ratio, reactorPressure, pode), axis=0)
    cai = np.flip(loaddata_sim(mechanism_all[1], nbr_run, equivalence_ratio, reactorPressure, pode), axis=0)
    sun = np.flip(loaddata_sim(mechanism_all[2], nbr_run, equivalence_ratio, reactorPressure, pode), axis=0)

    ax.semilogy(exp[:, 0], exp[:, 1], 'bx', label='exp_PODE' + str(pode))
    ax.semilogy(1000 / he[:, 0], he[:, 2], 'r-', label='sim_he_2018')
    ax.semilogy(1000 / cai[:, 0], cai[:, 2], 'g-', label='sim_cai_2019')
    ax.semilogy(1000 / sun[:, 0], sun[:, 2], 'y-', label='sim_sun_2017')

elif pode == 4:
    exp = loaddata_exp('OME4', reactorPressure, equivalence_ratio)
    ax.semilogy(exp[:, 0], exp[:, 1], 'bx', label='exp_PODE4')

    cai = np.flip(loaddata_sim(mechanism_all[1], nbr_run, equivalence_ratio, reactorPressure, pode), axis=0)
    ax.semilogy(1000 / cai[:, 0], cai[:, 2], 'g-', label='sim_cai_2019')

else:
    print('Entered PODE > 4 and not focus of this work')

ax.set_ylabel('IDT [ms]')
ax.set_xlabel('1000/T [1/K]')

# Add a second axis on top to plot the temperature for better readability
ax2 = ax.twiny()
ticks = ax.get_xticks()
ax2.set_xticks(ticks)
ax2.set_xticklabels((1000 / ticks).round(1))
ax2.set_xlim(ax.get_xlim())
ax2.set_xlabel('T [K]')

textstr = '$\\Phi$={:.1f}\np={:.0f}bar\nPODE{}'.format(equivalence_ratio, reactorPressure, pode)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top')

ax.set_yscale('log')
# ax.legend(bbox_to_anchor=(1, 0), loc="lower right", bbox_transform=fig.transFigure, ncol=3)

# ax.legend(bbox_to_anchor=(0, 1.07, 1, 0.7), loc="lower left", mode="expand", borderaxespad=0, ncol=3)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=4, prop={'size': 14})

plt.tight_layout()

path = Path(__file__).resolve()
path = path.parents[2] / 'data/00004-post-processing/delays_PODE{}_phi{}_p{}.pdf'\
    .format(pode, equivalence_ratio, reactorPressure)
plt.savefig(path)

plt.show()