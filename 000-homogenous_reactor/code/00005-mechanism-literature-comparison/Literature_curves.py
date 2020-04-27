#######################################################################################################################
# Plot the mechanism curves of the literature
#######################################################################################################################

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from pathlib import Path


# Load csv data
def loaddata(filename):
    path = Path(__file__).parents[2] / 'data/00005-mechanism-literature-comparison/curves-from-literature/{}.csv'\
        .format(filename)
    data = np.array(pd.read_csv(path, delimiter=';', decimal=','))
    return data


# taken from the Jacobs et al. Paper #######################################################################

he_2018_OME1_1_20 = loaddata('He_OME1_1_0_20bar')
sun_2017_OME1_1_20 = loaddata('Sun_OME1_1_0_20bar')
jacobs_2019_OME1_1_20 = loaddata('Jacobs_OME1_1_0_20bar_1')

# Plot the data
plt.plot(he_2018_OME1_1_20[:112, 0], he_2018_OME1_1_20[:112, 1], 'r-', label='lit_he_2018')
plt.plot(sun_2017_OME1_1_20[:97, 0], sun_2017_OME1_1_20[:97, 1], 'y-', label='lit_sun 2017')
plt.plot(jacobs_2019_OME1_1_20[:121, 0], jacobs_2019_OME1_1_20[:121, 1], 'b-', label='lit_jacobs_2019')
plt.yscale('log')
plt.legend()
plt.ylabel('Time (ms)')
plt.xlabel(r'$\frac{1000}{T (K)}$')
plt.title('Plot from Jacobs et al. with OME1, 20bar and $\Phi=1.0$ at $t_0$')
path = Path(__file__).parents[2] / 'data/00005-mechanism-literature-comparison/curves-from-literature/he-jacobs-sun.png'
plt.savefig(path)
plt.show()

# taken from the Cai et al. Paper #####################################################################################

CAI_OME2_05 = loaddata('Cai_05')
CAI_OME2_10 = loaddata('Cai_10')
CAI_OME2_20 = loaddata('Cai_20')

# Plot the data
plt.plot(CAI_OME2_05[:, 0], CAI_OME2_05[:, 1], 'r-', label='$\Phi$ = 0.5')
plt.plot(CAI_OME2_10[:163, 0], CAI_OME2_10[:163, 1], 'y-', label='$\Phi$ = 1.0')
plt.plot(CAI_OME2_20[:, 0], CAI_OME2_20[:, 1], 'b-', label='$\Phi$ = 2.0')
plt.yscale('log')
plt.legend()
plt.ylabel('Time (ms)')
plt.xlabel(r'$\frac{1000}{T (K)}$')
plt.title('Mechansim of Cai et al. with OME2 and 20bar as starting conditions')
path = Path(__file__).parents[2] / 'data/00005-mechanism-literature-comparison/curves-from-literature/cai.png'
plt.savefig(path)
plt.show()

# %% Plot the literature and my results
def loaddata_npy(filename):
    path_npy = Path(__file__).parents[2] / 'data/00005-mechanism-literature-comparison/simulated-delays/{}.npy'\
        .format(filename)
    data = np.load(path_npy)
    return data


he_2018_OME1_my = np.flip(loaddata_npy('delays_he_2018.xml_1.0_OME1'), axis=0)
sun_2017_OME1_my = np.flip(loaddata_npy('delays_sun_2017.xml_1.0_OME1'), axis=0)

he_2018_OME1_my[:, 2] = 1000 / he_2018_OME1_my[:, 2]
sun_2017_OME1_my[:, 2] = 1000 / sun_2017_OME1_my[:, 2]

# load the experimental data
exp_OME1_20_1_0_path = Path(__file__).parents[2] / 'data/00004-mechanism-exp-comparison/Exp_OME1_20_1_0.csv'
exp_OME1_20_1_0 = np.array(pd.read_csv(exp_OME1_20_1_0_path, delimiter=';', decimal=','))

# Plot the data
plt.plot(he_2018_OME1_1_20[:112, 0], he_2018_OME1_1_20[:112, 1], 'r-', label='lit_he_2018')
plt.plot(sun_2017_OME1_1_20[:97, 0], sun_2017_OME1_1_20[:97, 1], 'y-', label='lit_sun_2017')

plt.plot(he_2018_OME1_my[:, 2], he_2018_OME1_my[:, 4], 'r^-', label='sim_he_2018')
plt.plot(sun_2017_OME1_my[:, 2], sun_2017_OME1_my[:, 4], 'y^-', label='sim_sun_2017')

plt.plot(exp_OME1_20_1_0[:12, 0], exp_OME1_20_1_0[:12, 1], 'bx', label='exp_OME1')

plt.yscale('log')
plt.legend()
plt.ylabel('Time (ms)')
plt.xlabel(r'$\frac{1000}{T (K)}$')
plt.title('Plot comparison with OME1, 20bar and $\Phi=1.0$ at $t_0$')
path = Path(__file__).parents[2] / 'data/00005-mechanism-literature-comparison/plt-comparison-he-sun.png'
plt.savefig(path)
plt.show()