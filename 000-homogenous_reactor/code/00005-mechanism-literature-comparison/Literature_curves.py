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
    path = Path(__file__).parents[2] / 'data/00005-mechanism-literature-comparison/{}.csv'.format(filename)
    data = np.array(pd.read_csv(path, delimiter=';', decimal=','))
    return data


# taken from the Jacobs et al. Paper #######################################################################

he_2018_OME1_1_20 = loaddata('He_OME1_1_0_20bar')
sun_2017_OME1_1_20 = loaddata('Sun_OME1_1_0_20bar')
jacobs_2019_OME1_1_20 = loaddata('Jacobs_OME1_1_0_20bar_1')

# Plot the data
plt.plot(he_2018_OME1_1_20[:112, 0], he_2018_OME1_1_20[:112, 1], 'r-', label='He 2018')
plt.plot(sun_2017_OME1_1_20[:97, 0], sun_2017_OME1_1_20[:97, 1], 'y-', label='Sun 2017')
plt.plot(jacobs_2019_OME1_1_20[:121, 0], jacobs_2019_OME1_1_20[:121, 1], 'b-', label='Jacobs 2019')
plt.yscale('log')
plt.legend()
plt.ylabel('Time (ms)')
plt.xlabel(r'$\frac{1000}{T (K)}$')
plt.title('Plot from Jacobs et al. with OME1, 20bar and $\Phi=1.0$ at $t_0$')
path = Path(__file__).parents[2] / 'data/00005-mechanism-literature-comparison/he-jacobs-sun.png'
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
path = Path(__file__).parents[2] / 'data/00005-mechanism-literature-comparison/cai.png'
plt.savefig(path)
plt.show()
