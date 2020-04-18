#######################################################################################################################
# Compare the different detailed mechanism
#######################################################################################################################

# Import packages
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# %% Load arrays
mechanism_all = np.array([['he_2018.xml', 'DMM3'], ['cai_ome14_2019.xml', 'OME3'], ['sun_2017.xml', 'DMM3']])
path = Path(__file__).parents[2] / 'data/00002-reactor-OME/delays_{mechanism}_{equivalence_ratio}.npy'.format

delays_he_1_0_20 = np.load(path(mechanism=mechanism_all[0, 0], equivalence_ratio=1.0))
delays_cai = np.load(path(mechanism=mechanism_all[1, 0]))
delays_sun = np.load(path(mechanism=mechanism_all[0, 0]))


# %% compare ignition delays, Cai mechanism as object to compare to

if delays_he.shape == delays_cai.shape:
    diff_1 = delays_he[:, 3:] - delays_cai[:, 3:]
    diff_sum_1 = np.sum(diff_1, axis=1)

    plt.plot(diff_1[:, 0], label='first ignition')
    plt.plot(diff_1[:, 2], label='main ignition')
    plt.plot(diff_sum_1, label='sum both')
    plt.legend()
    plt.show
else:
    print('The overall number of ignitions in the interval is not the same')

if delays_sun.shape == delays_cai.shape:
    diff_2 = delays_sun[:, 3:] - delays_cai[:, 3:]
    diff_sum_2 = np.sum(diff_2, axis=1)

    plt.plot(diff_2[:, 0], label='first ignition')
    plt.plot(diff_2[:, 2], label='main ignition')
    plt.plot(diff_sum_2, label='sum both')
    plt.legend()
    plt.show
else:
    print('The overall number of ignitions in the interval is not the same')

