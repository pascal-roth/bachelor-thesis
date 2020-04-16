#######################################################################################################################
# Compare the different detailed mechanism
#######################################################################################################################

# Import packages
import numpy as np
import matplotlib.pyplot as plt

# %% Load arrays
mechanism_all = np.array([['he_2018.xml', 'DMM3'], ['cai_ome14_2019.xml', 'OME3'], ['sun_2017.xml', 'DMM3']])
path = '/media/pascal/DATA/000-Homogeneous-Reactor/delays_{mechansim}.npy'.format

delays_he = np.load(path(mechanism=mechanism_all[0, 0]))
delays_cai = np.load(path(mechanism=mechanism_all[1, 0]))
delays_sun = np.load(path(mechanism=mechanism_all[0, 0]))


# %% compare ignition delays, Cai mechanism as object to compare to

if delays_he.shape == delays_cai.shape:
    diff = delays_he[:, 3:] - delays_cai[:, 3:]
    diff_sum = np.sum(diff, axis=1)

