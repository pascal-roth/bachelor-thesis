import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cantera as ct
ct.suppress_thermo_warnings()

# Load enthalpy of formation
h0 = pd.read_csv('enthalpies_of_formation.csv')
h0_mass = h0[['h0_mass']]
h0_mass = h0_mass.to_numpy()

# Load samples and show absolute enthalpy
samples_normal = pd.read_csv('004_train_samples.csv')
samples_normal.plot('time', 'H', style='b-')
plt.xlabel('time')
plt.ylabel('H')
plt.show()

# get initial and final absolute enthalpy
enthalpies = samples_normal[['H']]
H_0 = enthalpies.iloc[0]
H_end = enthalpies.iloc[len(enthalpies)-1]

# initialize cantera object
pode = ct.Solution('cai_ome14_2019.xml')
pode.basis = 'mass' # in that way enthalpy_mass can be set an initial state
# pode.set_equivalence_ratio(1.0, 'OME3', 'O2:{} N2:{}'.format(0.21, 0.79))

H_0 = H_0.to_numpy() + (np.sum(h0_mass * pode.Y))
H_end = H_end.to_numpy() + (np.sum(h0_mass * pode.Y))

pode.HPY = H_end, 20 * ct.one_atm, 'OME3: 0.141755 O2:0.1999 N2:0.658345'
print(pode.T)

pode.HPY = H_0, 20 * ct.one_atm, 'OME3: 0.141755 O2:0.1999 N2:0.658345'
print(pode.T)