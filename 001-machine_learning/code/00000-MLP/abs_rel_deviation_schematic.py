#######################################################################################################################
# create an plot which shows the absolute and relative deviations a species mass fraction curve
######################################################################################################################

# %% import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fc_pre_processing_load
from fc_post_processing import unit
plt.style.use('stfs_2')

equivalence_ratio = 1.0
pressure = 20
temperature = 950
pode = 3

mechanism_input = 'cai'
number_train_run = '000'
features = 'PV'
labels = 'H2O'

# Percentual ranges
abs_tol = 0.01
rel_tol = 0.05

feature_select = {'pode': [pode], 'phi': equivalence_ratio, 'P_0': [pressure], 'T_0': [temperature]}

x_samples, y_samples = fc_pre_processing_load.load_samples(mechanism_input, number_train_run, feature_select,
                                                           features, labels, select_data='include',
                                                           category='train')
x_samples = pd.DataFrame(x_samples)
y_samples = pd.DataFrame(y_samples)

x_samples.columns = [features]
y_samples.columns = [labels]

y_max_value = np.amax(y_samples)
y_max = np.ones(len(y_samples)) * y_max_value[0] * abs_tol

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
ax.plot(x_samples[features].values, y_samples[labels].values)
ax.fill_between(x_samples[features].values, y_samples[labels].values - y_max, y_samples[labels].values + y_max,
                alpha=0.2, color='b', label='abs tol')
ax.fill_between(x_samples[features].values, y_samples[labels].values * (1-rel_tol), y_samples[labels].values * (1+rel_tol),
                alpha=0.2, color='r', label='rel tol')

ax.set_xlabel('$Y_c$')
label_unit = unit(labels)
ax.set_ylabel('{}'.format(label_unit))

plt.title('PODE{} $\\Phi$={:.2f} p={}bar $T_0$={:.0f}K'.format(pode, equivalence_ratio, pressure, temperature))

plt.tight_layout()

plt.legend(loc='upper left')
plt.show()
