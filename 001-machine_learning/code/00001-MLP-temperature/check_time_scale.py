#######################################################################################################################
# Check if time can be rescaled
#######################################################################################################################

# import packages
from fc_pre_processing_load import load_samples, normalize_df, denormalize_df
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('stfs')

# parameters
mechanism_input = 'cai'
number_train_run = '000'
number_test_run = '001'
feature_select = {'P_0': [20]}
features = ['time', 'PV']
labels = ['P', 'T']

# load train and test samples
x_train, _ = load_samples(mechanism_input, number_train_run, feature_select, features, labels,
                          select_data='include', category='train')

x_test, _ = load_samples(mechanism_input, number_test_run, feature_select, features, labels,
                         select_data='include', category='test')

# normalize the samples
x_train_norm, x_scaler = normalize_df(x_train, scaler=None)

x_test_norm, _ = normalize_df(x_test, scaler=x_scaler)

# denormalize samples and compare the time
x_train_denorm = denormalize_df(x_test_norm, scaler=x_scaler)

x_test_denorm = denormalize_df(x_test_norm, scaler=x_scaler)

# show difference between the times

train_diff = x_train['time'] - x_train_denorm['time']
test_diff = x_test['time'] - x_test_denorm['time']

print('Maximum difference for the train samples: {:6.5e}'.format(np.amax(train_diff)))
print('Maximum difference for the test  samples: {:6.5e}'.format(np.amax(test_diff)))

