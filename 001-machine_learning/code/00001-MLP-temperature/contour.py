#######################################################################################################################
# Scatter and contour plot
#######################################################################################################################

# import packages
import argparse
import matplotlib.pyplot as plt
from fc_pre_processing_load import load_samples, normalize_df
from fc_post_processing import load_checkpoint
import torch
import numpy as np
import cantera as ct

#%% get arguments
# parser = argparse.ArgumentParser(description="Create contour plot")
#
# parser.add_argument("-nbr_net", "--number_net", type=str, default='000',
#                     help="chose number of the network")
#
# parser.add_argument("-phi", "--equivalence_ratio", nargs='+', type=float, default=[0.0],
#                     help="chose equivalence ratio")
#
# parser.add_argument("-Z", "--mixture_fraction", type=float, default=[0.0],
#                     help="chose mixture fraction")
#
# parser.add_argument("--pode", type=int, nargs='+', default=[0],
#                     help="chose degree of polymerization")
#
# parser.add_argument("-inf_print", "--information_print", default=True, action='store_false',
#                     help="chose if basic information are displayed")
#
# args = parser.parse_args()
# if args.information_print is True:
#     print('\n{}\n'.format(args))


mechanism_input = 'cai'
equivalence_ratio = [1.0]
pode = [3]
number_net = '002'

# Load MLP checkpoint --> get model, used training set and features/labels
model, criterion, features, labels, x_scaler, y_scaler, n_input, n_output, number_train_run = \
    load_checkpoint(number_net)

# Load training set
feature_select = {'pode': pode, 'phi': equivalence_ratio, 'P_0': [20 * ct.one_atm]}


x_samples, y_samples = load_samples(mechanism_input, number_train_run, feature_select, features=features,
                                    labels=labels, select_data='include', category='train')

# output of the MLP
x_samples_normalized, _ = normalize_df(x_samples, x_scaler)
x_samples_normalized = torch.tensor(x_samples_normalized.values).float()
model.eval()
y_samples_nn = model.forward(x_samples_normalized)
y_samples_nn = y_samples_nn.detach().numpy()
y_samples_nn = y_scaler.inverse_transform(y_samples_nn)

# Calculate the difference between reactor and MLP output
y_samples_diff = y_samples - y_samples_nn

# %% Scatter plot
plt_scatter = False

if plt_scatter:
    x = x_samples[['PV']]
    y = y_samples

    plt.scatter(x, y) #, s=area, c=colors, alpha=0.5)
    plt.title('Scatter plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# %% Contour Plot
Us = x_samples[['U']].round(-5)
Us = Us.drop_duplicates()
indexes = Us.index.values

z_reactor = np.zeros((len(Us), 7000))
z_nn = np.zeros((len(Us), 7000))
z_diff = np.zeros((len(Us), 7000))
PV = np.zeros((len(Us), 7000))
U = np.zeros((len(Us), 7000))

n = 0

for i in range(len(Us)):

    if not i == len(Us)-1:
        n_samples = indexes[i+1] - indexes[i]
    else:
        n_samples = x_samples.index.max() - indexes[i]

    z_reactor[i, :n_samples] = np.ravel(y_samples.iloc[n:(n+n_samples)])
    z_nn[i, :n_samples] = np.ravel(y_samples_nn[n:(n+n_samples)])
    z_diff[i, :n_samples] = np.ravel(y_samples_diff.iloc[n:(n+n_samples)])

    PV[i, :n_samples] = np.ravel(x_samples[['PV']].iloc[n:(n+n_samples)])
    U[i, :n_samples] = np.ravel(x_samples[['U']].iloc[n:(n+n_samples)])

    n += n_samples

#%%
# PV, U = np.meshgrid(PV, U, sparse=True)   # no orthogonal separation in the PV

U = U / 1.e+6
Z = x_samples[['Z']].iloc[0]

fig, axs = plt.subplots(nrows=1, ncols=3)

img = axs[0].contourf(PV, U, z_reactor) #, levels=100, cmap='RdGy_r')
axs[0].set_xlabel('PV')
axs[0].set_ylabel('U [MJ]')
axs[0].ticklabel_format(axis='both', style='sci', scilimits=[0, 10], useOffset=None, useLocale=None, useMathText=True)
fig.colorbar(img, ax=axs[0], label='T [K]')
axs[0].set_title('Z={:.2f}'.format(Z[0]))

img = axs[1].contourf(PV, U, z_nn) #, levels=100, cmap='RdGy_r')
axs[1].set_xlabel('PV')
axs[1].set_ylabel('U [MJ]')
axs[1].ticklabel_format(axis='both', style='sci', scilimits=[0, 10], useOffset=None, useLocale=None, useMathText=True)
fig.colorbar(img, ax=axs[1], label='T [K]')
axs[1].set_title('Z={:.2f}'.format(Z[0]))

img = axs[2].contourf(PV, U, z_diff) #, levels=100)
axs[2].set_xlabel('PV')
axs[2].set_ylabel('h_mean [MJ]')
axs[2].ticklabel_format(axis='both', style='sci', scilimits=[0, 10], useOffset=None, useLocale=None, useMathText=True)
fig.colorbar(img, ax=axs[2], label='T [K]')
axs[2].set_title('Z={:.2f}'.format(Z[0]))

# plt.tight_layout()
plt.show()
