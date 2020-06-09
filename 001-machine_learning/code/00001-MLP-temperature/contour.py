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
plt_scatter = True

if plt_scatter:
    x = x_samples[['PV']]
    y = x_samples[['H']] / 1.e+6
    z = y_samples

    fig, axs = plt.subplots(nrows=1, ncols=2)

    axs[0].scatter(x, y) #, s=area, c=colors, alpha=0.5)
    axs[0].set_title('Energy over OV')
    axs[0].set_xlabel('PV')
    axs[0].set_ylabel('H [MJ/kg]')

    axs[1].scatter(x, z) #, s=area, c=colors, alpha=0.5)
    axs[1].set_title('Temperature over PV')
    axs[1].set_xlabel('PV')
    axs[1].set_ylabel('T [K]')

    plt.show()

# %% Contour Plot
interpolate = True

if interpolate:
    PV_max = np.amax(x_samples[['PV']])
    PV_min = np.amin(x_samples[['PV']])

    x_samples[['H']] = x_samples[['H']].round(-3)
    U_max = np.amax(x_samples[['H']])
    U_min = np.amin(x_samples[['H']])

    grid_x, grid_y = np.mgrid[PV_min[0]:PV_max[0]:7000j, U_min[0]:U_max[0]:200j]

    from scipy.interpolate import griddata

    grid_reactor = griddata(x_samples[['PV', 'H']].values, y_samples.values, (grid_x, grid_y), method='linear')
    grid_nn = griddata(x_samples[['PV', 'H']].values, y_samples_nn, (grid_x, grid_y), method='linear')
    grid_diff = griddata(x_samples[['PV', 'H']].values, y_samples_diff.values, (grid_x, grid_y), method='linear')

    grid_reactor = np.squeeze(grid_reactor)
    grid_nn = np.squeeze(grid_nn)
    grid_diff = np.squeeze(grid_diff)

else:  # manually create grid
    Us = x_samples[['H']].round(-5)
    Us = Us.drop_duplicates()
    indexes = Us.index.values

    grid_reactor = np.zeros((len(Us), 7000))
    grid_nn = np.zeros((len(Us), 7000))
    grid_diff = np.zeros((len(Us), 7000))
    grid_x = np.zeros((len(Us), 7000))
    grid_y = np.zeros((len(Us), 7000))

    n = 0

    for i in range(len(Us)):

        if not i == len(Us)-1:
            n_samples = indexes[i+1] - indexes[i]
        else:
            n_samples = x_samples.index.max() - indexes[i]

        grid_reactor[i, :n_samples] = np.ravel(y_samples.iloc[n:(n+n_samples)])
        grid_nn[i, :n_samples] = np.ravel(y_samples_nn[n:(n+n_samples)])
        grid_diff[i, :n_samples] = np.ravel(y_samples_diff.iloc[n:(n+n_samples)])

        grid_x[i, :n_samples] = np.ravel(x_samples[['PV']].iloc[n:(n+n_samples)])
        grid_y[i, :n_samples] = np.ravel(x_samples[['H']].iloc[n:(n+n_samples)])

        n += n_samples

#%% create plot
grid_y = grid_y / 1.e6
Z = x_samples[['Z']].iloc[0]

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=[15, 5])

img = axs[0].contourf(grid_x, grid_y, grid_reactor, levels=100, cmap='gist_heat_r')
axs[0].set_xlabel('PV')
axs[0].set_ylabel('h [MJ/kg]')

fig.colorbar(img, ax=axs[0], label='T_reactor [K]')
axs[0].set_title('Z={:.2f}'.format(Z[0]))

img = axs[1].contourf(grid_x, grid_y, grid_nn, levels=100, cmap='gist_heat_r')
axs[1].set_xlabel('PV')
axs[1].set_ylabel('h [MJ/kg]')
fig.colorbar(img, ax=axs[1], label='T_MLP [K]')
axs[1].set_title('Z={:.2f}'.format(Z[0]))

img = axs[2].contourf(grid_x, grid_y, grid_diff, levels=100, cmap='gist_heat_r')
axs[2].set_xlabel('PV')
axs[2].set_ylabel('h [MJ/kg]')
fig.colorbar(img, ax=axs[2], label='T_diff [K]')
axs[2].set_title('Z={:.2f}'.format(Z[0]))

# plt.tight_layout()
plt.show()
