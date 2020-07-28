#######################################################################################################################
# Scatter and contour plot
#######################################################################################################################

# import packages
import argparse
from fc_pre_processing_load import load_samples, normalize_df
from fc_post_processing import load_checkpoint
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use('stfs')


#%% get arguments
def parseArgs():
    parser = argparse.ArgumentParser(description="Create contour plot")

    parser.add_argument("-mech", "--mechanism_input", type=str, choices=['he', 'sun', 'cai'], default='cai',
                        help="chose reaction mechanism")

    parser.add_argument("-nbr_net", "--number_net", type=str, default='001',
                        help="chose number of the network")

    parser.add_argument("-phi", "--equivalence_ratio", type=float, default=1.0,
                        help="chose equivalence ratio")

    parser.add_argument("--pode", type=int, nargs='+', default=[3],
                        help="chose degree of polymerization")

    parser.add_argument("-p", "--pressure", nargs='+', type=int, default=[20],
                        help="chose reactor pressure")

    parser.add_argument("-inf_print", "--information_print", default=True, action='store_false',
                        help="chose if basic information are displayed")

    args = parser.parse_args()
    if args.information_print is True:
        print('\n{}\n'.format(args))

    return args


#%% functions
def get_y_samples(x_samples, x_scaler, y_samples, percentage):
    # output of the MLP
    x_samples_normalized, _ = normalize_df(x_samples, x_scaler)
    x_samples_normalized = torch.tensor(x_samples_normalized.values).float()
    model.eval()
    y_samples_nn = model.forward(x_samples_normalized)
    y_samples_nn = y_samples_nn.detach().numpy()
    y_samples_nn = y_scaler.inverse_transform(y_samples_nn)

    # Calculate the difference between reactor and MLP output
    if percentage:
        y_samples_diff = (y_samples - y_samples_nn) / y_samples
    else:
        y_samples_diff = y_samples - y_samples_nn

    return y_samples_nn, y_samples_diff


def create_grid(x_samples, y_samples, y_samples_nn, y_samples_diff):
    PV_max = np.amax(x_samples[['PV']])
    PV_min = np.amin(x_samples[['PV']])

    # x_samples[['H']] = x_samples[['H']].round(-3)
    H_max = np.amax(x_samples[['H']])
    H_min = np.amin(x_samples[['H']])

    grid_x, grid_y = np.mgrid[PV_min[0]:PV_max[0]:7000j, H_min[0]:H_max[0]:200j]

    from scipy.interpolate import griddata

    grid_reactor = griddata(x_samples[['PV', 'H']].values, y_samples.values, (grid_x, grid_y), method='linear')
    grid_nn = griddata(x_samples[['PV', 'H']].values, y_samples_nn, (grid_x, grid_y), method='linear')
    grid_diff = griddata(x_samples[['PV', 'H']].values, y_samples_diff.values, (grid_x, grid_y), method='linear')

    grid_reactor = np.squeeze(grid_reactor)
    grid_nn = np.squeeze(grid_nn)
    grid_diff = np.squeeze(grid_diff)

    return grid_x, grid_y, grid_reactor, grid_nn, grid_diff


def plotter(x_samples, grid_x, grid_y, grid_reactor, grid_nn, grid_diff, number_net, label, equivalence_ratio,
            pressure, percentage):

    from matplotlib.backends.backend_pdf import PdfPages

    grid_y = grid_y / 1.e6
    Z = x_samples[['Z']].iloc[0]

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=[15, 5], dpi=300, sharex=True, sharey=True)

    img = axs[0].contourf(grid_x, grid_y, grid_reactor, levels=100, cmap='gist_heat_r')
    axs[0].set_xlabel('PV')
    axs[0].set_ylabel('h [MJ/kg]')
    fig.colorbar(img, ax=axs[0], label='Y_{}'.format(label))
    axs[0].set_title('Homogeneous Reactor Z={:.2f}'.format(Z[0]))

    img = axs[1].contourf(grid_x, grid_y, grid_nn, levels=100, cmap='gist_heat_r')
    axs[1].set_xlabel('PV')
#    axs[1].set_ylabel('h [MJ/kg]')
    fig.colorbar(img, ax=axs[1], label='Y_{}'.format(label))
    axs[1].set_title('MLP Z={:.2f}'.format(Z[0]))

    img = axs[2].contourf(grid_x, grid_y, grid_diff, levels=100, cmap='gist_heat_r')
    axs[2].set_xlabel('PV')
#    axs[2].set_ylabel('h [MJ/kg]')
    if percentage:
        fig.colorbar(img, ax=axs[2], label='{} difference in %'.format(label))
    else:
        fig.colorbar(img, ax=axs[2], label='{} difference'.format(label))
    axs[2].set_title('Difference Z={:.2f}'.format(Z[0]))

    plt.tight_layout()

    path = Path(__file__).resolve()
    path_plt = PdfPages(path.parents[2] / 'data/00001-MLP-temperature/{}_plt_{}_contour_{}_{}.pdf'.format \
        (number_net, label, equivalence_ratio, pressure[0]))

    plt.savefig(path_plt, format='pdf', bbox_inches='tight')

    plt.show()
    path_plt.close()


#%%
if __name__ == "__main__":

    args = parseArgs()
    percentage = True
    print('Load model ...')

    # Load MLP checkpoint --> get model, Hsed training set and features/labels
    model, criterion, features, labels, x_scaler, y_scaler, n_input, n_output, number_train_run = \
        load_checkpoint(args.number_net)

    print('Model loaded, load samples ...')

    # Load training set
    feature_select = {'pode': args.pode, 'phi': args.equivalence_ratio, 'P_0': args.pressure}

    x_samples, y_samples = load_samples(args.mechanism_input, number_train_run, feature_select, features=features,
                                        labels=labels, select_data='include', category='train')

    print('DONE!, Create Plot ...')

    y_samples_nn, y_samples_diff = get_y_samples(x_samples, x_scaler, y_samples, percentage)

    for i, label in enumerate(labels):
        y_samples_run = y_samples.iloc[:, i]
        y_samples_nn_run = y_samples_nn[:, i]
        y_samples_diff_run = y_samples_diff.iloc[:, i]

        grid_x, grid_y, grid_reactor, grid_nn, grid_diff = create_grid(x_samples, y_samples_run, y_samples_nn_run,
                                                                       y_samples_diff_run)

        plotter(x_samples, grid_x, grid_y, grid_reactor, grid_nn, grid_diff, args.number_net, label,
                args.equivalence_ratio, args.pressure, percentage)



# # %% Scatter plot
# plt_scatter = False
#
# if plt_scatter:
#     x = x_samples[['PV']]
#     y = x_samples[['H']] / 1.e+6
#     z = y_samples
#
#     fig, axs = plt.subplots(nrows=1, ncols=2)
#
#     axs[0].scatter(x, y) #, s=area, c=colors, alpha=0.5)
#     axs[0].set_title('Energy over OV')
#     axs[0].set_xlabel('PV')
#     axs[0].set_ylabel('H [MJ/kg]')
#
#     axs[1].scatter(x, z) #, s=area, c=colors, alpha=0.5)
#     axs[1].set_title('Temperature over PV')
#     axs[1].set_xlabel('PV')
#     axs[1].set_ylabel('T [K]')
#
#     plt.show()
