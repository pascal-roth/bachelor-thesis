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
import pandas as pd
plt.style.use('stfs_2')


#%% get arguments
def parseArgs():
    parser = argparse.ArgumentParser(description="Create contour plot")

    parser.add_argument("-mech", "--mechanism_input", type=str, choices=['he', 'sun', 'cai'], default='cai',
                        help="chose reaction mechanism")

    parser.add_argument("-nbr_net", "--number_net", type=str, default='000',
                        help="chose number of the network")

    parser.add_argument("-phi", "--equivalence_ratio", type=float, default=1.0,
                        help="chose equivalence ratio")

    parser.add_argument("--pode", type=int, nargs='+', default=[3],
                        help="chose degree of polymerization")

    parser.add_argument("-p", "--pressure", nargs='+', type=int, default=[40],
                        help="chose reactor pressure")

    parser.add_argument("-inf_print", "--information_print", default=True, action='store_false',
                        help="chose if basic information are displayed")

    args = parser.parse_args()
    if args.information_print is True:
        print('\n{}\n'.format(args))

    return args


#%% functions
def get_y_samples(x_samples, x_scaler, y_samples, y_scaler):
    # output of the MLP
    x_samples_normalized, _ = normalize_df(x_samples, x_scaler)

    x_samples_normalized = torch.tensor(x_samples_normalized.values).float()
    model.eval()
    y_samples_nn_normalized = model.forward(x_samples_normalized)
    y_samples_nn_normalized = y_samples_nn_normalized.detach().numpy()

    y_samples_nn = y_scaler.inverse_transform(y_samples_nn_normalized)

    # Calculate the difference between reactor and MLP output
    # round y_sample after 2 decimals
    y_samples = np.round(y_samples, decimals=3)
    y_samples[np.abs(y_samples) < 0.001] = 0.001
    y_samples_nn = np.round(y_samples_nn, decimals=3)
    y_samples_nn[np.abs(y_samples_nn) < 0.001] = 0.001

    y_samples_max = np.amax(y_samples)
    y_samples_diff_abs = np.zeros(y_samples.shape)
    y_samples_diff_rel = (y_samples - y_samples_nn) / y_samples

    for i in range(y_samples.shape[1]):
        y_samples_diff_abs[:, i] = (y_samples.iloc[:, i] - y_samples_nn[:, i]) / y_samples_max[i]

    y_samples_diff_abs = pd.DataFrame(y_samples_diff_abs)
    return y_samples_nn, y_samples_diff_rel, y_samples_diff_abs


def create_grid(x_samples, y_samples, y_samples_nn, y_samples_diff_rel, y_samples_diff_abs):
    PV_max = np.amax(x_samples[['PV']])
    PV_min = np.amin(x_samples[['PV']])

    # x_samples[['H']] = x_samples[['H']].round(-3)
    H_max = np.amax(x_samples[['H']])
    H_min = np.amin(x_samples[['H']])

    grid_x, grid_y = np.mgrid[PV_min[0]:PV_max[0]:500j, H_min[0]:H_max[0]:500j]

    from scipy.interpolate import griddata

    grid_reactor = griddata(x_samples[['PV', 'H']].values, y_samples.values, (grid_x, grid_y), method='linear')
    grid_nn = griddata(x_samples[['PV', 'H']].values, y_samples_nn, (grid_x, grid_y), method='linear')

    grid_diff_rel = griddata(x_samples[['PV', 'H']].values, y_samples_diff_rel.values, (grid_x, grid_y),
                             method='linear')
    grid_diff_rel = np.squeeze(grid_diff_rel)
    grid_diff_abs = griddata(x_samples[['PV', 'H']].values, y_samples_diff_abs.values, (grid_x, grid_y),
                             method='linear')
    grid_diff_abs = np.squeeze(grid_diff_abs)

    grid_reactor = np.squeeze(grid_reactor)
    grid_nn = np.squeeze(grid_nn)

    return grid_x, grid_y, grid_reactor, grid_nn, grid_diff_rel, grid_diff_abs


def plotter(x_samples, y_samples_run, grid_x, grid_y, grid_reactor, grid_nn, grid_diff_rel, grid_diff_abs, number_net,
            label, equivalence_ratio, pressure):

    from matplotlib.backends.backend_pdf import PdfPages

    grid_y = grid_y / 1.e6               # scaling to MJ
    Z = x_samples[['Z']].iloc[0]

    if label == 'P':  # scale to MPa
        grid_reactor = grid_reactor / 1.e+6
        grid_nn = grid_nn / 1.e+6
        y_samples_run = y_samples_run / 1.e+6

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=[20, 5], dpi=300, sharex=True, sharey=True)

    # same scale of the contour plots by defining vmin and vmax
    vmin = np.amin(y_samples_run)
    vmax = np.amax(y_samples_run)

    if label == 'T':
        vstep = (np.round(vmax, decimals=-2) - np.round(vmin, decimals=-1)) / 10
        zticks = np.round(np.arange(vmin, vmax, vstep))
    elif label == 'P':
        vstep = (np.round(vmax, decimals=1) - np.round(vmin, decimals=1)) / 10
        zticks = np.round(np.arange(vmin, vmax, vstep), decimals=1)
    else:
        vstep = (np.round(vmax, decimals=3) - np.round(vmin, decimals=3)) / 10
        zticks = np.round(np.arange(vmin, vmax, vstep), decimals=3)

    label_unit = unit(label)

    img = axs[0].contourf(grid_x, grid_y, grid_reactor, levels=100, cmap='gist_rainbow', vmin=vmin, vmax=vmax)
    axs[0].set_xlabel('PV [-]')
    axs[0].set_ylabel('H [MJ/kg]')
    # fig.colorbar(img, ax=axs[0], label='{}'.format(label_unit))
    cbar_0 = fig.colorbar(img, ax=axs[0], label='{}'.format(label_unit))
    cbar_0.set_ticks(zticks)
    cbar_0.set_ticklabels(zticks)
    axs[0].set_title('HR Z={:.2f}'.format(Z[0]))

    img = axs[1].contourf(grid_x, grid_y, grid_nn, levels=100, cmap='gist_rainbow', vmin=vmin, vmax=vmax)
    axs[1].set_xlabel('PV [-]')
    cbar_1 = fig.colorbar(img, ax=axs[1], label='{}'.format(label_unit))
    cbar_1.set_ticks(zticks)
    cbar_1.set_ticklabels(zticks)
    axs[1].set_title('MLP Z={:.2f}'.format(Z[0]))

    grid_diff_rel = grid_diff_rel * 100  # scaling to percent
    grid_diff_abs = grid_diff_abs * 100  # scaling to percent

    img = axs[2].contourf(grid_x, grid_y, grid_diff_rel, levels=100, cmap='gist_rainbow')
    axs[2].set_xlabel('PV [-]')
    fig.colorbar(img, ax=axs[2], label='{} difference in %'.format(label_unit))
    axs[2].set_title('Relative Difference Z={:.2f}'.format(Z[0]))

    img = axs[3].contourf(grid_x, grid_y, grid_diff_abs, levels=100, cmap='gist_rainbow')
    axs[3].set_xlabel('PV [-]')
    fig.colorbar(img, ax=axs[3], label='{} difference in %'.format(label_unit))
    axs[3].set_title('Absolute Difference Z={:.2f}'.format(Z[0]))

    plt.tight_layout()

    path = Path(__file__).resolve()
    path_plt = PdfPages(path.parents[2] / 'data/00000-MLP/{}_plt_{}_contour_phi{}_p{}.pdf'.format \
        (number_net, label, equivalence_ratio, pressure[0]))

    plt.savefig(path_plt, format='pdf', bbox_inches='tight')

    plt.show()
    path_plt.close()


def unit(label):
    """ Corresponding unit for the selected label

    :parameter
    :param label:       - str -             label

    :returns:
    :return label_unit: - str -             label with corresponding unit
    """

    if label == 'P':
        label_unit = 'P [MPa]'
    elif label == 'T':
        label_unit = 'T [K]'
    elif label == 'HRR':
        label_unit = "HRR [W/$m^3$/kg]"
    else:
        label_unit = 'Y_{}'.format(label)

    return label_unit


#%%
if __name__ == "__main__":

    args = parseArgs()
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

    y_samples_nn, y_samples_diff_rel, y_samples_diff_abs = get_y_samples(x_samples, x_scaler, y_samples, y_scaler)

    for i, label in enumerate(labels):
        y_samples_run = y_samples.iloc[:, i]
        y_samples_nn_run = y_samples_nn[:, i]

        if y_samples_diff_rel is not None:
            y_samples_diff_rel_run = y_samples_diff_rel.iloc[:, i]
        else:
            y_samples_diff_rel_run = None

        if y_samples_diff_abs is not None:
            y_samples_diff_abs_run = y_samples_diff_abs.iloc[:, i]
        else:
            y_samples_diff_abs_run = None

        grid_x, grid_y, grid_reactor, grid_nn, grid_diff_rel, grid_diff_abs = create_grid(x_samples, y_samples_run,
                                                                                          y_samples_nn_run,
                                                                                          y_samples_diff_rel_run,
                                                                                          y_samples_diff_abs_run)

        plotter(x_samples, y_samples_run, grid_x, grid_y, grid_reactor, grid_nn, grid_diff_rel, grid_diff_abs, args.number_net, label,
                args.equivalence_ratio, args.pressure)
