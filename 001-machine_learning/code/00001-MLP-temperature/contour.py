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

    parser.add_argument("-diff", "--abs_rel_difference", type=str, choices=['abs', 'rel', 'both'], default='both',
                        help="chose if absolute and/or relative difference is plotted")

    parser.add_argument("-inf_print", "--information_print", default=True, action='store_false',
                        help="chose if basic information are displayed")

    args = parser.parse_args()
    if args.information_print is True:
        print('\n{}\n'.format(args))

    return args


#%% functions
def get_y_samples(x_samples, x_scaler, y_samples, diff):
    # output of the MLP
    x_samples_normalized, _ = normalize_df(x_samples, x_scaler)
    x_samples_normalized = torch.tensor(x_samples_normalized.values).float()
    model.eval()
    y_samples_nn = model.forward(x_samples_normalized)
    y_samples_nn = y_samples_nn.detach().numpy()
    y_samples_nn = y_scaler.inverse_transform(y_samples_nn)

    # Calculate the difference between reactor and MLP output
    if diff == 'rel':
        y_samples_diff_rel = (y_samples - y_samples_nn) / y_samples
        y_samples_diff_abs = None
    elif diff == 'abs':
        y_samples_diff_abs = y_samples - y_samples_nn
        y_samples_diff_rel = None
    else:
        y_samples_diff_abs = y_samples - y_samples_nn
        y_samples_diff_rel = (y_samples - y_samples_nn) / y_samples

    return y_samples_nn, y_samples_diff_rel, y_samples_diff_abs


def create_grid(x_samples, y_samples, y_samples_nn, y_samples_diff_rel, y_samples_diff_abs, diff):
    PV_max = np.amax(x_samples[['PV']])
    PV_min = np.amin(x_samples[['PV']])

    # x_samples[['H']] = x_samples[['H']].round(-3)
    H_max = np.amax(x_samples[['H']])
    H_min = np.amin(x_samples[['H']])

    grid_x, grid_y = np.mgrid[PV_min[0]:PV_max[0]:7000j, H_min[0]:H_max[0]:200j]

    from scipy.interpolate import griddata

    grid_reactor = griddata(x_samples[['PV', 'H']].values, y_samples.values, (grid_x, grid_y), method='linear')
    grid_nn = griddata(x_samples[['PV', 'H']].values, y_samples_nn, (grid_x, grid_y), method='linear')

    if diff == 'rel':
        grid_diff_rel = griddata(x_samples[['PV', 'H']].values, y_samples_diff_rel.values, (grid_x, grid_y),
                                 method='linear')
        grid_diff_rel = np.squeeze(grid_diff_rel)
        grid_diff_abs = None
    elif diff == 'abs':
        grid_diff_abs = griddata(x_samples[['PV', 'H']].values, y_samples_diff_abs.values, (grid_x, grid_y),
                                 method='linear')
        grid_diff_abs = np.squeeze(grid_diff_abs)
        grid_diff_rel = None
    else:
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
            label, equivalence_ratio, pressure, diff):

    from matplotlib.backends.backend_pdf import PdfPages

    grid_y = grid_y / 1.e6
    Z = x_samples[['Z']].iloc[0]

    if diff == 'both':
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=[20, 5], dpi=300, sharex=True, sharey=True)
    else:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=[15, 5], dpi=300, sharex=True, sharey=True)

    # same scale of the contour plots by defining vmin and vmax
    vmin = np.amin(y_samples_run)
    vmax = np.amax(y_samples_run)

    img = axs[0].contourf(grid_x, grid_y, grid_reactor, levels=100, cmap='gist_rainbow', vmin=vmin, vmax=vmax)
    axs[0].set_xlabel('PV')
    axs[0].set_ylabel('h [MJ/kg]')
    fig.colorbar(img, ax=axs[0], label='Y_{}'.format(label))
    axs[0].set_title('Homogeneous Reactor Z={:.2f}'.format(Z[0]))

    img = axs[1].contourf(grid_x, grid_y, grid_nn, levels=100, cmap='gist_rainbow', vmin=vmin, vmax=vmax)
    axs[1].set_xlabel('PV')
    fig.colorbar(img, ax=axs[1], label='Y_{}'.format(label))
    axs[1].set_title('MLP Z={:.2f}'.format(Z[0]))

    if diff == 'rel':
        img = axs[2].contourf(grid_x, grid_y, grid_diff_rel, levels=100, cmap='gist_rainbow')
        axs[2].set_xlabel('PV')
        fig.colorbar(img, ax=axs[2], label='{} difference in %'.format(label))
        axs[2].set_title('Difference Z={:.2f}'.format(Z[0]))
    elif diff == 'abs':
        img = axs[2].contourf(grid_x, grid_y, grid_diff_abs, levels=100, cmap='gist_rainbow')
        axs[2].set_xlabel('PV')
        fig.colorbar(img, ax=axs[2], label='{} difference'.format(label))
        axs[2].set_title('Difference Z={:.2f}'.format(Z[0]))
    else:
        img = axs[2].contourf(grid_x, grid_y, grid_diff_rel, levels=100, cmap='gist_rainbow')
        axs[2].set_xlabel('PV')
        fig.colorbar(img, ax=axs[2], label='{} difference in %'.format(label))
        axs[2].set_title('Relative Difference Z={:.2f}'.format(Z[0]))

        img = axs[3].contourf(grid_x, grid_y, grid_diff_abs, levels=100, cmap='gist_rainbow')
        axs[3].set_xlabel('PV')
        fig.colorbar(img, ax=axs[3], label='{} difference'.format(label))
        axs[3].set_title('Absolute Difference Z={:.2f}'.format(Z[0]))

    plt.tight_layout()

    path = Path(__file__).resolve()
    path_plt = PdfPages(path.parents[2] / 'data/00001-MLP-temperature/{}_plt_{}_contour_phi{}_p{}.pdf'.format \
        (number_net, label, equivalence_ratio, pressure[0]))

    plt.savefig(path_plt, format='pdf', bbox_inches='tight')

    plt.show()
    path_plt.close()


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

    y_samples_nn, y_samples_diff_rel, y_samples_diff_abs = get_y_samples(x_samples, x_scaler, y_samples,
                                                                         args.abs_rel_difference)

    for i, label in enumerate(labels):
        y_samples_run = y_samples.iloc[:, i]
        y_samples_nn_run = y_samples_nn[:, i]
        y_samples_diff_rel_run = y_samples_diff_rel.iloc[:, i]
        y_samples_diff_abs_run = y_samples_diff_abs.iloc[:, i]

        grid_x, grid_y, grid_reactor, grid_nn, grid_diff_rel, grid_diff_abs = create_grid(x_samples, y_samples_run,
                                                                                          y_samples_nn_run,
                                                                                          y_samples_diff_rel_run,
                                                                                          y_samples_diff_abs_run,
                                                                                          args.abs_rel_difference)

        plotter(x_samples, y_samples_run, grid_x, grid_y, grid_reactor, grid_nn, grid_diff_rel, grid_diff_abs, args.number_net, label,
                args.equivalence_ratio, args.pressure, args.abs_rel_difference)
