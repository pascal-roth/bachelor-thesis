#######################################################################################################################
# MLP to predict the temperature for each value of the PV while only the input of the homogeneous reactor is given
#######################################################################################################################

# import packages
import argparse
import numpy as np

from fc_NN import load_checkpoint, NN_output
from fc_HR import load_samples
from fc_GRM import load_IDTs, load_GRM_data
from fc_plot import plot_IDT, plot_outputs


#%% get arguments
def parseArgs():
    parser = argparse.ArgumentParser(description="Create contour plot")

    parser.add_argument("-mech", "--mechanism_input", type=str, choices=['he', 'sun', 'cai'], default='cai',
                        help="chose reaction mechanism")

    parser.add_argument("--comparison", type=str, choices=['IDT', 'thermo', 'educts', 'products'], default='thermo',
                        help="chose what should be compared")

    parser.add_argument("--pode", type=int, default=3,
                        help="chose degree of polymerization")

    parser.add_argument("-phi", "--equivalence_ratio", type=float, choices=[0.5, 0.67, 1.0], default=1.0,
                        help="chose equivalence ratio")

    parser.add_argument("-p", "--pressure", type=int, default=40,
                        help="chose reactor pressure")

    parser.add_argument("-T", "--temperature", type=int, default=720,
                        help="chose reactor pressure")

    parser.add_argument("-inf_print", "--information_print", default=True, action='store_false',
                        help="chose if basic information are displayed")

    args = parser.parse_args()
    if args.information_print is True:
        print('\n{}\n'.format(args))

    return args


if __name__ == "__main__":

    args = parseArgs()
    print('Load model ...')

    # set parameters
    batch_fraction = 100
    nbr_net_thermo = '012'
    nbr_net_products = '013'
    nbr_net_educts = '014'
    nbr_net_HRR = '015'

    if args.equivalence_ratio == 1.0:
        phi = int(1)
    else:
        phi = args.equivalence_ratio

    if args.comparison == 'IDT':
        # load model with features=['pode', 'Z', 'H', 'PV'] and labels=[HRR]
        model, criterion, features, labels, x_scaler, y_scaler, n_input, n_output, _ = load_checkpoint(nbr_net_HRR)
        print('Model loaded, Calculate IDTs of MLP, GRM and HR (as reference)...')

        IDT_MLP_PV = np.zeros((15, 2))
        IDT_GRM_PV = np.zeros((15, 2))
        IDT_HR_PV = np.zeros((15, 2))

        # iterated through all temperatures given
        for i, temperature_run in enumerate(np.arange(680, 1240 + 40, 40)):

            # get samples of HR
            feature_select = {'pode': [args.pode], 'phi': args.equivalence_ratio, 'P_0': [args.pressure],
                              'T_0': [temperature_run]}
            x_samples, y_samples = load_samples(args.mechanism_input, nbr_run='002', feature_select=feature_select,
                                                features=features, labels=labels, select_data='include', category='test')

            # IDT of MLP ##############################################################################################
            y_samples_nn = NN_output(model, x_samples, x_scaler, y_scaler)
            IDT_MLP_location = np.argmax(y_samples_nn)
            IDT_MLP_PV[i, :] = (temperature_run, x_samples['PV'].iloc[IDT_MLP_location])

            # IDT of GRM ##############################################################################################
            IDT_GRM_PV_run = load_IDTs(args.pode, phi, args.pressure, temperature_run)
            IDT_GRM_PV[i, :] = (temperature_run, IDT_GRM_PV_run)

            # IDT of HR ###############################################################################################
            IDT_HR_location = np.argmax(y_samples)
            IDT_HR_PV[i, :] = (temperature_run, x_samples['PV'].iloc[IDT_HR_location])

        print('DONE! Create plot ...')

        plot_IDT(IDT_MLP_PV, IDT_GRM_PV, IDT_HR_PV, args)

    else:

        if args.comparison == 'thermo':
            nbr_net = nbr_net_thermo    # load model with features=['pode', 'Z', 'H', 'PV'] and labels=[T, P]
        elif args.comparison == 'products':
            nbr_net = nbr_net_products  # load model with features=['pode', 'Z', 'H', 'PV'] and labels=[CO, CO2, H2O]
        elif args.comparison == 'educts':
            nbr_net = nbr_net_educts    # load model with features=['pode', 'Z', 'H', 'PV'] and labels=[PODE, O2]
        else:
            print('WARNING! Entered comparison not valid')

        model, criterion, features, labels, x_scaler, y_scaler, n_input, n_output, _ = load_checkpoint(nbr_net)
        print('Model loaded, Calculate temperature and pressure values of MLP, GRM ...')

        # get samples of HR
        feature_select = {'pode': [args.pode], 'phi': args.equivalence_ratio, 'P_0': [args.pressure],
                          'T_0': [args.temperature]}
        x_samples, y_samples = load_samples(args.mechanism_input, nbr_run='002', feature_select=feature_select,
                                            features=features, labels=labels, select_data='include', category='test')

        # temperature and pressure values of the MLP
        y_samples_nn = NN_output(model, x_samples, x_scaler, y_scaler)

        # temperature and pressure values of the GRM
        samples_grm = load_GRM_data(args.pode, phi, args.pressure, args.temperature)

        plot_outputs(y_samples, y_samples_nn, samples_grm, x_samples, features, labels)
