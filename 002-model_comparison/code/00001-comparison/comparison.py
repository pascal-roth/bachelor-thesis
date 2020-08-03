#######################################################################################################################
# MLP to predict the temperature for each value of the PV while only the input of the homogeneous reactor is given
#######################################################################################################################

# import packages
import argparse
from fc_NN_load import load_samples, load_dataloader, load_checkpoint
import pandas as pd
from pathlib import Path
import cantera as ct
import numpy as np


#%% get arguments
def parseArgs():
    parser = argparse.ArgumentParser(description="Create contour plot")

    parser.add_argument("-mech", "--mechanism_input", type=str, choices=['he', 'sun', 'cai'], default='cai',
                        help="chose reaction mechanism")

    parser.add_argument("--comparison", type=str, choices=['IDT', 'thermo', 'species'], default='IDT',
                        help="chose what should be compared")

    parser.add_argument("--pode", type=int, default=3,
                        help="chose degree of polymerization")

    parser.add_argument("-phi", "--equivalence_ratio", type=float, default=1.0,
                        help="chose equivalence ratio")

    parser.add_argument("-p", "--pressure", type=int, default=20,
                        help="chose reactor pressure")

    parser.add_argument("-T", "--temperature", type=int, default=680,
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

    if args.comparison == 'IDT':

    # Load MLP checkpoint --> get model, Hsed training set and features/labels
    model, criterion, features, labels, x_scaler, y_scaler, n_input, n_output, number_train_run = \
        load_checkpoint(args.number_net)

    print('Model loaded, load samples ...')

    # Load training set
    feature_select = {'pode': args.pode, 'phi': args.equivalence_ratio, 'P_0': args.pressure}

    x_samples, y_samples = load_samples(args.mechanism_input, number_train_run, feature_select, features=features,
                                        labels=labels, select_data='include', category='train')

    print('DONE!, Create Plot ...')


pode = 3
phi = 1
pressure = 40
temperature = 680


