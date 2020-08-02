#######################################################################################################################
# MLP to predict the temperature for each value of the PV while only the input of the homogeneous reactor is given
#######################################################################################################################

# import packages
import argparse
from fc_NN_load import load_samples, load_dataloader, load_checkpoint
import pandas as pd
from pathlib import Path

pode = 3
phi = 1
pressure = 40
temperature = 680

path = Path(__file__).parents[1] / 'data/00000-global-reactor/OME{}_phi{}_p{}_T{}.txt'.\
        format(pode, phi, pressure, temperature)
data = pd.read_csv(path, sep=" |,", engine='python')



