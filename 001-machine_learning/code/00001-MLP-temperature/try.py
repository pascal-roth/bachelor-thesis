import argparse
import torch
import fc_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fc_pre_processing_load
from torch import nn
from pathlib import Path
import fc_post_processing

mechanism_input = 'cai'
number_run = '000'
equivalence_ratio = [1.0]
pressure = [20]
temperature = [740]
pode = [3]
number_net = '001'

model, criterion = fc_post_processing.load_checkpoint(number_net)
print(model)

samples, labels, scaler_samples, scaler_labels = fc_pre_processing_load.loaddata_samples(mechanism_input, number_run,
                                                                                         equivalence_ratio=[0], reactorPressure=[0], reactorTemperature=[0], pode=[0],
                                                                                         category='train')

fc_post_processing.plot_train(model, samples, labels, scaler_samples, scaler_labels, number_net,
                                pode, equivalence_ratio, pressure, temperature)