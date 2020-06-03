import torch
import fc_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fc_pre_processing_load import loaddata
from torch import nn
from fc_post_processing import load_checkpoint
import fc_pre_processing_load
from pathlib import Path
import fc_post_processing
import fc_post_processing

mechanism_input = 'cai'
equivalence_ratio = [0.0]
pressure = [0]
temperature = [0]
pode = [0]
number_net = '000'
number_train_run = '001'
number_run = '000'
n_epochs = 2

# get the model
model, criterion, s_paras, l_paras, scaler_samples, scaler_labels, _, _, _, number_train_run = fc_post_processing.load_checkpoint(
    number_net)
print(model)

# get samples and labels not included in training data
train_samples, train_labels = fc_pre_processing_load.loaddata_samples(mechanism_input, number_train_run,
                                                                      equivalence_ratio, pressure,
                                                                      temperature, pode, category='train',
                                                                      s_paras=s_paras, l_paras=l_paras)

test_samples, test_labels = fc_pre_processing_load.loaddata_samples(mechanism_input, number_run,
                                                                    equivalence_ratio, pressure,
                                                                    temperature, pode, category='test',
                                                                    s_paras=s_paras, l_paras=l_paras)

train_samples, _ = fc_pre_processing_load.normalize_df(train_samples, scaler=scaler_samples)
train_labels, _ = fc_pre_processing_load.normalize_df(train_labels, scaler=scaler_labels)

test_samples, _ = fc_pre_processing_load.normalize_df(test_samples, scaler=scaler_samples)
test_labels, _ = fc_pre_processing_load.normalize_df(test_labels, scaler=scaler_labels)

# plot the output of NN and reactor together with the closest parameter in the training set (data between the
# interpolation took place)
fc_post_processing.plot_data(model, train_samples, train_labels, test_samples, test_labels, scaler_samples, scaler_labels, number_net, plt_nbrs=False)