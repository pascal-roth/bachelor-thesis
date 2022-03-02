import argparse
import pandas as pd
import matplotlib.pyplot as plt
import fc_pre_processing_load
from pathlib import Path
import fc_model
import torch
import numpy as np
from torch import nn
plt.style.use('stfs')

def load_checkpoint(epoch):
    """Load Checkpointing of a saved model

    :parameter
    :param nbr_net: - int -     number to identify the saved MLP
    """

    path = Path(__file__).resolve()
    path_pth = path.parents[2] / 'data/00000-MLP/gif_checkpoint_{}.pth'.format(epoch)
    checkpoint = torch.load(path_pth, map_location=torch.device('cpu'))

    # Create model and load its criterion
    model = fc_model.Net(checkpoint['input_size'],
                         checkpoint['output_size'],
                         checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    criterion = nn.MSELoss()
    criterion.load_state_dict(checkpoint['criterion'])

    # load the parameters of the x_samples and y_samples, as well as the corresponding MinMaxScalers
    number_train_run = checkpoint['number_train_run']
    features = checkpoint['features']
    labels = checkpoint['labels']
    x_scaler = checkpoint['x_scaler']
    y_scaler = checkpoint['y_scaler']

    # parameters needed to save the model again
    n_input = checkpoint['input_size']
    n_output = checkpoint['output_size']

    return model, criterion, features, labels, x_scaler, y_scaler, n_input, n_output, \
           number_train_run


path = Path(__file__).resolve()
path_loss = path.parents[2] / 'data/00000-MLP/gif_10_losses.csv'
losses = pd.read_csv(path_loss)
losses = losses.to_numpy()

loss_max = np.max(losses[:, 1])
loss_min = np.min(losses[:, 1])

epochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
mechanism_input = 'cai'
number_test_run = '001'

for i, epoch_run in enumerate(epochs):
    model, _, features, labels, x_scaler, y_scaler, _, _, number_train_run = load_checkpoint(epoch_run)

    feature_select = {}

    x_test, y_test = fc_pre_processing_load.load_samples(mechanism_input, number_test_run,
                                                         feature_select, features, labels,
                                                         select_data='exclude', category='test')

    #  Load  test tensors
    test_loader = fc_pre_processing_load.load_dataloader(x_test, y_test, batch_fraction=100, split=False,
                                                         x_scaler=x_scaler, y_scaler=y_scaler, features=None)

    x_test, _ = fc_pre_processing_load.normalize_df(x_test, scaler=x_scaler)

    # transform data to tensor and compute output of MLP
    samples_tensor = torch.tensor(x_test.values).float()
    model.eval()
    output = model(samples_tensor)

    # denormalize output for plotting
    output = output.detach().numpy()
    output = y_scaler.inverse_transform(output)

    y_test = np.squeeze(y_test[['CO']])
    output = np.squeeze(output[:, 1])

    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(9, 6))

    # plot the MLP and reactor output
    ax2.plot(x_test[['PV']], y_test, 'b-', label='HR Output')
    ax2.plot(x_test[['PV']], output, 'r-', label='NN Output')
    ax2.set_xlabel('PV [-]')
    ax2.set_ylabel('Y_CO [-]')
    ax2.set_ylim(-0.03, 0.14)
    ax2.legend()

    ax1.plot(losses[:epoch_run, 0], losses[:epoch_run, 1], 'b-', label='training_loss')
    ax1.plot(losses[:epoch_run, 0], losses[:epoch_run, 2], 'r-', label='valid_loss')
    ax1.set_xlabel('Epochs [-]')
    ax1.set_ylabel('loss [-]')
    ax1.set_yscale('log')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(loss_min, loss_max)
    ax1.legend()

    plt.tight_layout()

    path = Path(__file__).parents[2] / 'data/evolution/plot_{}.png'.format(epoch_run)
    plt.savefig(path)

    plt.show()