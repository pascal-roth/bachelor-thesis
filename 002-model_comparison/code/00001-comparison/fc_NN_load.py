#######################################################################################################################
# Functions to load the MLP model
#######################################################################################################################

# import packages #####################################################################################################
from torch import nn
import torch
import torch.nn.functional as F
from pathlib import Path


# model building class ################################################################################################
class Net(nn.Module):
    def __init__(self, n_input, n_output, hidden_layers):
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(n_input, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], n_output)

        self.dropout = nn.Dropout(p=0)

    def forward(self, x):
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return x


# load checkpoint ######################################################################################################
def load_checkpoint(nbr_net):
    """Load Checkpoint of a saved model

    :parameter
    :param nbr_net: - int -     number to identify the saved MLP
    """

    path = Path(__file__).resolve()
    path_pth = path.parents[2] / 'data/00001-MLP-temperature/{}_checkpoint.pth'.format(nbr_net)
    checkpoint = torch.load(path_pth, map_location=torch.device('cpu'))

    # Create model and load its criterion
    model = Net(checkpoint['input_size'], checkpoint['output_size'], checkpoint['hidden_layers'])
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
