# Imports

import torch
import torch.nn as nn


###################
# model ###########
###################


class RNN(nn.Module):
    def __init__(self, latent_C, stroke, hidden, bias=True):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.FC = nn.Linear(, 4 * hidden_size, bias = bias)
