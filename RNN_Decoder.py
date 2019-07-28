# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

##########################
### MODEL
##########################

class RNN(torch.nn.Module):
    def __init__(self, latent_vector, last_stroke, last_hidden):
        super(RNN, self).__init__()
        self.fc1 = nn.Linear((latent_vector * last_stroke * hidden), 64)
        self.fc2 = nn.Linear(64, 13)


    def forward(self, x):
        hidden = self.fc1(x)
        output = self.fc2(hidden)
        return output, hidden



for input in inputs:
    ct, ht = LSTMCELL(ct, ht, input)
