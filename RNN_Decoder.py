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

    def __init__(self, latent_vector, last_stroke, hidden):
        super(RNN, self).__init__()
        self.fc1 = nn.Linear((latent_vector * last_stroke * hidden), 13)



    def forward(self, x):
        x = self.fc1(x)
        return x
