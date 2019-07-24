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
        self.fc1 = nn.Linear((latent_vector * last_stroke * hidden), 128)
        self.fc2 = nn.Linear(128, 13)


    def forward(self, x):
        hidden = self.fc1(x)
        output = self.fc2(hidden)
        return output



'''
def LSTMCELL(prev_ct, prev_ht, input):
    combine = prev_ht + input
    ft = forget_layer(combine)
    candidate = candidate_layer(combine)
    it = input_layer(combine)
    Ct = prev_ct * ft + candidate * it
    ot = output_layer(combine)
    ht = ot * tanh(Ct)
    return ht, Ct


ct = [0,0,0]
ht = [0,0,0]

for input in inputs:
    ct, ht = LSTMCELL(ct, ht, input)

'''
