"""
This file contains the parameters for the simulations with real data
* Constant Velocity Model (CV)
"""

import torch

nlb_sr = 100 #nlb sampling rate is 100 Hz
delta_t = 1/nlb_sr

#use constant velocity model for F
F_CV = torch.tensor([
    [1, 0, delta_t, 0],
    [0, 1, 0, delta_t],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]).float()