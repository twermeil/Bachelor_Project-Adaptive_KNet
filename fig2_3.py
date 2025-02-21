import torch
from filters.KalmanFilter_test import KFTest
from simulations.Linear_sysmdl import SystemModel

#data loading
data1 = torch.load("ADAPTIVE-KNET_ICASSP24/data/Linear_CA/r2=0.1_q2=1.0.pt")
data2 = torch.load("ADAPTIVE-KNET_ICASSP24/data/Linear_CA/r2=0.01_q2=1.0.pt")
data3 = torch.load("ADAPTIVE-KNET_ICASSP24/data/Linear_CA/r2=1.0_q2=1.0.pt")
data4 = torch.load("ADAPTIVE-KNET_ICASSP24/data/Linear_CA/r2=10.0_q2=1.0.pt")



