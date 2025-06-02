import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from datetime import datetime

from simulations.Linear_sysmdl import SystemModel
from simulations.utils import SplitData, extract_mc_maze, extract_mc_rtt, extract_area_2b
import simulations.config as config
from simulations.real_data.parameters import F_CV

from filters.KalmanFilter_test import KFTest

from hnets.hnet import HyperNetwork
from hnets.hnet_deconv import hnet_deconv
from mnets.KNet_mnet_allCM import KalmanNetNN as KNet_mnet

from pipelines.Pipeline_cm import Pipeline_cm
from pipelines.Pipeline_EKF import Pipeline_EKF


print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

#########################
### Parameter Setting ###
#########################
args = config.general_settings()
args.use_cuda = True # use GPU or not
if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
      torch.set_default_tensor_type(torch.cuda.FloatTensor)
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

### dataset parameters ##################################################
# determine noise distribution normal/exp (DEFAULT: "normal (=Gaussian)")
args.proc_noise_distri = "normal"
args.meas_noise_distri = "normal"

args.mixed_dataset = False #to use batch size list

# args.N_E = 1000 #training dataset size
# args.N_CV = 100 #cross validation
args.N_T = 48 #testing dataset size
args.per_cv = 0.2 #cross validation percentage
args.per_test = 0.2 #testing percentage
args.per_train = 0.6 #training percentage
args.k_pca = 30
# sequence length
args.T = 100
args.T_test = 100
train_lengthMask = None
cv_lengthMask = None
test_lengthMask = None

### training parameters ##################################################
args.wandb_switch = False
if args.wandb_switch:
   import wandb
   wandb.init(project="HKNet_Linear")
args.knet_trainable = True

# training parameters for KNet
args.in_mult_KNet = 1
args.out_mult_KNet = 1

args.n_steps = 50000
args.n_batch = 32 
args.lr = 1e-6
args.wd = 1e-3
args.k_pca = 10
args.gauss_width = 50


### paths ##################################################
path_results = 'simulations/real_data/results/'

# spikes_pca1, target_1 = extract_mc_maze(args)
spikes_pca2, target_2 = extract_mc_rtt(args)
# spikes_pca3, target_3 = extract_area_2bump(args)

# target1 = torch.from_numpy(target_1).float().to(device)
# spikespca1 = torch.from_numpy(spikes_pca1).float().to(device)

target2 = torch.from_numpy(target_2).float().to(device)
spikespca2 = torch.from_numpy(spikes_pca2).float().to(device)

# target3 = torch.from_numpy(target_3).float().to(device)
# spikespca3 = torch.from_numpy(spikes_pca3).float().to(device)

# torch.save(target1, 'target1.pt')
# torch.save(spikespca1, 'spikespca1.pt')

torch.save(target2, 'target2.pt')
torch.save(spikespca2, 'spikespca2.pt')

# torch.save(target3, 'target3.pt')
# torch.save(spikespca3, 'spikespca3.pt')

target1 = torch.load('target1.pt').to(device)
spikespca1 = torch.load('spikespca1.pt').to(device)

# target2 = torch.load('target2.pt').to(device)
# spikespca2 = torch.load('spikespca2.pt').to(device)

# target3 = torch.load('target3.pt').to(device)
# spikespca3 = torch.load('spikespca3.pt').to(device)

# input_list = [spikespca1, spikespca2, spikespca3]
# target_list = [target1, target2, target3]

input_list = [spikespca1]
target_list = [target1]

train_input_list = []
train_target_list = []
cv_input_list = []
cv_target_list = []
test_input_list = []
test_target_list = []
train_init_list = []
cv_init_list = []
test_init_list = []

## artificial SoW (not used for kalmannet)
SoW = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

for i in range(len(input_list)):
   train_input, train_target, train_init, cv_input, cv_target, cv_init, test_input, test_target, test_init = SplitData(args, input_list[i], target_list[i])
   train_input_list.append([train_input, SoW[i]])
   train_target_list.append([train_target, SoW[i]])
   cv_input_list.append([cv_input, SoW[i]])
   cv_target_list.append([cv_target, SoW[i]])
   test_input_list.append([test_input, SoW[i]])
   test_target_list.append([test_target, SoW[i]])
   train_init_list.append(train_init)
   cv_init_list.append(cv_init)
   test_init_list.append(test_init) # = x0

#############
### Model ###
#############
F = F_CV.to(device)
n = spikespca1.shape[1]
m = target1.shape[1]
m1_0 = torch.zeros(m, 1)
m1_0 = m1_0.to(device)
# deterministic initial condition
m2_0 = 0 * torch.eye(m)
m2_0 = m2_0.to(device) 
## estimate H using linear regression:
## change to only use train set (use train_input and spikes_input)
# target_all = torch.cat([target1, target2, target3], dim=0)
# spikes_all = torch.cat([spikespca1, spikespca2, spikespca3], dim=0)
target_all = torch.cat([target1])
spikes_all = torch.cat([spikespca1])
Y = target_all  # (N, 4)
X = spikes_all  # (N, 30)
H_est = (torch.linalg.pinv(X) @ Y)
#H_est = torch.linalg.lstsq(X, Y).solution.T  # Shape (4, 30)
H_est = H_est.to(device)

## Q and R structure
Q_structure = torch.eye(m)
R_structure = torch.eye(n)

## artficial q2 and r2 for sysmodel with real data
q2 = 1
r2 = 1

SoW_train_range = list(range(len(SoW)))

## model
sys_model = []
for i in range(len(SoW)):
   sys_model_i = SystemModel(F, q2*Q_structure, H_est, r2*R_structure, args.T, args.T_test, q2, r2)
   sys_model_i.InitSequence(m1_0, m2_0)
   sys_model.append(sys_model_i)


##############################
### Evaluate Kalman Filter ###
##############################


print("Evaluate Kalman Filter with GT noise cov")
i=0
test_input = test_input_list[i][0]
test_target = test_target_list[i][0]
test_init = test_init_list[i] 
print(f"Dataset {i}") 
[MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(args, sys_model[i], test_input, test_target)

##########################
### KalmanNet Pipeline ###
##########################
## train and test KalmanNet
i = 0
print(f"KalmanNet pipeline start, train on dataset {i}")
KalmanNet_model = KNet_mnet()
KalmanNet_model.NNBuild(sys_model[i], args)
print("Number of trainable parameters for KalmanNet:",sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))
## Train Neural Network
KalmanNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
KalmanNet_Pipeline.setssModel(sys_model[i])
KalmanNet_Pipeline.setModel(KalmanNet_model)
KalmanNet_Pipeline.setTrainingParams(args)
KalmanNet_Pipeline.NNTrain(sys_model[i], cv_input_list[i][0], cv_target_list[i][0], train_input_list[i][0], train_target_list[i][0], path_results)
for i in range(len(SoW)):
   print(f"Dataset {i}") 
   KalmanNet_Pipeline.NNTest(sys_model[0], test_input_list[i][0], test_target_list[i][0], path_results)

## Close wandb run
if args.wandb_switch: 
   wandb.finish() 