import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from datetime import datetime
import os

from simulations.Linear_sysmdl import SystemModel
from simulations.utils import SplitData, extract_mc_maze, extract_mc_rtt, extract_area_2b, estimate_H
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
args.use_cuda = True
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

args.proc_noise_distri = "normal"
args.meas_noise_distri = "normal"
args.mixed_dataset = False
args.per_cv = 0.2
args.per_test = 0.2
args.per_train = 0.6
args.k_pca = 50
args.gauss_width = 50
args.T = 100
args.T_test = 100
args.wandb_switch = False
args.knet_trainable = True
args.in_mult_KNet = 1
args.out_mult_KNet = 1
args.n_steps = 50000
args.n_batch = 32
args.lr = 1e-3
args.wd = 1e-3
# max trial numbers for data preprocessing
args.max_trials = 200 

### paths ##################################################
path_results = 'simulations/real_data/results/'
path_data = f'data/real_data/{args.max_trials}_trials/'
os.makedirs(path_data, exist_ok=True)

############
### Data ###
############

## --- Extract and Save ---
spikes_pca1, target_1 = extract_mc_maze(args, max_trials=args.max_trials)
spikes_pca2, target_2 = extract_mc_rtt(args, max_trials=args.max_trials)
spikes_pca3, target_3 = extract_area_2b(args, max_trials=args.max_trials)

target1 = torch.from_numpy(target_1).float().to(device)
spikespca1 = torch.from_numpy(spikes_pca1).float().to(device)

target2 = torch.from_numpy(target_2).float().to(device)
spikespca2 = torch.from_numpy(spikes_pca2).float().to(device)

target3 = torch.from_numpy(target_3).float().to(device)
spikespca3 = torch.from_numpy(spikes_pca3).float().to(device)

torch.save(target1, os.path.join(path_data, 'target1.pt'))
torch.save(spikespca1, os.path.join(path_data, 'spikespca1.pt'))

torch.save(target2, os.path.join(path_data, 'target2.pt'))
torch.save(spikespca2, os.path.join(path_data, 'spikespca2.pt'))

torch.save(target3, os.path.join(path_data, 'target3.pt'))
torch.save(spikespca3, os.path.join(path_data, 'spikespca3.pt'))

## Optional: loading from saved files
# target1 = torch.load(os.path.join(path_data, 'target1.pt')).to(device)
# spikespca1 = torch.load(os.path.join(path_data, 'spikespca1.pt')).to(device)
# target2 = torch.load(os.path.join(path_data, 'target2.pt')).to(device)
# spikespca2 = torch.load(os.path.join(path_data, 'spikespca2.pt')).to(device)
# target3 = torch.load(os.path.join(path_data, 'target3.pt')).to(device)
# spikespca3 = torch.load(os.path.join(path_data, 'spikespca3.pt')).to(device)

input_list = [spikespca1, spikespca2, spikespca3]
target_list = [target1, target2, target3]

train_input_list = []
train_target_list = []
cv_input_list = []
cv_target_list = []
test_input_list = []
test_target_list = []
train_init_list = []
cv_init_list = []
test_init_list = []

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
   test_init_list.append(test_init)

#############
### Model ###
#############
F = F_CV.to(device)
q2 = 1
r2 = 1
sys_model = []

for i in range(len(input_list)):
    Y = target_list[i]
    X = input_list[i]

    # Estimate H robustly
    H_est = estimate_H(X, Y)
    
    m = Y.shape[1]
    n = X.shape[1]

    m1_0 = torch.zeros(m, 1).to(device)
    m2_0 = torch.eye(m).to(device) * 0  # if needed, change this

    Q_structure = torch.eye(m)
    R_structure = torch.eye(n)

    sys_model_i = SystemModel(F, q2 * Q_structure, H_est.to(device), r2 * R_structure, args.T, args.T_test, q2, r2)
    sys_model_i.InitSequence(m1_0, m2_0)
    sys_model.append(sys_model_i)

##########
### KF ###
##########
print("Evaluate Kalman Filter")
for i in range(len(SoW)):
  test_input = test_input_list[i][0]
  test_target = test_target_list[i][0]
  test_init = test_init_list[i]
  args.N_T = test_input.shape[0] #automatically matches dataset size
  print(f"Dataset {i}")
  [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(args, sys_model[i], test_input, test_target)

#################
### Kalmannet ###
#################
## Decide which dataset you want to train on
j = 0 #change value to 0,1,2 for different datasets

print(f"KalmanNet pipeline start, train on dataset {j}")
KalmanNet_model = KNet_mnet()
KalmanNet_model.NNBuild(sys_model[j], args)
print("Number of trainable parameters for KalmanNet:", sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))
KalmanNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
KalmanNet_Pipeline.setssModel(sys_model[j])
KalmanNet_Pipeline.setModel(KalmanNet_model)
KalmanNet_Pipeline.setTrainingParams(args)
KalmanNet_Pipeline.NNTrain(sys_model[j], cv_input_list[j][0], cv_target_list[j][0], train_input_list[j][0], train_target_list[j][0], path_results)

for i in range(len(SoW)):
   print(f"Dataset {i}")
   KalmanNet_Pipeline.NNTest(sys_model[0], test_input_list[i][0], test_target_list[i][0], path_results)

if args.wandb_switch:
   wandb.finish()
