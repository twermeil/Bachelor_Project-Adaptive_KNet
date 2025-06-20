import torch
import torch.nn as nn
from datetime import datetime

from simulations.Linear_sysmdl import SystemModel
from simulations.utils import DataGen
import simulations.config as config
from simulations.linear_canonical.parameters import F, H, Q_structure, R_structure, Q_structure_nonid, R_structure_nonid,\
   m, m1_0

from filters.KalmanFilter_test import KFTest

from hnets.hnet import HyperNetwork
from hnets.hnet_deconv import hnet_deconv

if m==2: # 2x2 system
   from mnets.KNet_mnet import KalmanNetNN as KNet_mnet
else: # 5x5, 10x10 system
   from mnets.KNet_mnet_MAML import KalmanNetNN as KNet_mnet

from pipelines.Pipeline_cm import Pipeline_cm
from pipelines.Pipeline_EKF import Pipeline_EKF

import numpy as np
import matplotlib.pyplot as plt

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
args.use_cuda = False # use GPU or not
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
# determine noise distribution normal/exp (DEFAULT: "normal" (=Gaussian))
args.proc_noise_distri = "exponential"
args.meas_noise_distri = "exponential"

F = F.to(device)
H = H.to(device)
if args.proc_noise_distri == "normal":
   Q_structure = Q_structure_nonid.to(device)
elif args.proc_noise_distri == "exponential":
   Q_structure = Q_structure.to(device)
else:
   raise ValueError("args.proc_noise_distri not recognized")
if args.meas_noise_distri == "normal":
   R_structure = R_structure_nonid.to(device)
elif args.meas_noise_distri == "exponential":
   R_structure = R_structure.to(device)
else:
   raise ValueError("args.meas_noise_distri not recognized")
m1_0 = m1_0.to(device)

args.N_E = 1000 #training dataset size
args.N_CV = 100 #cross validation
args.N_T = 200 #testing dataset size
# deterministic initial condition
m2_0 = 0 * torch.eye(m)
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
args.in_mult_KNet = 40
args.out_mult_KNet = 40
args.n_steps = 50000
#already tuned parameters
args.n_batch = 32 #how many samples for GD --> the higher the more stable gradient, but may overfit the training batch
args.lr = 1e-3 #learning rate (larger --> loss decrease faster but more unstable)
args.wd = 1e-3 #L2 ridge regression (higher --> more regularization --> more risk of underfitting)

# training parameters for Hypernet
args.hnet_arch = "GRU" # "deconv" or "GRU
if args.hnet_arch == "GRU": # settings for GRU hnet
   args.hnet_hidden_size_discount = 100 #by how much you divide the layer's hidden size (see hnet.py)
elif args.hnet_arch == "deconv": # settings for deconv hnet
   # 2x2 system
   embedding_dim = 4
   hidden_channel_dim = 32
else:
   raise Exception("args.hnet_arch not recognized")
n_steps = 10000
n_batch_list = [32,32,32,32] # batch size for each dataset
lr = 1e-3
wd = 1e-3

### True model ##################################################
# SoW (state of world)
SoW = torch.tensor([[10,-10], [0,-10], [-10,-10],[-20,-10]])
SoW_train_range = list(range(len(SoW))) # first *** number of datasets are used for training
print("SoW_train_range: ", SoW_train_range)
SoW_test_range = list(range(len(SoW))) # last *** number of datasets are used for testing
# noise
r2_dB = SoW[:, 0]
q2_dB = SoW[:, 1]

r2 = 10 ** (r2_dB/10)
q2 = 10 ** (q2_dB/10)

# change SoW to q2/r2 ratio
SoW = q2/r2

for i in range(len(SoW)):
   print(f"SoW of dataset {i}: ", SoW[i])
   print(f"r2 [linear] and q2 [linear] of dataset  {i}: ", r2[i], q2[i])

# model
sys_model = []
for i in range(len(SoW)):
   sys_model_i = SystemModel(F, q2[i]*Q_structure, H, r2[i]*R_structure, args.T, args.T_test, q2[i], r2[i])
   sys_model_i.InitSequence(m1_0, m2_0)
   sys_model.append(sys_model_i)

### paths ##################################################
path_results = 'simulations/linear_canonical/results/2x2/exp/'
dataFolderName = 'data/linear_canonical/2x2/exp_train' + '/'
dataFileName = []
for i in range(len(SoW)):
   dataFileName.append('r2=' + str(r2_dB[i].item())+"dB"+"_" +"q2="+ str(q2_dB[i].item())+"dB" + '.pt')
###################################
### Data Loader (Generate Data) ###
###################################
print("Start Data Gen")
for i in range(len(SoW)):
   DataGen(args, sys_model[i], dataFolderName + dataFileName[i])
   
print("Data Load")
train_input_list = []
train_target_list = []
cv_input_list = []
cv_target_list = []
test_input_list = []
test_target_list = []
train_init_list = []
cv_init_list = []
test_init_list = []

for i in range(len(SoW)):  
   [train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init] = torch.load(dataFolderName + dataFileName[i], map_location=device)
   train_input_list.append((train_input, SoW[i])) #input = y
   train_target_list.append((train_target, SoW[i])) #target = x
   cv_input_list.append((cv_input, SoW[i]))
   cv_target_list.append((cv_target, SoW[i]))
   test_input_list.append((test_input, SoW[i]))
   test_target_list.append((test_target, SoW[i]))
   train_init_list.append(train_init)
   cv_init_list.append(cv_init)
   test_init_list.append(test_init) #=x0

##############################
### Evaluate Kalman Filter ###
##############################

print("Evaluate Kalman Filter True")
for i in range(len(SoW)):
   test_input = test_input_list[i][0]
   test_target = test_target_list[i][0]
   test_init = test_init_list[i]  
   test_lengthMask = None 
   print(f"Dataset {i}") 
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(args, sys_model[i], test_input, test_target, test_lengthMask=test_lengthMask)


##################################
### Hyper - KalmanNet Pipeline ###
##################################
### train and test KalmanNet on dataset i
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
#KalmanNet_Pipeline.NNTrain(sys_model[i], cv_input_list[i][0], cv_target_list[i][0], train_input_list[i][0], train_target_list[i][0], path_results)
for i in range(len(SoW)):
   print(f"Dataset {i}") 
   KalmanNet_Pipeline.NNTest(sys_model[i], test_input_list[i][0], test_target_list[i][0], path_results)

### frozen KNet weights, train hypernet to generate CM weights on multiple datasets
# load frozen weights from file where pipeline saved it
frozen_weights = torch.load(path_results + 'knet_best-model.pt', map_location=device) 
### frozen KNet weights, train hypernet to generate CM weights on multiple datasets
args.knet_trainable = False # frozen KNet weights
args.use_context_mod = True # use CM
args.mixed_dataset = True # use mixed dataset training
## training parameters for Hypernet
args.n_steps = n_steps
args.n_batch_list = n_batch_list # will be multiplied by num of datasets
args.lr = lr
args.wd = wd
## Build Neural Networks
print("Build HNet and KNet")
KalmanNet_model = KNet_mnet()
cm_weight_size = KalmanNet_model.NNBuild(sys_model[0], args, frozen_weights=frozen_weights)
print("Number of CM parameters:", cm_weight_size)

# Split into gain and shift
cm_weight_size = torch.tensor([cm_weight_size / 2]).int().item()

if args.hnet_arch == "deconv":
   HyperNet_model = hnet_deconv(args, 1, cm_weight_size, embedding_dim=embedding_dim, hidden_channel_dim = hidden_channel_dim)
   weight_size_hnet = HyperNet_model.print_num_weights()
elif args.hnet_arch == "GRU":
   HyperNet_model = HyperNetwork(args, 1, cm_weight_size)
   weight_size_hnet = sum(p.numel() for p in HyperNet_model.parameters() if p.requires_grad)
   print("Number of parameters for HyperNet:", weight_size_hnet)
else:
   raise ValueError("Unknown hnet_arch")

## Set up pipeline
hknet_pipeline = Pipeline_cm(strTime, "pipelines", "hknet")
hknet_pipeline.setModel(HyperNet_model, KalmanNet_model)
hknet_pipeline.setTrainingParams(args)
## Optinal: record parameters to wandb
if args.wandb_switch:
   wandb.log({
   "total_params": cm_weight_size + weight_size_hnet,
   "batch_size": args.n_batch,
   "learning_rate": args.lr,  
   "weight_decay": args.wd})
## Train Neural Networks
#hknet_pipeline.NNTrain_mixdatasets(SoW_train_range, sys_model, cv_input_list, cv_target_list, train_input_list, train_target_list, path_results,cv_init_list,train_init_list)

## Test Neural Networks for each dataset  
print("Training points: ")
hknet_pipeline.NNTest_alldatasets(SoW_test_range, sys_model, test_input_list, test_target_list, path_results,test_init_list)

print("Figure Points")

################
#### Figure ####
################

SoW2 = torch.tensor([[0, -20], [-10, -30], [-20, -40], #SoW = -20
                     [10, 0], [-10, -20], [-20, -30], #SoW = -10
                     [10, 10], [0, 0], [-20, -20], #SoW = 0
                     [10, 20], [0, 10], [-10, 0]]) #SoW = 10

SoW_train_range2 = list(range(len(SoW2))) # first *** number of datasets are used for training
print("SoW_train_range_2: ", SoW_train_range2)
SoW_test_range2 = list(range(len(SoW2))) # last *** number of datasets are used for testing
# noise
r2_dB2 = SoW2[:, 0]
q2_dB2 = SoW2[:, 1]

r2_2 = 10 ** (r2_dB2/10) # = 10 / 1 / 10^-1 / 10^-2
# => 1/r2 = 10^-1 / 1 / 10 / 10^2 
q2_2 = 10 ** (q2_dB2/10) # = 10^-1 / 10^-1 / 10^-1 / 10^-1

# change SoW to q2/r2 ratio
SoW2 = q2_2/r2_2 # = 10^-2 / 10^-1 / 1 / 10

# 1/r2 [dB] = [-10, 0, 10, 20]
# SoW [dB] = [-20, -10, 0, 10]

for i in range(len(SoW2)):
   print(f"SoW of dataset {i}: ", SoW2[i])
   print(f"r2 [linear] and q2 [linear] of dataset  {i}: ", r2_2[i], q2_2[i])

# model
sys_model2 = []
for i in range(len(SoW2)):
   sys_model_i = SystemModel(F, q2_2[i]*Q_structure, H, r2_2[i]*R_structure, args.T, args.T_test, q2_2[i], r2_2[i])
   sys_model_i.InitSequence(m1_0, m2_0)
   sys_model2.append(sys_model_i)
   
### paths ##################################################
dataFolderName2 = 'data/linear_canonical/2x2/exp_test' + '/'
dataFileName2 = []
for i in range(len(SoW2)):
   dataFileName2.append('r2=' + str(r2_dB2[i].item())+"dB"+"_" +"q2="+ str(q2_dB2[i].item())+"dB" + '.pt')
   
### Data Generation
print("Start Data Gen")
for i in range(len(SoW2)):
   DataGen(args, sys_model2[i], dataFolderName2 + dataFileName2[i])
   
print("Data Load")
train_input_list2 = []
train_target_list2 = []
cv_input_list2 = []
cv_target_list2 = []
test_input_list2 = []
test_target_list2 = []
train_init_list2 = []
cv_init_list2 = []
test_init_list2 = []

for i in range(len(SoW2)):  
   [train_input2, train_target2, cv_input2, cv_target2, test_input2, test_target2 ,train_init2, cv_init2, test_init2] = torch.load(dataFolderName2 + dataFileName2[i], map_location=device)
   train_input_list2.append((train_input2, SoW2[i])) #input = y
   train_target_list2.append((train_target2, SoW2[i])) #target = x
   cv_input_list2.append((cv_input2, SoW2[i]))
   cv_target_list2.append((cv_target2, SoW2[i]))
   test_input_list2.append((test_input2, SoW2[i]))
   test_target_list2.append((test_target2, SoW2[i]))
   train_init_list2.append(train_init2)
   cv_init_list2.append(cv_init2)
   test_init_list2.append(test_init2) #=x0
   
print("Test points: ")
   
hknet_pipeline.NNTest_alldatasets(SoW_test_range2, sys_model2, test_input_list2, test_target_list2, path_results, test_init_list2)

#test_points = hknet_pipeline.MSE_test_dB_avg

#print('Test points : ', test_points)

## Kalmanfilter points ##

#kf_pts = MSE_KF_dB_avg
#print('KF: ', kf_pts)

## Figure ##

#fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# ---- Left Plot ----
#ax = axes[0]
#ax.grid(True)

#x = np.linspace(-10, 20, 100)
#x1 = [-10, 0, 10, 20] # = 1/r2 [dB]
#x2 = [0, 10, 20, -10, 10, 20, -10, 0, 20, -10, 0, 10]

# Dashed lines for Kalman Filter (KF)
#ax.plot(x, kf_pts[0], 'r--', linewidth=2, label='SoW$_t$=-20dB')
#ax.plot(x, kf_pts[1], 'g--', linewidth=2, label='SoW$_t$=-10dB')
#ax.plot(x, kf_pts[2], 'b--', linewidth=2, label='SoW$_t$=1dB')
#ax.plot(x, kf_pts[3], 'm--', linewidth=2, label='SoW$_t$=10dB')

# Markers for AKNet training and inference

#train pts
#ax.plot(x1[0], train_pts[0], 'ro', markersize=8, label='AKNet: training')#SoW = -20
#ax.plot(x1[1], train_pts[1], 'go', markersize=8, label='AKNet: training')#SoW = -10
#ax.plot(x1[2], train_pts[2], 'bo', markersize=8, label='AKNet: training')#SoW = 0
#ax.plot(x1[3], train_pts[3], 'mo', markersize=8, label='AKNet: training')#SoW = 10

#test pts
#ax.plot(x2[:3], test_points[:3], 'r+', markersize=8, label='AKNet: inference')#SoW = -20
#ax.plot(x2[3:6], test_points[3:6], 'g+', markersize=8, label='AKNet: inference')#SoW = -10
#ax.plot(x2[6:9], test_points[6:9], 'b+', markersize=8, label='AKNet: inference')#SoW = 0
#ax.plot(x2[9:], test_points[9:], 'm+', markersize=8, label='AKNet: inference')#SoW = 10

# Labels and Legend
#ax.set_xlabel('$1/r_t^2$ [dB]')
#ax.set_ylabel('MSE LOSS [dB]')
#ax.legend(loc='lower left')

## Close wandb run
if args.wandb_switch: 
   wandb.finish() 