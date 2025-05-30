import torch
import torch.nn as nn
from datetime import datetime

from simulations.Linear_sysmdl import SystemModel
from simulations.utils import DataGen
import simulations.config as config
from simulations.linear_canonical.parameters import F, H, Q_structure, R_structure,\
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

from noise_estimator.search import Pipeline_NE
from noise_estimator.KF_search import KF_NE

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
F = F.to(device)
H = H.to(device)
Q_structure = Q_structure.to(device)
R_structure = R_structure.to(device)
m1_0 = m1_0.to(device)

args.N_E = 1000
args.N_CV = 100
args.N_T = 200
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

# training parameters for KNet
args.knet_trainable = True
# depending on m and n, scale the input and output dimension multiplier on the FC layers and LSTM layers of KNet 
# 2x2 system
args.in_mult_KNet = 40
# 5x5, 10x10 system
# args.in_mult_KNet = 1
# args.out_mult_KNet = 1
args.n_steps = 50000
args.n_batch = 32 
args.lr = 1e-3
args.wd = 1e-3

# training parameters for Hypernet
args.hnet_arch = "GRU" # "deconv" or "GRU
if args.hnet_arch == "GRU": # settings for GRU hnet
   args.hnet_hidden_size_discount = 100

elif args.hnet_arch == "deconv": # settings for deconv hnet
   # 2x2 system
   embedding_dim = 4
   hidden_channel_dim = 32

else:
   raise Exception("args.hnet_arch not recognized")

args.UnsupervisedLoss = True # use unsupervised loss
n_steps = 5000
n_batch_list = [32]  # will be multiplied by num of datasets
lr = 1e-3
wd = 1e-3

# parameters for SoW search
shift = 10 # shift factor for R (shift between train and inference, R_inference = shift * R_train)
args.grid_size_dB = 0.1 # step size for grid search of SoW in dB
args.forget_factor = 0.3 # forget factor for innovation based estimation
args.max_iter = 100 # max number of iterations for SoW search
args.SoW_conv_error = 1e-4 # convergence error for SoW search

### True model ##################################################
# SoW
# SoW = torch.tensor([[10,10], [10,1], [10,0.1], [10,0.01],
#                     [1,10], [1,1], [1,0.1], [1,0.01],
#                     [0.1,10], [0.1,1], [0.1,0.1], [0.1,0.01],
#                     [0.01,10], [0.01,1], [0.01,0.1], [0.01,0.01]])
SoW = torch.tensor([[10,0.1],[1,0.1],[0.1,0.1],[0.01,0.1]]) # different q2/r2 ratios
# SoW = torch.tensor([[5,0.1],[0.5,0.1],[0.05,0.1]]) # interpolation
SoW_train_range = list(range(len(SoW))) # these datasets are used for training
n_batch_list = n_batch_list * len(SoW_train_range)
print("SoW_train_range: ", SoW_train_range)
SoW_test_range = list(range(len(SoW))) # these datasets are used for testing
# noise
r2 = SoW[:, 0]
q2 = SoW[:, 1]

# change SoW to q2/r2 ratio
SoW = q2/r2
SoW_dB = 10 * torch.log10(SoW)
# Calculate the range (min, max)
min_dB = torch.min(SoW_dB)
max_dB = torch.max(SoW_dB)
SoW_range_dB = (-20, 10)
print("SoW_range: ", SoW_range_dB, "[dB]")
for i in range(len(SoW)):
   print(f"SoW of dataset {i}: ", SoW[i])
   print(f"r2 [linear] and q2 [linear] of dataset  {i}: ", r2[i], q2[i])

# model
sys_model = []
for i in range(len(SoW)):
   sys_model_i = SystemModel(F, q2[i]*Q_structure, H, r2[i]*R_structure, args.T, args.T_test, q2[i], r2[i])
   sys_model_i.InitSequence(m1_0, m2_0)
   sys_model.append(sys_model_i)

# sys_model_init = SystemModel(F, Q_structure, H, R_structure, args.T, args.T_test, 1, 1)
# sys_model_init.InitSequence(m1_0, m2_0)

sys_model_init = []
SoW_init = torch.zeros(len(SoW)) # initial SoW [linear scale]
for i in range(len(SoW)):
   sys_model_init_i = SystemModel(F, q2[i]*Q_structure, H, shift*r2[i]*R_structure, args.T, args.T_test, q2[i], shift*r2[i])
   sys_model_init_i.InitSequence(m1_0, m2_0)
   sys_model_init.append(sys_model_init_i)
   SoW_init[i] = q2[i]/r2[i]/shift # initial SoW [linear scale]


### paths ##################################################
path_results = 'simulations/linear_canonical/results/2x2/'
dataFolderName = 'data/linear_canonical/2x2/30dB' + '/'
dataFileName = []
rounding_digits = 4 # round to # digits after decimal point
for i in range(len(SoW)):
   r2_rounded = round(r2[i].item() * 10**rounding_digits) / 10**rounding_digits
   q2_rounded = round(q2[i].item() * 10**rounding_digits) / 10**rounding_digits
   dataFileName.append('r2=' + str(r2_rounded)+"_" +"q2="+ str(q2_rounded)+ '.pt')
###################################
### Data Loader (Generate Data) ###
###################################
# print("Start Data Gen")
# for i in range(len(SoW)):
#    DataGen(args, sys_model[i], dataFolderName + dataFileName[i])
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
   train_input_list.append([train_input, SoW[i]])
   train_target_list.append([train_target, SoW[i]])
   cv_input_list.append([cv_input, SoW[i]])
   cv_target_list.append([cv_target, SoW[i]])
   test_input_list.append([test_input, SoW[i]])
   test_target_list.append([test_target, SoW[i]])
   train_init_list.append(train_init)
   cv_init_list.append(cv_init)
   test_init_list.append(test_init)

##############################
### Evaluate Kalman Filter ###
##############################
print("Evaluate Kalman Filter with GT noise cov")
for i in range(len(SoW)):
   test_input = test_input_list[i][0]
   test_target = test_target_list[i][0]
   test_init = test_init_list[i] 
   print(f"Dataset {i}") 
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(args, sys_model[i], test_input, test_target)

#  Use innovation based method to estimate Q and R
print("Estimate noise cov with innovation based method")
KF_noise_est = KF_NE(strTime, "filters", "KF")
KF_noise_est.setParams(args)
for i in range(len(SoW)):
   test_input = test_input_list[i][0]
   test_target = test_target_list[i][0]
   test_init = test_init_list[i] 
   print(f"Dataset {i}") 
   print("GT Q: ", sys_model[i].Q)
   print("GT R: ", sys_model[i].R)
   ###  Use innovation based method to estimate Q and R (init with sys_model_init)
   R_i, Q_i = KF_noise_est.linear_innovation_based_estimation(sys_model_init[i], test_input, test_init)
   ### Use innovation based method to estimate Q and R (init with GT)
   # R_i, Q_i = KF_noise_est.linear_innovation_based_estimation(sys_model[i], test_input, test_init) 
   print("Estimated Q: ", Q_i)
   print("Estimated R: ", R_i)
   q2_i = KF_noise_est.estimate_scalar(Q_i, Q_structure)
   print("Estimated q2:", q2_i)
   r2_i = KF_noise_est.estimate_scalar(R_i, R_structure)
   print("Estimated r2:", r2_i)
   ### both q and r are time-varying
   # sys_model_feed = SystemModel(F, Q_i, H, R_i, args.T, args.T_test, q2_i, r2_i)
   ### fixed q
   sys_model_feed = SystemModel(F, sys_model[i].Q, H, R_i, args.T, args.T_test, q2[i], r2_i)
   sys_model_feed.InitSequence(m1_0, m2_0)
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(args, sys_model_feed, test_input, test_target)
   


##########################
### KalmanNet Pipeline ###
##########################
## train and test KalmanNet
# i = 0
# print(f"KalmanNet pipeline start, train on dataset {i}")
# KalmanNet_model = KNet_mnet()
# KalmanNet_model.NNBuild(sys_model[i], args)
# print("Number of trainable parameters for KalmanNet:",sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))
# ## Train Neural Network
# KalmanNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
# KalmanNet_Pipeline.setssModel(sys_model[i])
# KalmanNet_Pipeline.setModel(KalmanNet_model)
# KalmanNet_Pipeline.setTrainingParams(args)
# KalmanNet_Pipeline.NNTrain(sys_model[i], cv_input_list[i][0], cv_target_list[i][0], train_input_list[i][0], train_target_list[i][0], path_results)
# ## Test Neural Network on all datasets
# for i in range(len(SoW)):
#    print(f"Dataset {i}") 
#    KalmanNet_Pipeline.NNTest(sys_model[i], test_input_list[i][0], test_target_list[i][0], path_results)

######################################
### Hypernet(generate CM) Pipeline ###
######################################
### load frozen weights
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
   "batch_size(hnet)": args.n_batch,
   "learning_rate(hnet)": args.lr,  
   "weight_decay(hnet)": args.wd})
## Train Neural Networks
# hknet_pipeline.NNTrain_mixdatasets(SoW_train_range, sys_model, cv_input_list, cv_target_list, train_input_list, train_target_list, path_results,cv_init_list,train_init_list)

## Test Neural Networks for each dataset  
# hknet_pipeline.NNTest_alldatasets(SoW_test_range, sys_model, test_input_list, test_target_list, path_results,test_init_list)

###########################
### SoW search Pipeline ###
###########################
# load frozen weights
frozen_weights = torch.load(path_results + 'knet_best-model.pt', map_location=device) 
### frozen KNet weights, train hypernet to generate CM weights on multiple datasets
args.knet_trainable = False # frozen KNet weights
args.use_context_mod = True # use CM
KalmanNet_model = KNet_mnet()
cm_weight_size = KalmanNet_model.NNBuild(sys_model[0], args, frozen_weights=frozen_weights)
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
SoW_pipeline = Pipeline_NE(strTime, "pipelines", "hknet")
SoW_pipeline.setModel(HyperNet_model, KalmanNet_model)
SoW_pipeline.setTrainingParams(args)
## Optinal: record parameters to wandb
if args.wandb_switch:
   wandb.log({
   "grid size SoW [dB]": args.grid_size_dB})

SoW_opt = torch.zeros(len(SoW))
for i in range(len(SoW)):
   test_input = test_input_list[i][0]
   test_target = test_target_list[i][0]
   test_init = test_init_list[i] 
   print(f"Dataset {i}") 
   # # Grid search 
   # SoW_opt[i] = SoW_pipeline.grid_search(SoW_range_dB, sys_model[i], test_input, path_results, test_init, test_target=test_target, SoW_true=SoW[i])
   
   # Inn-based estimation 
   print("GT Q: ", sys_model[i].Q)
   print("GT R: ", sys_model[i].R)
   print("GT SoW: ", SoW[i])
   ### Use innovation based method to estimate Q and R (init with sys_model_init)
   R_est, Q_est = SoW_pipeline.innovation_based_estimation(SoW_init[i], sys_model_init[i].Q, sys_model_init[i].R, sys_model[i], test_input, path_results, test_init)
   ### Use innovation based method to estimate Q and R (init with GT)
   # R_est, Q_est = SoW_pipeline.innovation_based_estimation(SoW[i], sys_model[i].Q, sys_model[i].R, sys_model[i], test_input, path_results, test_init)
   ### q and r are time-varying
   # SoW_opt[i] = SoW_pipeline.update_SoW(SoW_range_dB, Q_est, R_est, Q_structure, R_structure)
   ### fixed q
   SoW_opt[i] = SoW_pipeline.update_SoW(SoW_range_dB, sys_model[i].Q, R_est, Q_structure, R_structure)
   
   print("Estimated Q: ", Q_est)
   print("Estimated R: ", R_est)
   print("Searched SoW: ", SoW_opt[i])

   # Update SoW
   test_input_list[i][1] = SoW_opt[i]
   test_target_list[i][1] = SoW_opt[i]

## Test Neural Networks for each dataset with searched SoW  
print("Test model with searched SoW")
hknet_pipeline.NNTest_alldatasets(SoW_test_range, sys_model, test_input_list, test_target_list, path_results,test_init_list)

## Close wandb run
if args.wandb_switch: 
   wandb.finish() 