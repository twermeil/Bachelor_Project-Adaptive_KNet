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
from mnets.KNet_mnet_MAML import KalmanNetNN as KNet_mnet

from pipelines.Pipeline_cm import Pipeline_cm
from pipelines.Pipeline_EKF_MAML import Pipeline_EKF_MAML

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
# determine noise distribution normal/exp (DEFAULT: "normal (=Gaussian)")
args.proc_noise_distri = "normal"
args.meas_noise_distri = "normal"

args.mixed_dataset = True #to use batch size list

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
args.n_batch = 32
args.update_lr = 1e-3 #0.4
args.meta_lr = 1e-3 #0.001
args.spt_percentage = 0.3
args.update_step = 5
args.maml_wd = [0.3, 0.3, 0.2, 0.1, 0.1, 0.01]

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
n_steps = 10000
args.n_batch_list = [32,32,32,32] # batch size for each support dataset
args.n_batch_list_query = [32,32,32,32] # batch size for each query dataset
#lr and wd for hypernet (not used for knet)
args.lr = 1e-3
args.knet_wd = 1e-3
args.hypernet_wd = 1e-3

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
path_results = 'simulations/maml/results/2x2/normal/'
dataFolderName = 'data/maml/2x2/normal' + '/'
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
   test_init_list.append(test_init) # = x0

##############################
### Evaluate Kalman Filter ###
##############################
#print("Evaluate Kalman Filter True")
#for i in range(len(SoW)):
   #test_input = test_input_list[i][0]
   #test_target = test_target_list[i][0]
   #test_init = test_init_list[i]  
   #test_lengthMask = None 
   #print(f"Dataset {i}") 
   #[MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(args, sys_model[i], test_input, test_target, test_lengthMask=test_lengthMask)


##################################
### Hyper - KalmanNet Pipeline ###
##################################
### train and test KalmanNet on dataset i
#i = 0
print(f"KalmanNet pipeline start")
for i in range(len(SoW)):
    KalmanNet_model = KNet_mnet()
    KalmanNet_model.NNBuild(sys_model[0], args)
    print("Number of trainable parameters for KalmanNet on dataset ", i," : ",sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))

## Train Neural Network
KalmanNet_model = KNet_mnet()
KalmanNet_model.NNBuild(sys_model[0], args)
KalmanNet_Pipeline = Pipeline_EKF_MAML(strTime, "KNet", "KalmanNet")
#KalmanNet_Pipeline.setssModel(sys_model[i]) #don't use it in MAML_train
KalmanNet_Pipeline.setModel(KalmanNet_model)
KalmanNet_Pipeline.setTrainingParams(args)
KalmanNet_Pipeline.MAML_train_second(SoW_train_range, sys_model, cv_input_list, cv_target_list, train_input_list, train_target_list, path_results, 
                              cv_init_list, train_init_list, args)

#for i in range(len(SoW)):
   #print(f"Dataset {i}") 
   #KalmanNet_Pipeline.NNTest_alldatasets(sys_model, test_input_list[i][0], test_target_list[i][0], path_results)

## Close wandb run
if args.wandb_switch: 
   wandb.finish() 