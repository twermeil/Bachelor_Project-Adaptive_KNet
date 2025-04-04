"""
This file contains the class Pipeline_EKF, 
which is used to train and test KalmanNet.
"""

import torch
import torch.nn as nn
import random
import time
from Plot import Plot_KF
import math
from mnets.KNet_mnet_allCM import KalmanNetNN as KNet_mnet


    
class Pipeline_EKF_MAML:

    def __init__(self, Time, folderName, modelName): #where to save best model for further loading
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel): #sysmodel
        self.ssModel = ssModel
    
    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, args):
        self.args = args
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.N_steps = args.n_steps  # Number of Training Steps

        if args.mixed_dataset:
            self.N_B = args.n_batch_list # Number of Samples in Batch for each dataset
        else:
            self.N_B = args.n_batch # Number of Samples in Batch

        self.learningRate = args.lr # Learning Rate
        self.weightDecay = args.knet_wd # L2 Weight Regularization - Weight Decay
        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean') # Loss Function for single dataset and CV
        # Loss Function for multiple datasets

        self.loss_fn_train = nn.MSELoss(reduction='mean')
            
        self.update_lr = args.update_lr  # 0.4 = alpha
        self.meta_lr = args.meta_lr  # 0.001 = beta
        #new args param for tuning
        self.spt_percentage = args.spt_percentage # 0.3
        
        #self.model = KNet_mnet().to(self.device)

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.meta_optimizer = torch.optim.Adam(self.model.weights.parameters(), lr=self.meta_lr, weight_decay=self.weightDecay)
        
    def NNTrain_mixdatasets(self, SoW_train_range, SysModel, cv_input_tuple, cv_target_tuple, train_input_tuple, train_target_tuple, path_results, \
        cv_init, train_init, MaskOnState=False, train_lengthMask=None,cv_lengthMask=None):

        ### Optional: start training from previous checkpoint
        # model_weights = torch.load(path_results+'knet_best-model.pt', map_location=self.device) 
        # self.model.load_state_dict(model_weights)

        if self.args.wandb_switch: 
            import wandb
        
        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0
        # Init MSE Loss
        self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])
        self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_train_dB_epoch = torch.zeros([self.N_steps])

        # dataset size
        for i in SoW_train_range[:-1]:# except the last one
            assert(train_target_tuple[i][0].shape[1]==train_target_tuple[i+1][0].shape[1])
            assert(train_input_tuple[i][0].shape[1]==train_input_tuple[i+1][0].shape[1])
            assert(train_input_tuple[i][0].shape[2]==train_input_tuple[i+1][0].shape[2])
            # check all datasets have the same m, n, T   
        sysmdl_m = train_target_tuple[0][0].shape[1] # state x dimension
        sysmdl_n = train_input_tuple[0][0].shape[1] # input y dimension
        sysmdl_T = train_input_tuple[0][0].shape[2] # sequence length 
        
        if MaskOnState:
            mask = torch.tensor([True,False,False])
            if SysModel.m == 2: 
                mask = torch.tensor([True,False])

        ##############
        ### Epochs ###
        ##############

        for ti in range(0, self.N_steps):
            # each turn, go through all datasets
            #################
            ### Training  ###
            #################    
            self.optimizer.zero_grad()        
            # Training Mode
            self.model.train()
            MSE_trainbatch_linear_LOSS = torch.zeros([len(train_target_tuple)]) # loss for each dataset
            
            for i in SoW_train_range: # dataset i 
                self.model.batch_size = self.N_B[i]
                # Init Training Batch tensors
                y_training_batch = torch.zeros([self.N_B[i], sysmdl_n, sysmdl_T]).to(self.device)
                train_target_batch = torch.zeros([self.N_B[i], sysmdl_m, sysmdl_T]).to(self.device)
                x_out_training_batch = torch.zeros([self.N_B[i], sysmdl_m, sysmdl_T]).to(self.device)
                # Init Sequence
                train_init_batch = torch.empty([self.N_B[i], sysmdl_m,1]).to(self.device)
                # Init Hidden State
                self.model.init_hidden()  
                # SoW: make sure SoWs are consistent
                assert torch.allclose(cv_input_tuple[i][1], cv_target_tuple[i][1]) 
                assert torch.allclose(train_input_tuple[i][1], train_target_tuple[i][1]) 
                self.model.UpdateSystemDynamics(SysModel[i])
                # req grad
                train_input_tuple[i][1].requires_grad = True # SoW_train
                train_input_tuple[i][0].requires_grad = True # input y
                train_target_tuple[i][0].requires_grad = True # target x
                train_init[i].requires_grad = True # init x0
                # data size
                self.N_E = len(train_input_tuple[i][0]) # Number of Training Sequences
                # mask on state
                if MaskOnState:
                    mask = torch.tensor([True,False,False])
                    if sysmdl_m == 2: 
                        mask = torch.tensor([True,False])
                # Randomly select N_B training sequences
                assert self.N_B[i] <= self.N_E # N_B must be smaller than N_E
                n_e = random.sample(range(self.N_E), k=self.N_B[i])
                dataset_index = 0
                for index in n_e:
                    # Training Batch
                    if self.args.randomLength:
                        y_training_batch[dataset_index,:,train_lengthMask[i][index,:]] = train_input_tuple[i][0][index,:,train_lengthMask[index,:]]
                        train_target_batch[dataset_index,:,train_lengthMask[i][index,:]] = train_target_tuple[i][0][index,:,train_lengthMask[index,:]]
                    else:
                        y_training_batch[dataset_index,:,:] = train_input_tuple[i][0][index]
                        train_target_batch[dataset_index,:,:] = train_target_tuple[i][0][index]                                 
                    # Init Sequence
                    train_init_batch[dataset_index,:,0] = torch.squeeze(train_init[i][index])                  
                    dataset_index += 1
                self.model.InitSequence(train_init_batch, sysmdl_T)
                
                # Forward Computation
                for t in range(0, sysmdl_T):
                    if self.args.use_context_mod:
                        x_out_training_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(y_training_batch[:, :, t],2), train_input_tuple[i][1]))
                    else:
                        x_out_training_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(y_training_batch[:, :, t],2)))
                
                # Compute Training Loss
                if (self.args.CompositionLoss):
                    y_hat = torch.zeros([self.N_B[i], sysmdl_n, sysmdl_T])
                    for t in range(sysmdl_T):
                        y_hat[:,:,t] = torch.squeeze(SysModel[i].h(torch.unsqueeze(x_out_training_batch[:,:,t],2)))

                    if(MaskOnState):### FIXME: composition loss, y_hat may have different mask with x
                        
                        MSE_trainbatch_linear_LOSS[i] = self.alpha * self.loss_fn_train(x_out_training_batch[:,mask,:], train_target_batch[:,mask,:])+(1-self.alpha)*self.loss_fn_train(y_hat[:,mask,:], y_training_batch[:,mask,:])
                    else:# no mask on state
                        
                        MSE_trainbatch_linear_LOSS[i] = self.alpha * self.loss_fn_train(x_out_training_batch, train_target_batch)+(1-self.alpha)*self.loss_fn_train(y_hat, y_training_batch)
                else:# no composition loss
                    if(MaskOnState):    
                        MSE_trainbatch_linear_LOSS[i] = self.loss_fn_train(x_out_training_batch[:,mask,:], train_target_batch[:,mask,:])
                    else: # no mask on state                   
                        MSE_trainbatch_linear_LOSS[i] = self.loss_fn_train(x_out_training_batch, train_target_batch)
                        
                #compute theta prime here (in for loop -> for all datasets)

            # averaged Loss over all datasets           
            weights = torch.tensor(self.N_B, dtype=torch.float32) / sum(self.N_B) # weights according to batch mixture of different datasets  
            MSE_trainbatch_linear_LOSS_average = (MSE_trainbatch_linear_LOSS * weights).mean()                         
            self.MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS_average.item()
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################
            MSE_trainbatch_linear_LOSS_average.backward(retain_graph=True) # computes gradient of loss
            #save original theta before updating it 
            self.optimizer.step() #updates parameters theta

            #################################
            ### Validation Sequence Batch ###
            #################################
            # Cross Validation Mode
            self.model.eval()
            # data size
            self.N_CV = len(cv_input_tuple[i][0])
            sysmdl_T_test = cv_input_tuple[i][0].shape[2] 
            # loss for each dataset
            MSE_cvbatch_linear_LOSS = torch.zeros([len(cv_target_tuple)])                     
            # Update Batch Size
            self.model.batch_size = self.N_CV 

            with torch.no_grad():
                for i in SoW_train_range: # dataset i 
                    if self.args.randomLength:
                        MSE_cv_linear_LOSS = torch.zeros([self.N_CV])
                    # Init Output
                    x_out_cv_batch = torch.empty([self.N_CV, sysmdl_m, sysmdl_T_test]).to(self.device)

                    # Init Hidden State
                    self.model.init_hidden()              
                    
                    # Init Sequence                    
                    self.model.InitSequence(cv_init[i], sysmdl_T_test)                       
                    
                    for t in range(0, sysmdl_T_test):
                        if self.args.use_context_mod:
                            x_out_cv_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(cv_input_tuple[i][0][:, :, t],2), cv_input_tuple[i][1]))
                        else:
                            x_out_cv_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(cv_input_tuple[i][0][:, :, t],2)))
                    
                    # Compute CV Loss
                    if(MaskOnState):
                        if self.args.randomLength:
                            for index in range(self.N_CV):
                                MSE_cv_linear_LOSS[index] = self.loss_fn(x_out_cv_batch[index,mask,cv_lengthMask[i][index]], cv_target_tuple[i][0][index,mask,cv_lengthMask[index]])
                            MSE_cvbatch_linear_LOSS[i] = torch.mean(MSE_cv_linear_LOSS)
                        else:          
                            MSE_cvbatch_linear_LOSS[i] = self.loss_fn(x_out_cv_batch[:,mask,:], cv_target_tuple[i][0][:,mask,:])
                    else:
                        if self.args.randomLength:
                            for index in range(self.N_CV):
                                MSE_cv_linear_LOSS[index] = self.loss_fn(x_out_cv_batch[index,:,cv_lengthMask[i][index]], cv_target_tuple[i][0][index,:,cv_lengthMask[index]])
                            MSE_cvbatch_linear_LOSS[i] = torch.mean(MSE_cv_linear_LOSS)
                        else:
                            MSE_cvbatch_linear_LOSS[i] = self.loss_fn(x_out_cv_batch, cv_target_tuple[i][0])

                # Print loss for each dataset in train range    
                for i in SoW_train_range:
                    MSE_cvbatch_dB_LOSS_i = 10 * math.log10(MSE_cvbatch_linear_LOSS[i].item())
                    print(f"MSE Validation on dataset {i}:", MSE_cvbatch_dB_LOSS_i,"[dB]")
                
                # averaged dB Loss
                MSE_cvbatch_linear_LOSS = MSE_cvbatch_linear_LOSS.sum() / len(SoW_train_range)
                self.MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])
                # save model with best averaged loss on all datasets
                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    
                    torch.save(self.model.state_dict(), path_results + 'knet_best-model.pt')

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training Average:", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti],
                  "[dB]")
                      
            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")
            
            ### Optinal: record loss on wandb
            if self.args.wandb_switch:
                wandb.log({
                    "train_loss": self.MSE_train_dB_epoch[ti],
                    "val_loss": self.MSE_cv_dB_epoch[ti]})
            ###
        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]
    
    def MAML_train(self, SoW_train_range, SysModel, cv_input_tuple, cv_target_tuple, train_input_tuple, train_target_tuple, path_results,
                   cv_init, train_init, args, MaskOnState=False, train_lengthMask=None,cv_lengthMask=None):

        task_num = len(SoW_train_range)
        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0
        # Init MSE Loss
        self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])
        #self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_train_dB_epoch = torch.zeros([self.N_steps])
        
        # dataset size
        for i in SoW_train_range[:-1]:# except the last one
            assert(train_target_tuple[i][0].shape[1]==train_target_tuple[i+1][0].shape[1])
            assert(train_input_tuple[i][0].shape[1]==train_input_tuple[i+1][0].shape[1])
            assert(train_input_tuple[i][0].shape[2]==train_input_tuple[i+1][0].shape[2])
            # check all datasets have the same m, n, T   
        sysmdl_m = train_target_tuple[0][0].shape[1] # state x dimension
        sysmdl_n = train_input_tuple[0][0].shape[1] # input y dimension
        sysmdl_T = train_input_tuple[0][0].shape[2] # sequence length 
        
        for ti in range(0, self.N_steps):
            # torch.autograd.set_detect_anomaly(True)
            count_num = task_num #initialized to task_num for each step
            meta_loss = torch.tensor(0.) #initialized to 0 for each step
            is_qry_nan = False
            gradients = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    gradients[name] = torch.zeros_like(param)

            for i in range(task_num):
                
                task_model = KNet_mnet().to(self.device) # = new theta_i'
                task_model.NNBuild(SysModel[i], args)
                task_model.batch_size = self.N_B[i]
                task_model.load_state_dict(self.model.state_dict()) #base_net = original theta
                #loading original theta to avoid using previous theta'
                # Init Hidden State
                task_model.init_hidden()
                inner_optimizer = torch.optim.Adam(task_model.weights.parameters(), lr=self.update_lr, weight_decay=self.weightDecay)
                # Init Training Batch tensors
                y_training_batch_spt = torch.zeros([self.N_B[i], sysmdl_n, sysmdl_T]).to(self.device)
                train_target_batch_spt = torch.zeros([self.N_B[i], sysmdl_m, sysmdl_T]).to(self.device)
                x_out_training_batch_spt = torch.zeros([self.N_B[i], sysmdl_m, sysmdl_T]).to(self.device)
                y_training_batch_query = torch.zeros([self.N_B[i], sysmdl_n, sysmdl_T]).to(self.device)
                train_target_batch_query = torch.zeros([self.N_B[i], sysmdl_m, sysmdl_T]).to(self.device)
                x_out_training_batch_query = torch.zeros([self.N_B[i], sysmdl_m, sysmdl_T]).to(self.device)
                # Init Sequence
                train_init_batch_spt = torch.empty([self.N_B[i], sysmdl_m,1]).to(self.device)
                train_init_batch_query = torch.empty([self.N_B[i], sysmdl_m,1]).to(self.device)  
                # SoW: make sure SoWs are consistent
                assert torch.allclose(cv_input_tuple[i][1], cv_target_tuple[i][1]) 
                assert torch.allclose(train_input_tuple[i][1], train_target_tuple[i][1]) 
                self.model.UpdateSystemDynamics(SysModel[i])
                # req grad
                train_input_tuple[i][1].requires_grad = True # SoW_train
                train_input_tuple[i][0].requires_grad = True # input y
                train_target_tuple[i][0].requires_grad = True # target x
                train_init[i].requires_grad = True # init x0
                # data size
                N_E_spt = int(len(train_input_tuple[i][0]) * self.spt_percentage) # Number of Support Training Sequences
                N_E_query = len(train_input_tuple[i][0])-N_E_spt # Number of Query Training Sequences
                
                # Randomly select N_B support training sequences
                assert self.N_B[i] <= min(N_E_spt, N_E_query) # N_B must be smaller than N_E
                n_e_spt = random.sample(range(N_E_spt), k=self.N_B[i])
                dataset_index = 0
                for index in n_e_spt:
                    # Training Batch
                    if self.args.randomLength:
                        y_training_batch_spt[dataset_index,:,train_lengthMask[i][index,:]] = train_input_tuple[i][0][index,:,train_lengthMask[index,:]]
                        train_target_batch_spt[dataset_index,:,train_lengthMask[i][index,:]] = train_target_tuple[i][0][index,:,train_lengthMask[index,:]]
                    else:
                        y_training_batch_spt[dataset_index,:,:] = train_input_tuple[i][0][index]
                        train_target_batch_spt[dataset_index,:,:] = train_target_tuple[i][0][index]                                 
                    # Init Sequence
                    train_init_batch_spt[dataset_index,:,0] = torch.squeeze(train_init[i][index])                  
                    dataset_index += 1
                
                # Randomly select N_B query training sequences
                assert self.N_B[i] <= min(N_E_spt, N_E_query) # N_B must be smaller than N_E
                n_e_query = random.sample(range(N_E_spt, N_E_spt + N_E_query), k=self.N_B[i])
                dataset_index = 0
                for index in n_e_query:
                    # Training Batch
                    if self.args.randomLength:
                        y_training_batch_query[dataset_index,:,train_lengthMask[i][index,:]] = train_input_tuple[i][0][index,:,train_lengthMask[index,:]]
                        train_target_batch_query[dataset_index,:,train_lengthMask[i][index,:]] = train_target_tuple[i][0][index,:,train_lengthMask[index,:]]
                    else:
                        y_training_batch_query[dataset_index,:,:] = train_input_tuple[i][0][index]
                        train_target_batch_query[dataset_index,:,:] = train_target_tuple[i][0][index]                                 
                    # Init Sequence
                    train_init_batch_query[dataset_index,:,0] = torch.squeeze(train_init[i][index])                  
                    dataset_index += 1
                
                MSE_trainbatch_linear_LOSS_spt = torch.zeros([len(train_target_tuple)]) # inner loss for each task on support set
                
                task_model.InitSequence(train_init_batch_spt, sysmdl_T)   
                # Forward Computation
                for t in range(0, sysmdl_T):
                    if self.args.use_context_mod:
                        x_out_training_batch_spt[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(y_training_batch_spt[:, :, t],2), train_input_tuple[i][1]))
                    else:
                        x_out_training_batch_spt[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(y_training_batch_spt[:, :, t],2)))
                    
                # Compute Training Loss
                # no composition loss, no mask                 
                MSE_trainbatch_linear_LOSS_spt[i] = self.loss_fn_train(x_out_training_batch_spt, train_target_batch_spt)           
                if torch.isnan(MSE_trainbatch_linear_LOSS_spt[i]).item():
                    count_num -= 1
                    continue

                inner_optimizer.zero_grad()
                MSE_trainbatch_linear_LOSS_spt[i].backward() #computes gradient of step 6
                torch.nn.utils.clip_grad_norm_(task_model.parameters(), 1) #clip on gradient to stabilize (clip the value)
                inner_optimizer.step() #model update (theta-graident) (theta_i' -> task_model)

                for k in range(1, self.update_step):
                    
                    self.model.InitSequence(train_init_batch_query, sysmdl_T)
                    # Forward Computation
                    for t in range(0, sysmdl_T):
                        if self.args.use_context_mod:
                            x_out_training_batch_spt[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(y_training_batch_spt[:, :, t],2), train_input_tuple[i][1]))
                        else:
                            x_out_training_batch_spt[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(y_training_batch_spt[:, :, t],2)))
                    
                    # Compute Training Loss
                    # no composition loss, no mask                 
                    MSE_trainbatch_linear_LOSS_spt[i] = self.loss_fn_train(x_out_training_batch_spt, train_target_batch_spt)           
                    if torch.isnan(MSE_trainbatch_linear_LOSS_spt[i]).item():
                        count_num -= 1
                        continue

                    inner_optimizer.zero_grad()
                    MSE_trainbatch_linear_LOSS_spt[i].backward() #computes gradient of step 6
                    torch.nn.utils.clip_grad_norm_(task_model.parameters(), 1) #clip on gradient to stabilize (clip the value)
                    inner_optimizer.step() #model update (theta-graident) (theta_i' -> task_model)

                    if torch.isnan(MSE_trainbatch_linear_LOSS_spt[i]).item():
                        count_num -= 1
                        is_qry_nan = True
                        break

                    inner_optimizer.zero_grad()
                    MSE_trainbatch_linear_LOSS_spt[i].backward()
                    torch.nn.utils.clip_grad_norm_(task_model.parameters(), 1)
                    inner_optimizer.step() #theta -> theta'

                if is_qry_nan:
                    is_qry_nan = False
                    continue

                task_model.init_hidden() # task model has theta'
                
                #compute the loss with theta'
                MSE_trainbatch_linear_LOSS_query = torch.zeros([len(train_target_tuple)]) # inner loss for each task on query set
                task_model.InitSequence(train_init_batch_query, sysmdl_T)   
                # Forward Computation
                for t in range(0, sysmdl_T):
                    if self.args.use_context_mod:
                        x_out_training_batch_query[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(y_training_batch_query[:, :, t],2), train_input_tuple[i][1]))
                    else:
                        x_out_training_batch_query[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(y_training_batch_query[:, :, t],2)))
                    
                # Compute Training Loss
                # no composition loss, no mask                 
                MSE_trainbatch_linear_LOSS_query[i] = self.loss_fn_train(x_out_training_batch_query, train_target_batch_query)           
                if torch.isnan(MSE_trainbatch_linear_LOSS_query[i]).item():
                    count_num -= 1
                    continue

                meta_loss = meta_loss + MSE_trainbatch_linear_LOSS_query[i]

            if count_num == 0:
                return 0, 0

            meta_loss = meta_loss/task_num
            meta_loss.backward() #equation 21 on paper, computes Gtb
            self.MSE_train_linear_epoch[ti] = meta_loss
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])
            
            for name, param in task_model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.clone()
            #zero out gradient of base net (theta)
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.zero_()

            for name, param in self.model.named_parameters():
                if name in gradients:
                    param.grad = gradients[name].clone()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.meta_optimizer.step() #computes equation 24
            
            #################################
            ### Validation Sequence Batch ###
            #################################
            # Cross Validation Mode
            self.model.eval()
            # data size
            self.N_CV = len(cv_input_tuple[i][0])
            sysmdl_T_test = cv_input_tuple[i][0].shape[2] 
            # loss for each dataset
            MSE_cvbatch_linear_LOSS = torch.zeros([len(cv_target_tuple)])                     
            # Update Batch Size
            self.model.batch_size = self.N_CV 

            with torch.no_grad():
                for i in task_num: # dataset i 
                    if self.args.randomLength:
                        MSE_cv_linear_LOSS = torch.zeros([self.N_CV])
                    # Init Output
                    x_out_cv_batch = torch.empty([self.N_CV, sysmdl_m, sysmdl_T_test]).to(self.device)

                    # Init Hidden State
                    self.model.init_hidden()              
                    
                    # Init Sequence                    
                    self.model.InitSequence(cv_init[i], sysmdl_T_test)                       
                    
                    for t in range(0, sysmdl_T_test):
                        if self.args.use_context_mod:
                            x_out_cv_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(cv_input_tuple[i][0][:, :, t],2), cv_input_tuple[i][1]))
                        else:
                            x_out_cv_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(cv_input_tuple[i][0][:, :, t],2)))
                    
                    # Compute CV Loss
                    #if(MaskOnState):
                        #if self.args.randomLength:
                            #for index in range(self.N_CV):
                                #MSE_cv_linear_LOSS[index] = self.loss_fn(x_out_cv_batch[index,mask,cv_lengthMask[i][index]], cv_target_tuple[i][0][index,mask,cv_lengthMask[index]])
                        #MSE_cvbatch_linear_LOSS[i] = torch.mean(MSE_cv_linear_LOSS)
                        #else:          
                            #MSE_cvbatch_linear_LOSS[i] = self.loss_fn(x_out_cv_batch[:,mask,:], cv_target_tuple[i][0][:,mask,:])
                    #else:
                        #if self.args.randomLength:
                            #for index in range(self.N_CV):
                                #MSE_cv_linear_LOSS[index] = self.loss_fn(x_out_cv_batch[index,:,cv_lengthMask[i][index]], cv_target_tuple[i][0][index,:,cv_lengthMask[index]])
                            #MSE_cvbatch_linear_LOSS[i] = torch.mean(MSE_cv_linear_LOSS)
                        #else:
                            #MSE_cvbatch_linear_LOSS[i] = self.loss_fn(x_out_cv_batch, cv_target_tuple[i][0])
                    if self.args.randomLength:
                            for index in range(self.N_CV):
                                MSE_cv_linear_LOSS[index] = self.loss_fn(x_out_cv_batch[index,:,cv_lengthMask[i][index]], cv_target_tuple[i][0][index,:,cv_lengthMask[index]])
                            MSE_cvbatch_linear_LOSS[i] = torch.mean(MSE_cv_linear_LOSS)
                    else:
                            MSE_cvbatch_linear_LOSS[i] = self.loss_fn(x_out_cv_batch, cv_target_tuple[i][0])
                            
                # Print loss for each dataset in train range    
                for i in SoW_train_range:
                    MSE_cvbatch_dB_LOSS_i = 10 * math.log10(MSE_cvbatch_linear_LOSS[i].item())
                    print(f"MSE Validation on dataset {i}:", MSE_cvbatch_dB_LOSS_i,"[dB]")
                
                # averaged dB Loss
                MSE_cvbatch_linear_LOSS = MSE_cvbatch_linear_LOSS.sum() / len(SoW_train_range)
                self.MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])
                # save model with best averaged loss on all datasets
                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    
                    torch.save(self.model.state_dict(), path_results + 'knet_best-model.pt')

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training Average:", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti],
                "[dB]")
                    
            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")
            
            ### Optinal: record loss on wandb
            if self.args.wandb_switch:
                wandb.log({
                    "train_loss": self.MSE_train_dB_epoch[ti],
                    "val_loss": self.MSE_cv_dB_epoch[ti]})
                
        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]
        
    def NNTest_alldatasets(self, SoW_test_range, sys_model, test_input_tuple, test_target_tuple, path_results,test_init,\
        MaskOnState=False,load_model=False,load_model_path=None, test_lengthMask=None):
        if self.args.wandb_switch: 
            import wandb
        # Load model weights
        if load_model:
            model_weights = torch.load(load_model_path, map_location=self.device) 
        else:
            model_weights = torch.load(path_results+'knet_best-model.pt', map_location=self.device) 
        # Set the loaded weights to the model
        # FIXME: if not NNTrain before, the model is not defined
        self.model.load_state_dict(model_weights)

        # dataset size    
        for i in SoW_test_range[:-1]:# except the last one
            assert(test_target_tuple[i][0].shape[1]==test_target_tuple[i+1][0].shape[1])
            assert(test_target_tuple[i][0].shape[2]==test_target_tuple[i+1][0].shape[2])
            # check all datasets have the same m, T   
        sysmdl_m = test_target_tuple[0][0].shape[1]
        sysmdl_T_test = test_target_tuple[0][0].shape[2]
        total_size = 0 # total size for all datasets
        for i in SoW_test_range: 
            total_size += test_input_tuple[i][0].shape[0] 
        self.MSE_test_linear_arr = torch.zeros([total_size])
        x_out_test = torch.zeros([total_size, sysmdl_m,sysmdl_T_test]).to(self.device)
        current_idx = 0

        for i in SoW_test_range: # dataset i   
            # SoW
            assert torch.allclose(test_input_tuple[i][1], test_target_tuple[i][1]) 
            SoW_test = test_input_tuple[i][1]
            self.model.UpdateSystemDynamics(sys_model[i])
            # load data
            test_input = test_input_tuple[i][0]
            test_target = test_target_tuple[i][0]
            # data size
            self.N_T = test_input.shape[0]
            
            if MaskOnState:
                mask = torch.tensor([True,False,False])
                if sysmdl_m == 2: 
                    mask = torch.tensor([True,False])

            # MSE LOSS Function
            loss_fn = nn.MSELoss(reduction='mean')

            # Test mode
            self.model.eval()
            self.model.batch_size = self.N_T
            # Init Hidden State
            self.model.init_hidden()

            torch.no_grad()

            start = time.time()

            # Init Sequence
            self.model.InitSequence(test_init[i], sysmdl_T_test)               

            for t in range(0, sysmdl_T_test):
                if self.args.use_context_mod:
                    x_out_test[current_idx:current_idx+self.N_T,:, t] = torch.squeeze(self.model(torch.unsqueeze(test_input[:,:, t],2), SoW_test))
                else:
                    x_out_test[current_idx:current_idx+self.N_T,:, t] = torch.squeeze(self.model(torch.unsqueeze(test_input[:,:, t],2)))
            
            end = time.time()
            t = end - start

            # MSE loss
            for j in range(self.N_T):# cannot use batch due to different length and std computation  
                if(MaskOnState):
                    if self.args.randomLength:
                        self.MSE_test_linear_arr[current_idx+j] = loss_fn(x_out_test[current_idx+j,mask,test_lengthMask[j]], test_target[j,mask,test_lengthMask[j]]).item()
                    else:
                        self.MSE_test_linear_arr[current_idx+j] = loss_fn(x_out_test[current_idx+j,mask,:], test_target[j,mask,:]).item()
                else:
                    if self.args.randomLength:
                        self.MSE_test_linear_arr[current_idx+j] = loss_fn(x_out_test[current_idx+j,:,test_lengthMask[j]], test_target[j,:,test_lengthMask[j]]).item()
                    else:
                        self.MSE_test_linear_arr[current_idx+j] = loss_fn(x_out_test[current_idx+j,:,:], test_target[j,:,:]).item()
            
            # Average for dataset i
            MSE_test_linear_avg_dataset_i = torch.mean(self.MSE_test_linear_arr[current_idx:current_idx+self.N_T])
            MSE_test_dB_avg_dataset_i = 10 * torch.log10(MSE_test_linear_avg_dataset_i)

            # Standard deviation for dataset i
            MSE_test_linear_std_dataset_i = torch.std(self.MSE_test_linear_arr[current_idx:current_idx+self.N_T], unbiased=True)

            # Confidence interval for dataset i
            test_std_dB_dataset_i = 10 * torch.log10(MSE_test_linear_std_dataset_i + MSE_test_linear_avg_dataset_i) - MSE_test_dB_avg_dataset_i

            # Print MSE and std for dataset i
            str = self.modelName + "-" + f"dataset {i}" + "-" + "MSE Test:"
            print(str, MSE_test_dB_avg_dataset_i, "[dB]")
            str = self.modelName + "-"  + f"dataset {i}" + "-" + "STD Test:"
            print(str, test_std_dB_dataset_i, "[dB]")
            # Print Run Time
            print("Inference Time:", t)

            ### Optinal: record loss on wandb
            if self.args.wandb_switch:
                wandb.log({f'test_loss for dataset {i}':MSE_test_dB_avg_dataset_i})
            ###

            # update index
            current_idx += self.N_T
        
        # average MSE over all datasets
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)
        # Average std
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg
        # Print MSE and std
        str = self.modelName + "-" + "Average" + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-"  + "Average" + "-" + "STD Test:"
        print(str, self.test_std_dB, "[dB]")

        ### Optinal: record loss on wandb
        if self.args.wandb_switch:
            wandb.log({f'averaged test loss':self.MSE_test_dB_avg})
        ###

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test]

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot_KF(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_steps, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)