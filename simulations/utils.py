"""
The file contains utility functions for the simulations.
"""

import torch
from sklearn.decomposition import PCA

def DataGen(args, SysModel_data, fileName):

    ##################################
    ### Generate Training Sequence ###
    ##################################
    SysModel_data.GenerateBatch(args, args.N_E, args.T, randomInit=args.randomInit_train)
    train_input = SysModel_data.Input
    train_target = SysModel_data.Target
    ### init conditions ###
    train_init = SysModel_data.m1x_0_batch #size: N_E x m x 1
    ### length mask ###
    if args.randomLength:
        train_lengthMask = SysModel_data.lengthMask

    ####################################
    ### Generate Validation Sequence ###
    ####################################
    SysModel_data.GenerateBatch(args, args.N_CV, args.T, randomInit=args.randomInit_cv)
    cv_input = SysModel_data.Input
    cv_target = SysModel_data.Target
    cv_init = SysModel_data.m1x_0_batch #size: N_CV x m x 1
    ### length mask ###
    if args.randomLength:
        cv_lengthMask = SysModel_data.lengthMask

    ##############################
    ### Generate Test Sequence ###
    ##############################
    SysModel_data.GenerateBatch(args, args.N_T, args.T_test, randomInit=args.randomInit_test)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target
    test_init = SysModel_data.m1x_0_batch #size: N_T x m x 1
    ### length mask ###
    if args.randomLength:
        test_lengthMask = SysModel_data.lengthMask

    #################
    ### Save Data ###
    #################
    if(args.randomLength):
        torch.save([train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init, train_lengthMask,cv_lengthMask,test_lengthMask], fileName)
    else:
        torch.save([train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init], fileName)
        
def SplitData(args, spikes, target):
    total_samples = target.shape[0]
    # Calculate split indices
    train_end = int(args.per_train * total_samples)
    val_end = train_end + int(args.val_train * total_samples)

    # Split target (N x m x T) = (N x 4 x 100)
    target_train = target[:train_end] 
    target_val = target[train_end:val_end]
    target_test = target[val_end:]
    
    # Split spikes (N x n x T) = (N x 30 x 100)
    spikes_train = spikes[:train_end] 
    spikes_val = spikes[train_end:val_end] 
    spikes_test = spikes[val_end:]
    # use n = 30 (pca dimension)
    #split into N trials of length T = 100
    #dimension (N, n, T) N = number of trials (Ntrain... in args), n is pca dimension, T timelength of one trajectory (args.T)

    # Extract init sequences (assume first time step, or change if needed)
    train_init = target_train[:, :, 0].unsqueeze(-1)  # shape (N_train, m, 1)
    cv_init = target_val[:, :, 0].unsqueeze(-1)       # shape (N_val, m, 1)
    test_init = target_test[:, :, 0].unsqueeze(-1)    # shape (N_test, m, 1)

    return spikes_train, target_train, train_init, \
           spikes_val, target_val, cv_init, \
           spikes_test, target_test, test_init

def SplitData(args, spikes, target):
    print("Data Splitting:")
    T = args.T
    total_samples = target.shape[0]
    n = spikes.shape[1]
    m = target.shape[1]
    print("n = ", n, ", m = ", m)

    # Drop remainder so we can reshape evenly
    usable_len = (total_samples // T) * T
    spikes = spikes[:usable_len]
    target = target[:usable_len]

    # Reshape: (N, T, n) → (N, n, T)
    N = usable_len // T
    spikes = spikes.view(N, T, n).permute(0, 2, 1)   # (N, n, T)
    target = target.view(N, T, m).permute(0, 2, 1)   # (N, m, T)

    # Calculate split indices over N trials
    N_train = int(args.per_train * N)
    N_val = int(args.val_train * N)
    N_test = N - N_train - N_val

    # Split
    spikes_train = spikes[:N_train]
    spikes_val = spikes[N_train:N_train+N_val]
    spikes_test = spikes[N_train+N_val:]

    target_train = target[:N_train]
    target_val = target[N_train:N_train+N_val]
    target_test = target[N_train+N_val:]

    # Initial state from first time step of each trial
    train_init = target_train[:, :, 0].unsqueeze(-1)
    val_init = target_val[:, :, 0].unsqueeze(-1)
    test_init = target_test[:, :, 0].unsqueeze(-1)

    return (spikes_train, target_train, train_init,
            spikes_val, target_val, val_init,
            spikes_test, target_test, test_init)
              
def DecimateData(all_tensors, t_gen,t_mod, offset=0):
    
    # ratio: defines the relation between the sampling time of the true process and of the model (has to be an integer)
    ratio = round(t_mod/t_gen)

    i = 0
    all_tensors_out = all_tensors
    for tensor in all_tensors:
        tensor = tensor[:,(0+offset)::ratio]
        if(i==0):
            all_tensors_out = torch.cat([tensor], dim=0).view(1,all_tensors.size()[1],-1)
        else:
            all_tensors_out = torch.cat([all_tensors_out,tensor.view(1,all_tensors.size()[1],-1)], dim=0)
        i += 1

    return all_tensors_out

def Decimate_and_perturbate_Data(true_process, delta_t, delta_t_mod, N_examples, h, lambda_r, offset=0):
    
    # Decimate high resolution process
    decimated_process = DecimateData(true_process, delta_t, delta_t_mod, offset)

    noise_free_obs = getObs(decimated_process,h)

    # Replicate for computation purposes
    decimated_process = torch.cat(int(N_examples)*[decimated_process])
    noise_free_obs = torch.cat(int(N_examples)*[noise_free_obs])


    # Observations; additive Gaussian Noise
    observations = noise_free_obs + torch.randn_like(decimated_process) * lambda_r

    return [decimated_process, observations]

def getObs(sequences, h):
    i = 0
    sequences_out = torch.zeros_like(sequences)
    # sequences_out = torch.zeros_like(sequences)
    for sequence in sequences:
        for t in range(sequence.size()[1]):
            sequences_out[i,:,t] = h(sequence[:,t])
    i = i+1

    return sequences_out

def Short_Traj_Split(data_target, data_input, T):### Random Init is automatically incorporated
    data_target = list(torch.split(data_target,T+1,2)) # +1 to reserve for init
    data_input = list(torch.split(data_input,T+1,2)) # +1 to reserve for init

    data_target.pop()# Remove the last one which may not fullfill length T
    data_input.pop()# Remove the last one which may not fullfill length T

    data_target = torch.squeeze(torch.cat(list(data_target), dim=0))#Back to tensor and concat together
    data_input = torch.squeeze(torch.cat(list(data_input), dim=0))#Back to tensor and concat together
    # Split out init
    target = data_target[:,:,1:]
    input = data_input[:,:,1:]
    init = data_target[:,:,0]
    return [target, input, init]
