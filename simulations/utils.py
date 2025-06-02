"""
The file contains utility functions for the simulations.
"""

import torch
from sklearn.decomposition import PCA
from simulations.real_data.nwb_interface import NWBDataset
import numpy as np
from pynwb import NWBHDF5IO
import pandas as pd

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
    T = args.T
    total_samples = target.shape[0]
    n = spikes.shape[1]
    m = target.shape[1]

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
    N_val = int(args.per_cv * N)
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

def load_mc_maze_train():
    return NWBDataset("/content/drive/MyDrive/Bachelor_Project_Adaptive-KNet/real_data/MC_maze/", "MC-maze_train", split_heldout=False)

def load_mc_rtt_train():
    return NWBDataset("/content/drive/MyDrive/Bachelor_Project_Adaptive-KNet/real_data/MC_RTT/", "MC-RTT_train", split_heldout=False)

def load_area2_bump_train():
    return NWBDataset("/content/drive/MyDrive/Bachelor_Project_Adaptive-KNet/real_data/Area2_BUMP/", "Area2-BUMP_train", split_heldout=False)

def extract_mc_maze(args, max_trials=30):
    """
    Manually preprocess dataset similar to make_trial_data, with:
    - 1-sample binning
    - 80ms lag (8 bins at 100Hz)
    - PCA on smoothed spikes

    Parameters:
    - args: argparse-style object with args.k_pca and others
    - dataset_loader: function returning an NWBDataset instance
    - max_trials: int, number of trials to process (for speed/memory control)

    Returns:
    - spikes_pca: np.ndarray of shape (total_time, pca_dim)
    - target_all: np.ndarray of shape (total_time, 4)
    """
    lag_ms = 80
    gauss_width = 50

    # Load and smooth
    dataset = load_mc_maze_train()
    dataset.smooth_spk(gauss_width, name=f"smth_{gauss_width}")

    # Limit number of trials
    dataset.trial_info = dataset.trial_info.iloc[:max_trials]
    trial_info = dataset.trial_info

    spikes_list, targets_list = [], []

    for _, trial in trial_info.iterrows():
        align_time = trial["move_onset_time"]

        # Define trial window: 80 ms before to 400 ms after onset
        start_time = align_time - pd.to_timedelta(lag_ms, unit="ms")
        end_time = align_time + pd.to_timedelta(400, unit="ms")

        # Slice the continuous data
        trial_slice = dataset.data.loc[start_time:end_time]
        if len(trial_slice) == 0:
            continue

        spikes = trial_slice[(f"spikes_smth_{gauss_width}",)].to_numpy()

        try:
            pos = trial_slice[("hand_pos",)].to_numpy()
            vel = trial_slice[("hand_vel",)].to_numpy()
        except KeyError:
            pos = trial_slice[("finger_pos",)].to_numpy()
            vel = trial_slice[("finger_vel",)].to_numpy()

        target = np.hstack([pos, vel])

        # Drop invalid (NaN) rows
        valid = ~np.isnan(spikes).any(axis=1) & ~np.isnan(target).any(axis=1)
        spikes, target = spikes[valid], target[valid]

        if spikes.shape[0] == 0:
            print("Row is NaN")
            continue

        spikes_list.append(spikes)
        targets_list.append(target)

    if not spikes_list:
        raise ValueError("No valid trials found after filtering.")

    spikes_all = np.vstack(spikes_list)
    target_all = np.vstack(targets_list)

    # PCA
    pca = PCA(n_components=args.k_pca)
    spikes_pca = pca.fit_transform(spikes_all)

    return spikes_pca, target_all

def extract_mc_rtt(args, max_trials=30):
    """
    Preprocess the MC_RTT dataset with:
    - 1-sample binning
    - 120ms lag
    - PCA on smoothed spikes
    """
    lag_ms = 120
    gauss_width = 50

    dataset = load_mc_rtt_train()
    dataset.smooth_spk(gauss_width, name=f"smth_{gauss_width}")

    dataset.trial_info = dataset.trial_info.iloc[:max_trials]
    trial_info = dataset.trial_info

    spikes_list, targets_list = [], []

    for _, trial in trial_info.iterrows():
        align_time = trial["start_time"]

        start_time = align_time - pd.to_timedelta(lag_ms, unit="ms")
        end_time = align_time + pd.to_timedelta(500, unit="ms")

        trial_slice = dataset.data.loc[start_time:end_time]
        if len(trial_slice) == 0:
            continue

        spikes = trial_slice[(f"spikes_smth_{gauss_width}",)].to_numpy()

        try:
            pos = trial_slice[("hand_pos",)].to_numpy()
            vel = trial_slice[("hand_vel",)].to_numpy()
        except KeyError:
            pos = trial_slice[("finger_pos",)].to_numpy()[:, :2]
            vel = trial_slice[("finger_vel",)].to_numpy()

        target = np.hstack([pos, vel])
        valid = ~np.isnan(spikes).any(axis=1) & ~np.isnan(target).any(axis=1)
        spikes, target = spikes[valid], target[valid]

        if spikes.shape[0] == 0:
            print("Row is NaN")
            continue

        spikes_list.append(spikes)
        targets_list.append(target)

    if not spikes_list:
        raise ValueError("No valid trials found after filtering.")

    spikes_all = np.vstack(spikes_list)
    target_all = np.vstack(targets_list)

    pca = PCA(n_components=args.k_pca)
    spikes_pca = pca.fit_transform(spikes_all)

    return spikes_pca, target_all

def extract_area_2b(args, max_trials=30):
    """
    Preprocess the Area2_BUMP dataset with:
    - 1-sample binning
    - 40ms lag
    - PCA on smoothed spikes (gauss_width fixed at 40ms)
    """
    lag_ms = 40
    gauss_width = 40  # fixed as per dataset standard

    dataset = load_area2_bump_train()
    dataset.smooth_spk(gauss_width, name=f"smth_{gauss_width}")

    dataset.trial_info = dataset.trial_info.iloc[:max_trials]
    trial_info = dataset.trial_info

    spikes_list, targets_list = [], []

    for _, trial in trial_info.iterrows():
        align_time = trial["bump_time"]
        if pd.isna(align_time):
            continue

        start_time = align_time - pd.to_timedelta(lag_ms, unit="ms")
        end_time = align_time + pd.to_timedelta(350, unit="ms")

        trial_slice = dataset.data.loc[start_time:end_time]
        if len(trial_slice) == 0:
            continue

        spikes = trial_slice[(f"spikes_smth_{gauss_width}",)].to_numpy()

        try:
            pos = trial_slice[("hand_pos",)].to_numpy()
            vel = trial_slice[("hand_vel",)].to_numpy()
        except KeyError:
            pos = trial_slice[("finger_pos",)].to_numpy()
            vel = trial_slice[("finger_vel",)].to_numpy()

        target = np.hstack([pos, vel])
        valid = ~np.isnan(spikes).any(axis=1) & ~np.isnan(target).any(axis=1)
        spikes, target = spikes[valid], target[valid]

        if spikes.shape[0] == 0:
            print("Row is NaN")
            continue

        spikes_list.append(spikes)
        targets_list.append(target)

    if not spikes_list:
        raise ValueError("No valid trials found after filtering.")

    spikes_all = np.vstack(spikes_list)
    target_all = np.vstack(targets_list)

    pca = PCA(n_components=args.k_pca)
    spikes_pca = pca.fit_transform(spikes_all)

    return spikes_pca, target_all

def estimate_H(spikes, targets, clip_val: float = 1.0):
    X = spikes
    Y = targets
    H_est = torch.linalg.pinv(X) @ Y  # shape: (n, m)
    
    # Clip extreme values to prevent exploding gradients
    H_est = torch.clamp(H_est, min=-clip_val, max=clip_val)
    
    # Normalize H_est (optional: based on Frobenius norm)
    norm = torch.norm(H_est, p='fro')
    if norm > 0:
        H_est = H_est / norm

    return H_est

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
