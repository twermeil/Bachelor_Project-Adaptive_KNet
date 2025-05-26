from nlb_tools.nwb_interface import NWBDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import argparse
from meta import Meta
from Simulations.linear.linear_syntheticNShot import SyntheticNShot
from Simulations.real.RealNShot import RealNShot

## 1) split into train/test
## 2) concatenate all three train sets into a new one (save original train sets)
## 3) fit H on concatenated train sets
## 4) have f and H andfeed data to training part, each dataset as one task

def main(args):
    print(args)

    args.use_cuda = False
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print("Using GPU" if device.type == 'cuda' else "Using CPU")

    # Load datasets
    dataset1 = NWBDataset("data/MC_maze/", "*train", split_heldout=False); dataset1.load()
    dataset2 = NWBDataset("data/MC_RTT/", "*train", split_heldout=False); dataset2.load()
    dataset3 = NWBDataset("data/Area2_BUMP/", "*train", split_heldout=False); dataset3.load()

    hand_pos_mcmaze = dataset1.data['hand_pos']
    spikes1 = dataset1.data['spikes']
    cursor_pos = dataset2.data['cursor_pos']
    spikes2 = dataset2.data['spikes']
    hand_pos_a2b = dataset3.data['hand_pos']
    spikes3 = dataset3.data['spikes']

    # Apply PCA to each independently
    k = args.k_pca
    pca1 = PCA(n_components=k)
    spikes_pca1 = pca1.fit_transform(spikes1)
    pca2 = PCA(n_components=k)
    spikes_pca2 = pca2.fit_transform(spikes2)
    pca3 = PCA(n_components=k)
    spikes_pca3 = pca3.fit_transform(spikes3)

    # Convert to lists of torch tensors (per sequence)
    hand_pos_mcmaze = [torch.from_numpy(seq).float().to(device) for seq in hand_pos_mcmaze]
    spikespca1 = [torch.from_numpy(seq).float().to(device) for seq in spikes_pca1]

    cursor_pos = [torch.from_numpy(seq).float().to(device) for seq in cursor_pos]
    spikespca2 = [torch.from_numpy(seq).float().to(device) for seq in spikes_pca2]

    hand_pos_a2b = [torch.from_numpy(seq).float().to(device) for seq in hand_pos_a2b]
    spikespca3 = [torch.from_numpy(seq).float().to(device) for seq in spikes_pca3]

    input_list = spikespca1 + spikespca2 + spikespca3
    target_list = hand_pos_mcmaze + cursor_pos + hand_pos_a2b

    # Instantiate MAML meta-learner with real data loader
    db_train = RealNShot(input_list, target_list, k_shot=args.k_spt, q_query=args.q_qry, batch_size=args.task_num)
    maml = Meta(args, db_train, is_linear_net=True)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    weights = torch.load('./MAML_data/linear/train/weights.pt', weights_only=False)
    weights = torch.tensor(weights, device=device)

    for step in range(args.epoch):
        state_spt, obs_spt, state_qry, obs_qry, select_num = db_train.next()
        state_spt, obs_spt, state_qry, obs_qry = torch.from_numpy(state_spt), torch.from_numpy(obs_spt), \
            torch.from_numpy(state_qry), torch.from_numpy(obs_qry)
        epoch_weights = weights[select_num]
        state_spt, obs_spt, state_qry, obs_qry = state_spt.to(device), obs_spt.to(device), state_qry.to(device), obs_qry.to(device)
        # Pad if needed or convert to consistent format if batching across variable lengths
        epoch_weights = weights[select_num]

        if step <= args.epoch / 2:
            loss_dB, count_num = maml(state_spt, obs_spt, state_qry, obs_qry, epoch_weights)
        else:
            loss_dB, count_num = maml.forward_second(state_spt, obs_spt, state_qry, obs_qry)

        if step % 10 == 0:
            print(f'step: {step}  loss_dB: {loss_dB}  count_num: {count_num}')

        if step % 1000 == 0 and step != 0:
            torch.save(maml.base_net.state_dict(), f'./MAML_model/linear/basenet_{step}.pt')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, default=20001)
    argparser.add_argument('--n_way', type=int, default=8)
    argparser.add_argument('--k_spt', type=int, default=6)
    argparser.add_argument('--k_spt_test', type=int, default=30)
    argparser.add_argument('--q_qry', type=int, default=15)
    argparser.add_argument('--batch_size', type=int, default=4)
    argparser.add_argument('--task_num', type=int, default=16)
    argparser.add_argument('--meta_lr', type=float, default=1e-3)
    argparser.add_argument('--update_lr', type=float, default=0.05)
    argparser.add_argument('--update_step', type=int, default=4)
    argparser.add_argument('--update_step_test', type=int, default=6)
    argparser.add_argument('--use_cuda', type=int, default=False)
    argparser.add_argument('--k_pca', type=int, default=20)
    args = argparser.parse_args()

    main(args)