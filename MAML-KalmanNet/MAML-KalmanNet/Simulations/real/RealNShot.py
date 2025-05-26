import math
import torch
import random
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

class RealNShot:
    def __init__(self, input_list, target_list, k_shot, q_query, batch_size):
        assert len(input_list) == len(target_list), "Inputs and targets must have same length!"
        self.input_list = input_list
        self.target_list = target_list
        self.k_shot = k_shot
        self.q_query = q_query
        self.batch_size = batch_size
        self.num_tasks = len(input_list)

    def next(self):
        indices = np.random.choice(self.num_tasks, size=self.batch_size, replace=False)

        state_spt, obs_spt, state_qry, obs_qry = [], [], [], []

        for idx in indices:
            input_seq = self.input_list[idx]
            target_seq = self.target_list[idx]

            seq_len = input_seq.shape[0]
            total_required = self.k_shot + self.q_query

            if seq_len < total_required:
                raise ValueError(f"Sequence too short for k_shot + q_query: {seq_len} < {total_required}")

            # Randomly permute to avoid order bias
            perm = torch.randperm(seq_len)
            spt_idx = perm[:self.k_shot]
            qry_idx = perm[self.k_shot:self.k_shot + self.q_query]

            obs_spt.append(input_seq[spt_idx])
            state_spt.append(target_seq[spt_idx])
            obs_qry.append(input_seq[qry_idx])
            state_qry.append(target_seq[qry_idx])

        return state_spt, obs_spt, state_qry, obs_qry, indices
    
    def f(self, x):
        ##constant velocity
        batched_F = self.F.view(1, self.F.shape[0], self.F.shape[1]).expand(x.shape[0], -1, -1).to(x.device)
        return torch.bmm(batched_F, x)
    
    def g(self, x):
        ## = H, use linear regression on real data to find H 
        if self.is_linear:
            return x.reshape((x.shape[0], x.shape[1], 1))
        else:
            y1 = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
            y2 = torch.arctan2(x[:, 1], x[:, 0])
            return torch.cat((y1, y2), dim=1).reshape((x.shape[0], x.shape[1], 1)).to(x.device)