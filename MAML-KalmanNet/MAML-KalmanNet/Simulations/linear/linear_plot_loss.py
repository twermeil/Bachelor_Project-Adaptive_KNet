import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from filter import Filter
from linear_syntheticNShot import SyntheticNShot
from state_dict_learner import Learner
from torch.distributions.multivariate_normal import MultivariateNormal

argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, help='epoch number for danse', default=500)
argparser.add_argument('--n_way', type=int, help='n way', default=5)
argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
argparser.add_argument('--k_spt_test', type=int, help='k shot for support set test', default=8)
argparser.add_argument('--q_qry', type=int, help='q shot for query set', default=15)
argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=16)
argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.0018)
argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=8)
argparser.add_argument('--batch_size', type=int, help='batch size for train MAML', default=4)
argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=16)
argparser.add_argument('--task_num_train', type=int, help='task num for train', default=25)
argparser.add_argument('--task_num_test', type=int, help='task num for test', default=15)
argparser.add_argument('--use_cuda', type=int, help='use GPU to accelerate training', default=False)

args = argparser.parse_args()
args.use_cuda = False

if args.use_cuda:
    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device('cuda')
    else:
        raise Exception("No GPU found, please set args.use_cuda = False")
else:
    print("Using CPU")
    device = torch.device('cpu')

Generate_data = True
data_path = '../../MAML_data/linear/'
model = SyntheticNShot(batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_shot_test=args.k_spt_test,
                       q_query=args.q_qry,
                       data_path=data_path,
                       use_cuda=args.use_cuda,
                       Is_GenData=False)
my_filter = Filter(args, model, is_linear_net=True)
loss_fn = torch.nn.MSELoss()
random_net = Learner(model.x_dim, model.y_dim, args, is_linear_net=True).to(device)
my_dict = torch.load('../../MAML_model/linear/basenet.pt')

torch.manual_seed(333)
np.random.seed(333)

seq_num = 200
seq_len = 50
noise_q = [-30., -25., -20., -15., -10., -5., 0.]
v = [-10.]

noise_num = len(v) * len(noise_q)
if Generate_data:

    state = torch.zeros((noise_num, seq_num, model.x_dim, seq_len))
    obs = torch.zeros((noise_num, seq_num, model.y_dim, seq_len))
    j = 0
    for q2 in noise_q:

        for elem in v:

            q2 = 10 ** (q2 / 10)
            cov_q = q2 * torch.eye(model.x_dim)
            r2 = q2 * 10 ** (-elem / 10)
            cov_r = r2 * torch.eye(model.y_dim)
            print("Generating Data: q2: " + str(q2) + "  r2: " + str(r2))
            state_mtx = torch.zeros((seq_num, model.x_dim, seq_len), device=device)
            obs_mtx = torch.zeros((seq_num, model.y_dim, seq_len), device=device)
            # x_prev = torch.zeros((seq_num, model.x_dim, 1), device=device)
            min_val = -1.0
            max_val = 1.0
            x_prev = (max_val - min_val) * torch.rand((seq_num, model.x_dim, 1), device=device) + min_val
            with torch.no_grad():
                for i in range(seq_len):
                    # state：
                    xt = model.f(x_prev)
                    x_mean = torch.zeros(seq_num, model.x_dim)
                    distrib = MultivariateNormal(loc=x_mean, covariance_matrix=cov_q)
                    eq = distrib.rsample().view(seq_num, model.x_dim, 1).to(device)
                    xt = torch.add(xt, eq)
                    # measurement：
                    yt = model.H.matmul(xt)
                    y_mean = torch.zeros(seq_num, model.y_dim)
                    distrib = MultivariateNormal(loc=y_mean, covariance_matrix=cov_r)
                    er = distrib.rsample().view(seq_num, model.y_dim, 1).to(device)
                    yt = torch.add(yt, er)
                    x_prev = xt.clone()
                    state_mtx[:, :, i] = torch.squeeze(xt, 2)
                    obs_mtx[:, :, i] = torch.squeeze(yt, 2)

                state[j] = state_mtx
                obs[j] = obs_mtx
                j += 1
    torch.save(state, '../../MAML_data/linear/plot_data/state.pt')
    torch.save(obs, '../../MAML_data/linear/plot_data/obs.pt')
else:
    state = torch.load('../../MAML_data/linear/plot_data/state.pt')
    obs = torch.load('../../MAML_data/linear/plot_data/obs.pt')

for m in range(len(v)):
    selected_task_num = np.random.choice(seq_num, args.task_num_train + args.task_num_test, False)

    state_spt = state[m*len(noise_q):(m+1)*len(noise_q), selected_task_num[:args.task_num_train], :, :].to(device)
    obs_spt = obs[m*len(noise_q):(m+1)*len(noise_q), selected_task_num[:args.task_num_train], :, :].to(device)
    state_qry = state[m*len(noise_q):(m+1)*len(noise_q), selected_task_num[args.task_num_train:], :, :].to(device)
    obs_qry = obs[m*len(noise_q):(m+1)*len(noise_q), selected_task_num[args.task_num_train:], :, :].to(device)

    losses_dB_KF = []
    i = 0
    for state_qry_one, obs_qry_one in zip(state_qry, obs_qry):
        q2 = 10 ** (noise_q[i] / 10)
        r2 = q2 * 10 ** (-v[0] / 10)
        cov_q = q2 * torch.eye(model.x_dim)
        cov_r = r2 * torch.eye(model.y_dim)
        losses_dB_KF_test = my_filter.EKF(state_qry_one, obs_qry_one, cov_q, cov_r, use_initial_state=False)
        losses_dB_KF.append(losses_dB_KF_test.item())
        i = i + 1

    print('v= ' + str(v[m]) + ' KF loss(dB): ' + str(losses_dB_KF))

    losses_dB_last = []
    losses_dB = []
    losses_dB_8 = []

    for i, (state_spt_one, obs_spt_one, state_qry_one, obs_qry_one) in enumerate(zip(state_spt, obs_spt, state_qry, obs_qry)):

        my_filter.train_net.load_state_dict(my_dict)

        for k in range(args.update_step_test):

            loss = my_filter.compute_x_post(state_spt_one, obs_spt_one)
            my_filter.train_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(my_filter.train_net.parameters(), 5)
            my_filter.train_optim.step()

            with torch.no_grad():
                loss_q = my_filter.compute_x_post_qry(state_qry_one, obs_qry_one)
                losses_dB.append((10 * torch.log10(loss_q)).item())

        losses_dB_last.append(losses_dB[-1])
        losses_dB_8.append(losses_dB[-9])

    print('q2= ' + str(noise_q[m]) + ' Final update loss: ' + str(losses_dB_last))
    print('q2= ' + str(noise_q[m]) + ' 8 times update loss: ' + str(losses_dB_8))

    reversed_noise_q = [int(-x) for x in noise_q[::-1]]
    fontsize = 12

    # main figure
    plt.plot(reversed_noise_q, losses_dB_KF[::-1], color='red', label='KF', linestyle='--', marker='^',
             markerfacecolor='none', markeredgecolor='black', markersize=10)
    plt.plot(reversed_noise_q, losses_dB_8[::-1], label='MAML-KalmanNet (train rounds = 8)', marker='H',
             color='#FF7F01')
    plt.plot(reversed_noise_q, losses_dB_last[::-1], label='MAML-KalmanNet (train rounds = 16)', marker='^',
             color='#1F77B4')

    plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体
    plt.xlabel(r'$\dfrac{1}{q_2}[dB]$', fontsize=fontsize, fontweight='bold')
    plt.ylabel('MSE [dB]', fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc=3, prop={'size': 9.5})
    plt.grid()
    plt.tight_layout()

    # Zoomed-in view
    ax_inset = inset_axes(plt.gca(), width="45%", height="35%", loc="upper right")  # 插入小图
    start_idx = -2  # select last two points
    markersize = 5
    ax_inset.plot(reversed_noise_q[start_idx:], losses_dB_KF[::-1][start_idx:], color='red',
                  linestyle='--', marker='^', markerfacecolor='none', markeredgecolor='black', markersize=10)
    ax_inset.plot(reversed_noise_q[start_idx:], losses_dB_8[::-1][start_idx:], marker='H',
                  color='#FF7F01', markersize=markersize)
    ax_inset.plot(reversed_noise_q[start_idx:], losses_dB_last[::-1][start_idx:], marker='^',
                  color='#1F77B4', markersize=markersize)

    ax_inset.grid()
    ax_inset.tick_params(labelsize=8)

    plt.show()
