"""This file contains the settings for the simulation"""
import argparse

def general_settings():
    ### Dataset settings
        # Sizes
    parser = argparse.ArgumentParser(prog = 'KalmanNet',\
                                     description = 'Dataset, training and network parameters')
    parser.add_argument('--N_E', type=int, default=1000, metavar='trainset-size',
                        help='input training dataset size (# of sequences)')
    parser.add_argument('--N_CV', type=int, default=100, metavar='cvset-size',
                        help='input cross validation dataset size (# of sequences)')
    parser.add_argument('--N_T', type=int, default=200, metavar='testset-size',
                        help='input test dataset size (# of sequences)')
    parser.add_argument('--T', type=int, default=100, metavar='length',
                        help='input sequence length')
    parser.add_argument('--T_test', type=int, default=100, metavar='test-length',
                        help='input test sequence length')
    parser.add_argument('--k_pca', type=int, default=20, metavar='pca-dimension',
                        help='dimension for PCA')
    parser.add_argument('--per_cv', type=float, default=0.15, metavar='cross-validation-percentage',
                        help='percentage of data used for cv')
    parser.add_argument('--per_test', type=float, default=0.15, metavar='test-percentage',
                        help='percentage of data used for testing')
    parser.add_argument('--per_train', type=float, default=0.7, metavar='train-percentage',
                        help='percentage of data used for training')
    parser.add_argument('--max_trials', type=int, default=50, metavar='max-trials',
                        help='max trials used of real data')
    
        # Random length
    parser.add_argument('--randomLength', type=bool, default=False, metavar='rl',
                    help='if True, random sequence length')
    parser.add_argument('--T_max', type=int, default=1000, metavar='maximum-length',
                    help='if random sequence length, input max sequence length')
    parser.add_argument('--T_min', type=int, default=100, metavar='minimum-length',
                help='if random sequence length, input min sequence length')
        # Random initial state
    parser.add_argument('--randomInit_train', type=bool, default=False, metavar='ri_train',
                        help='if True, random initial state for training set')
    parser.add_argument('--randomInit_cv', type=bool, default=False, metavar='ri_cv',
                        help='if True, random initial state for cross validation set')
    parser.add_argument('--randomInit_test', type=bool, default=False, metavar='ri_test',
                        help='if True, random initial state for test set')
    parser.add_argument('--variance', type=float, default=100, metavar='variance',
                        help='input variance for the random initial state with uniform distribution')
    parser.add_argument('--init_distri', type=str, default='normal', metavar='init distribution',
                        help='input distribution for the random initial state (uniform/normal)')
        # Random noise (process/measurement) 
    parser.add_argument('--proc_noise_distri', type=str, default='normal', metavar='process noise distribution',
                        help='input distribution for process noise (normal/exponential)')
    parser.add_argument('--meas_noise_distri', type=str, default='normal', metavar='measurement noise distribution',
                        help='input distribution for measurement noise (normal/exponential)')


    ### Training settings
    parser.add_argument('--wandb_switch', type=bool, default=False, metavar='wandb',
                        help='if True, use wandb')
    parser.add_argument('--use_cuda', type=bool, default=False, metavar='CUDA',
                        help='if True, use CUDA')
    parser.add_argument('--mixed_dataset', type=bool, default=False, metavar='mixed_dataset',
                        help='if True, use mixed dataset training')
    
    parser.add_argument('--n_steps', type=int, default=1000, metavar='N_steps',
                        help='number of training steps (default: 1000)')
    parser.add_argument('--n_batch', type=int, default=20, metavar='N_B',
                        help='input batch size for training (default: 20)')
    parser.add_argument('--n_batch_list', type=list, default=[20], metavar='N_B_list',
                        help='input batch size for mixed dataset training')
    parser.add_argument('--n_batch_list_query', type=list, default=[20], metavar='N_B_list_query',
                        help='input query set batch size for mixed dataset training')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--update_lr', type=float, default=1e-3, metavar='Update_LR',
                        help='update learning rate for MAML (default: 1e-3)')
    parser.add_argument('--meta_lr', type=float, default=1e-3, metavar='Meta_LR',
                        help='meta learning rate for MAML (default: 1e-3)')
    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--knet_wd', type=float, default=1e-4, metavar='KNET_WD',
                        help='Kalmannet weight decay (default: 1e-4)')
    parser.add_argument('--hypernet_wd', type=float, default=1e-4, metavar='Hypernet_WD',
                        help='hypernetwork weight decay (default: 1e-4)')
    parser.add_argument('--spt_percentage', type=float, default=0.3, metavar='SPT_Percentage',
                        help='spt_percentage for MAML (default: 0.3)')
    parser.add_argument('--update_step', type=int, default=5, metavar='Update_step',
                        help='update steps for MAML (default: 5)')
    parser.add_argument('--maml_wd', type=list, default=[0.3, 0.3, 0.2, 0.1, 0.1, 0.01], metavar='Maml_Weight_Decay',
                        help='weight decay for MAML')
    
    parser.add_argument('--grid_size_dB', type=float, default=1, metavar='grid_size_dB',
                        help='input grid size for grid search of SoW in dB')
    parser.add_argument('--forget_factor', type=float, default=0.3, metavar='forget_factor',
                        help='input forget factor for noise covariance Q and R estimation')
    parser.add_argument('--max_iter', type=int, default=100, metavar='max_iter',
                        help='input maximum iteration for noise estimation')
    parser.add_argument('--SoW_conv_error', type=float, default=1e-3, metavar='SoW_conv_error',
                        help='input convergence error for SoW estimation')
    
    parser.add_argument('--CompositionLoss', type=bool, default=False, metavar='loss',
                        help='if True, use composition loss')
    parser.add_argument('--alpha', type=float, default=0.3, metavar='alpha',
                        help='input alpha [0,1] for the composition loss')
    parser.add_argument('--RobustScaler', type=bool, default=False, metavar='RobustScaler',
                        help='if True, use Robust Scaling for the losses of different datasets')
    parser.add_argument('--UnsupervisedLoss', type=bool, default=False, metavar='UnsupervisedLoss',
                        help='if True, use Unsupervised Loss for training')

    
    ### KalmanNet settings
    parser.add_argument('--in_mult_KNet', type=int, default=5, metavar='in_mult_KNet',
                        help='input dimension multiplier for KNet')
    parser.add_argument('--out_mult_KNet', type=int, default=40, metavar='out_mult_KNet',
                        help='output dimension multiplier for KNet')
    parser.add_argument('--use_context_mod', type=bool, default=False, metavar='use_context_mod',
                        help='if True, add context modulation layer to KNet')
    parser.add_argument('--knet_trainable', type=bool, default=False, metavar='knet_trainable',
                        help='if True, KNet is trainable, if False, KNet weights are generated by HyperNetwork')
    
    ### HyperNetwork settings
    parser.add_argument('--hnet_hidden_size_discount', type=int, default=100, metavar='discount factor on hnet hidden size',
                        help='hidden dimension divider for FC - GRU - FC HyperNetwork') 
    parser.add_argument('--hnet_arch', type=str, default='deconv', metavar='hnet_arch',
                        help='architecture for HyperNetwork (deconv/GRU)')

    args = parser.parse_args()
    return args
