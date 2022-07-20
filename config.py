import argparse
import os
import torch
def load_options():
    """ configs for training"""
    parser = argparse.ArgumentParser(
        description='visual navigation pose estimation')
    # basic parameters
    basic_args = parser.add_argument_group('basic')
    
    basic_args.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    basic_args.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='batch size (default: 50)')
    basic_args.add_argument('--episode_nums', type=int, default=50000, metavar='N',
                        help='maximum episodes number')
    basic_args.add_argument('--cuda', type=bool, default=True,
                        help='run on CUDA ')
    basic_args.add_argument('--save_folder', type=str, default="./results/",
                        help='folder to save files and logs')
    basic_args.add_argument('--name', type=str, default="pose_estimation",
                        help='running task name')
    basic_args.add_argument('--resume', type=bool, default=False,
                        help='hidden size (default: 256)')
    basic_args.add_argument('--pretrain', type=str, default=None,
                        help='pretrain model path')
    basic_args.add_argument('--save_interval', type=int, default=1000,
                        help='model save interval')
    basic_args.add_argument('--log', type=bool, default=False,
                        help='wandb log')
    basic_args.add_argument('--log_interval', type=int, default=30,
                        help='wandb logging interval')
    basic_args.add_argument('--dataset', type=str, default='laptop',
                        help='datasets category (can/bottle/bowl/laptop/mug/camera)')
    # policy parameters
    policy_args = parser.add_argument_group('policy')
        
    policy_args.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    policy_args.add_argument('--eval', type=int, default=0,
                        help='Evaluates a policy a policy every 10 episode (0: no evaluation, 1: eval on synthetic data, 2: eval on nocs data)')
    policy_args.add_argument('--gamma', type=float, default=0.8, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    policy_args.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficiet(τ)')
    policy_args.add_argument('--lr', type=float, default=0.00003, metavar='G',
                        help='learning rate')
    policy_args.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    policy_args.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')
    policy_args.add_argument('--max_step', type=int, default=5, metavar='N',
                        help='maximum training steps')
    policy_args.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    policy_args.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    policy_args.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    policy_args.add_argument('--demo_episodes', type=int, default=100, metavar='N',
                        help='Steps sampling random actions (default: 27000)')
    policy_args.add_argument('--replay_size', type=int, default=5000, metavar='N',
                        help='size of replay buffer (default: 27000000)')
    policy_args.add_argument('--gd_optimize', type=bool, default=False,
                        help='use GD to optimize pose after IL')
    policy_args.add_argument('--use_encoder', type=bool, default=False,
                        help='use encoder to initialize latent code')
    policy_args.add_argument('--image_size', type=int, default=64,
                        help='processing image size')
    # nocs image generator params
    args = parser.parse_args()
    return args, parser
