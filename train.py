from utils.regression_trainer import RegTrainer
import argparse
import os
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--exp-tag', default='',
                        help='the tag of the experiment')
    parser.add_argument('--dataset', default='RGBTCC', choices=['RGBTCC', 'DroneRGBT'],
                        help='the dataset to train, RGBTCC or DroneRGBT')
    parser.add_argument('--data-dir', default='/Crowd Counting/RGBT-CC',
                        help='training data directory')
    parser.add_argument('--save-dir', default='/Crowd Counting/RGBT-CC',
                        help='directory to save models.')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--crop-size', type=int, default=256,
                        help='defacult 256')
    
    # model setting
    parser.add_argument('--constr-hg', type=str, default='threshold', choices=['threshold', 'knn'],
                        help='the availabel methods for cross-modal hypergraph construction')
    parser.add_argument('--constr-k', type=int, default=4,
                        help='the k for knn method')
    parser.add_argument('--constr-threshold', type=float, default=0.8,
                        help='the threshold for threshold method, 0.8 is recommended for RGBTCC, 0.5 is recommended for DroneRGBT')

    # default training setting
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed to set')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=200,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='the num of steps to val')
    parser.add_argument('--val-start', type=int, default=20,
                        help='the epoch start to val')
    parser.add_argument('--test-epoch', type=int, default=1,
                        help='the num of steps to test')
    parser.add_argument('--test-start', type=int, default=20,
                        help='the epoch start to test')
    parser.add_argument('--save-all-best', type=bool, default=True,
                        help='whether to load opt state')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='train batch size')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')
    parser.add_argument('--downsample-ratio', type=int, default=8,
                        help='downsample ratio')
    parser.add_argument('--use-background', type=bool, default=True,
                        help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=8.0,
                        help='sigma for likelihood')
    parser.add_argument('--background-ratio', type=float, default=0.15,
                        help='background ratio')
    args = parser.parse_args()
    return args

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    
    

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()
