import easydict
import torch
import numpy as np
import random

def get_arguments():                   
    args = easydict.EasyDict({
        # device setting
        'device': 0,
        'seed' : 42,

        # training setting
        'model' : 'SVM',
        'batch_size' : 128,
        'num_workers' : 2,
        'epoch' : 200,
        'num_cls' : 100,

        # optimizer & criterion
        'lr' : 0.01,
        'momentum' : 0.9,
        'weight_decay' : 1e-4,
        'nesterov' : True,

        # directory
        'data_path' : './',
        'save_path' : './',

        # etc
        'print_freq' : 10
    })
    return args

def setup(args):
    # for cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # for mac
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)
    return device