import numpy as np
import time
import torch
from torch import nn, optim
import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l

features, labels = d2l.get_data_ch7()
features.shape # torch.Size([1500, 5])

def sgd(params, states, hyperparams):
    for p in params:
        p.data -= hyperparams['lr'] * p.grad.data

def train_sgd(lr, batch_size, num_epochs=2):
    d2l.train_ch7(sgd, None, {'lr': lr}, features, labels, batch_size, num_epochs)
train_sgd(1, 1500, 6)

train_sgd(0.005, 1)

train_sgd(0.05, 10)

d2l.train_pytorch_ch7(optim.SGD, {"lr": 0.05}, features, labels, 10)