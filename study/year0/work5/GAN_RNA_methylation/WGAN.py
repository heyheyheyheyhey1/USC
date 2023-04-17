import torch
import argparse
from torch import nn
import pickle
import numpy as np
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1800, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--feature_size', type=int, default=934, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
opt = parser.parse_args()

class Discriminator(nn.Module):
    def __init__(self):
        pass
    def forward(self,p):
        pass

class Generator(nn.Module):
    def __init__(self):
        pass
    def forward(self,p):
        pass
