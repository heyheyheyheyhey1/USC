import torch
import argparse
from torch import nn
import pickle
import numpy as np
import pandas as pd



class Discriminator(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(p.in_dim, 8192),
            nn.Linear(8192, 10240),
            nn.Linear(10240, 8192),
            nn.Linear(8192, p.in_dim),
            nn.Linear(p.in_dim, 2048),
            nn.Linear(2048,1024),
            nn.Linear(1024,512),
            nn.Linear(512,256),
            nn.Linear(256,1),
        )

    def forward(self, x):
        out = self.model(x)
        return out


class Generator(nn.Module):
    def __init__(self,p):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(p.in_dim,8192),
            nn.Linear(8192,8192),
            nn.Linear(8192,8192),
            nn.Linear(8192,8192),
            nn.Linear(8192,8192),
            nn.Linear(8192,p.out_dim)
        )

    def forward(self, x):
        out = self.model(x)
        return out

class WGAN():
    def __init__(self,args):
        self.train_data = args.train_data,
        self.arg_g = args.arg_g
        self.arg_d = args.arg_d
        pass
    def train(self):
        pass
