import torch
import argparse
from torch import nn
import pickle
import numpy as np
import pandas as pd
import tqdm


class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, in_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        out = self.model(x)
        return out


class Generator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.Linear(2048, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 2048),
            nn.Linear(2048, out_dim),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.model(x)
        return out


def initialize_weights(m):
    print("initialized")
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight.data, 0, 0.01)
        m.bias.data.zero_()


class WGAN():
    def __init__(self, args):
        self.train_data = args["train_data"]
        self.train_opt = args["train_opt"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        arg_g = {"in_dim": self.train_data.shape[1], "out_dim": self.train_data.shape[1]}
        arg_d = {"in_dim": self.train_data.shape[1]}
        self.generator = Generator(**arg_g).to(self.device).apply(initialize_weights)
        self.discriminator = Discriminator(**arg_d).to(self.device).apply(initialize_weights)
        self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=self.train_opt.lr)
        self.optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.train_opt.lr)
        self.losses_D = []
        self.losses_G = []

    def train_data_enumerate(self):
        for i in range(int(len(self.train_data) / self.train_opt.batch_size)):
            data_i = self.train_data[i * self.train_opt.batch_size: (i + 1) * self.train_opt.batch_size]
            yield torch.Tensor(data_i).to(self.device)

    def train(self):
        for i in tqdm.tqdm(range(self.train_opt.n_epochs)):
            self.train_one_epoch(i + 1)

    def train_one_epoch(self, n_epoch):
        for real_data in self.train_data_enumerate():
            # discriminator training
            for j in range(self.train_opt.n_critic):
                self.optimizer_D.zero_grad()
                # random noise
                noise = torch.randn(real_data.shape).to(self.device)
                # generator fake data through random noise
                fake_data = self.generator(noise).detach()
                # loss
                loss_D = torch.mean(self.discriminator(real_data)) - torch.mean(self.discriminator(fake_data))
                loss_D.backward()
                self.optimizer_D.step()
                for params in self.discriminator.parameters():
                    params.data.clamp_(-self.train_opt.clip_value, self.train_opt.clip_value)
                self.losses_D.append(loss_D)
            # generator training
            self.optimizer_G.zero_grad()
            noise = torch.randn(real_data.shape).to(self.device)
            gen_data = self.generator(noise)
            loss_G = torch.mean(self.discriminator(gen_data))
            loss_G.backward()
            self.optimizer_G.step()
            self.losses_G.append(loss_G)
        print("\n[Epoch %d/%d] [D loss: %f] [G loss: %f]" % (n_epoch, self.train_opt.n_epochs,
                                                             torch.mean(torch.FloatTensor(self.losses_D)),
                                                             torch.mean(torch.FloatTensor(self.losses_G))))
