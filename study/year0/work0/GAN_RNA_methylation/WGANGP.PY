import os

import torch
import argparse
from torch import nn
import pickle
import numpy as np
import pandas as pd
import tqdm
import time
import matplotlib.pyplot as plt


class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out = self.model(x)
        return out


class Generator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 1024),
            nn.Linear(1024, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, out_dim),

            nn.Tanh()
        )

    def forward(self, x):
        out = self.model(x)
        return out


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.zero_()


class WGANGP():
    def __init__(self, args):
        self.train_data = args["train_data"]
        self.train_opt = args["train_opt"]
        self.g_save_dir = args["g_sav_dir"]
        self.d_save_dir = args["d_sav_dir"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        arg_g = {"in_dim": self.train_opt.latent_dim, "out_dim": self.train_data.shape[1]}
        arg_d = {"in_dim": self.train_data.shape[1]}
        self.generator = Generator(**arg_g).to(self.device).apply(initialize_weights)
        self.discriminator = Discriminator(**arg_d).to(self.device).apply(initialize_weights)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.train_opt.lr,
                                            betas=(self.train_opt.beta1, self.train_opt.beta2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.train_opt.lr,
                                            betas=(self.train_opt.beta1, self.train_opt.beta2))
        self.losses_G = []
        self.losses_D = []

    def cal_gradient_penalty(self, real_data, fake_data):
        alpha = torch.Tensor(np.random.random((real_data.size(0), 1, 1, 1))).to(self.device)
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
        pass

    def train_data_enumerator(self):
        for i in range(int(len(self.train_data) / self.train_opt.batch_size)):
            data_i = self.train_data[i * self.train_opt.batch_size: (i + 1) * self.train_opt.batch_size]
            yield i, torch.Tensor(data_i).to(self.device)

    def train(self):
        for i in tqdm.tqdm(range(self.train_opt.n_epochs)):
            self.train_one_epoch(i + 1)
        self.draw()

    def draw(self):
        x = range(len(self.losses_D))
        plt.plot(x, self.losses_D, label='D_loss')
        plt.plot(x, self.losses_G, label='G_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join("train_loss.png"))

    def noise(self, num):
        return torch.rand([num, self.train_opt.latent_dim]).to(self.device)

    def train_one_epoch(self, n_epoch):
        epoch_losses_G = []
        epoch_losses_D = []
        for i, real_data in self.train_data_enumerator():
            # discriminator training
            for j in range(self.train_opt.n_critic):
                self.optimizer_D.zero_grad()
                # random noise
                noise = self.noise(self.train_opt.batch_size)
                # generator fake data through random noise
                fake_data = self.generator(noise).detach()
                # calculate gradient penalty
                gp = self.cal_gradient_penalty(real_data, fake_data)
                # loss
                loss_D = -torch.mean(self.discriminator(real_data)) + torch.mean(
                    self.discriminator(fake_data)) + self.train_opt.r * gp
                loss_D.backward()
                self.optimizer_D.step()
                # for params in self.discriminator.parameters():
                #     params.data.clamp_(-self.train_opt.clip_value, self.train_opt.clip_value)
                epoch_losses_D.append(loss_D)
            # generator training
            self.optimizer_G.zero_grad()
            noise = self.noise(self.train_opt.batch_size)
            z = self.generator(noise)
            loss_G = -torch.mean(self.discriminator(z))
            loss_G.backward()
            self.optimizer_G.step()
            epoch_losses_G.append(loss_G)
        if (n_epoch % 5 == 0):
            print("\n[Epoch %d/%d] [D loss: %f] [G loss: %f]" % (n_epoch, self.train_opt.n_epochs,
                                                                 torch.mean(torch.FloatTensor(epoch_losses_D)),
                                                                 torch.mean(torch.FloatTensor(epoch_losses_G))))
        if (n_epoch % 200 == 0):
            torch.save(self.generator.state_dict(),
                       os.path.join(self.g_save_dir,
                                    f'generator_n_{n_epoch}.pth'))
            torch.save(self.discriminator.state_dict(),
                       os.path.join(self.d_save_dir,
                                    f'discriminator_n_{n_epoch}.pth'))
        self.losses_D.append(torch.mean(torch.FloatTensor(epoch_losses_D)).cpu().numpy())
        self.losses_G.append(torch.mean(torch.FloatTensor(epoch_losses_G)).cpu().numpy())
