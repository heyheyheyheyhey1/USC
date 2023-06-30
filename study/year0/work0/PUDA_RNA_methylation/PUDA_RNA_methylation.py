import os

import torch
from torch import nn
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate, LeaveOneOut
from sklearn.metrics import make_scorer, accuracy_score, precision_score
from sklearn.utils import shuffle
from random import sample


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
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1)
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
            nn.Linear(1024, out_dim),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.model(x)
        return out


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()


class PUGAN():
    def __init__(self, args):
        self.positive_data = args["positive_data"]
        self.unlabel_data = args["unlabel_data"]
        self.train_opt = args["train_opt"]
        self.g_save_dir = args["g_sav_dir"]
        self.d_save_dir = args["d_sav_dir"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        arg_g = {"in_dim": self.train_opt.latent_dim, "out_dim": self.positive_data.shape[1]}
        arg_d = {"in_dim": self.positive_data.shape[1]}
        self.generator = Generator(**arg_g).to(self.device).apply(initialize_weights)
        self.discriminator = Discriminator(**arg_d).to(self.device).apply(initialize_weights)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.train_opt.lr_g,
                                            betas=(self.train_opt.beta1, self.train_opt.beta2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.train_opt.lr_d,
                                            betas=(self.train_opt.beta1, self.train_opt.beta2))
        self.losses_G = []
        self.losses_D = []

    def pur_loss(self,pred):
        prior = self.train_opt.prior
        p_above = - torch.nn.functional.logsigmoid(pred[:, 0]).mean()
        p_below = - torch.nn.functional.logsigmoid(-pred[:, 0]).mean()
        u = - torch.nn.functional.logsigmoid(pred[:, 0].unsqueeze(-1) - pred[:, 1:]).mean()
        if u > prior * p_below:
            return prior * p_above - prior * p_below + u
        else:
            return prior * p_above


    def train_data_enumerator(self):
        for i in range(int(len(self.positive_data) / self.train_opt.batch_size)):
            data_i = self.positive_data[i * self.train_opt.batch_size: (i + 1) * self.train_opt.batch_size]
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

    def get_unlabel(self):
        unlabel_batch = sample(list(self.unlabel_data),self.train_opt.neg_num * self.train_opt.batch_size)
        return torch.tensor(unlabel_batch,dtype=torch.float32).to(self.device)

    def C2ST(self):
        self.discriminator.eval()
        self.generator.eval()
        len = self.positive_data.shape[0]
        fake_data = self.generator(self.noise(len)).detach().cpu()
        c2st_x = np.concatenate([fake_data, self.positive_data])
        c2st_y = np.concatenate([np.zeros([len, ]), np.ones([len, ])])
        c2st_x, c2st_y = shuffle(c2st_x, c2st_y)
        SVM_model = SVC()
        loocv = LeaveOneOut()
        scorings = {'accuracy': make_scorer(accuracy_score),
                    'precision': make_scorer(precision_score),
                    }
        cv_result = cross_validate(SVM_model, c2st_x, c2st_y, scoring=scorings, cv=5)
        return cv_result["test_accuracy"].mean(), cv_result["test_precision"].mean()

    def train_one_epoch(self, n_epoch):
        epoch_losses_G = []
        epoch_losses_D = []
        self.discriminator.train()
        self.generator.train()
        for i, real_data in self.train_data_enumerator():

            # generator training
            self.optimizer_G.zero_grad()
            noise = self.noise(self.train_opt.neg_num * self.train_opt.batch_size)
            fake_data = self.generator(noise)
            unlabel = self.get_unlabel()
            fake_data = torch.concat([fake_data,unlabel])
            pred_fake = self.discriminator(fake_data.detach())
            pred_real = self.discriminator(real_data.detach())
            pred = torch.cat([pred_real.view(pred_real.shape[0], -1), pred_fake.view(pred_real.shape[0], -1)], dim=-1)
            loss_G = -self.pur_loss(pred)
            loss_G.backward()
            self.optimizer_G.step()
            epoch_losses_G.append(loss_G)


            # discriminator training
            # for j in range(self.train_opt.n_critic):
            self.optimizer_D.zero_grad()
            noise = self.noise(self.train_opt.neg_num * self.train_opt.batch_size)
            fake_data = self.generator(noise)
            unlabel = self.get_unlabel()
            fake_data = torch.concat([fake_data,unlabel])
            pred_fake = self.discriminator(fake_data)
            pred_real = self.discriminator(real_data)
            pred = torch.cat([pred_real.view(pred_real.shape[0], -1), pred_fake.view(pred_real.shape[0], -1)], dim=-1)
            loss_D = self.pur_loss(pred)
            loss_D.backward()
            self.optimizer_D.step()
            epoch_losses_D.append(loss_D)

        if (n_epoch % 5 == 0):
            torch.save(self.generator.state_dict(),
                       os.path.join(self.g_save_dir,
                                    f'generator_n_{n_epoch}.pth'))
            torch.save(self.discriminator.state_dict(),
                       os.path.join(self.d_save_dir,
                                    f'discriminator_n_{n_epoch}.pth'))
            print("\n[Epoch %d/%d] [D loss: %f] [G loss: %f]  " % (n_epoch, self.train_opt.n_epochs,
                                                                                      torch.mean(
                                                                                          torch.FloatTensor(
                                                                                              epoch_losses_D)),
                                                                                      torch.mean(
                                                                                          torch.FloatTensor(
                                                                                              epoch_losses_G)),

                                                                                      ))
        self.losses_D.append(torch.mean(torch.FloatTensor(epoch_losses_D)).cpu().numpy())
        self.losses_G.append(torch.mean(torch.FloatTensor(epoch_losses_G)).cpu().numpy())
