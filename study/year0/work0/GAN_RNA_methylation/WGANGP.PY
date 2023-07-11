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


class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            # layers.append(nn.Dropout(p=0.5))
            return layers

        self.model = nn.Sequential(
            *block(in_dim, in_dim // 2),
            *block(in_dim // 2, in_dim // 2 // 2),
            *block(in_dim // 2 // 2, in_dim // 2 // 2 // 2),
            nn.Linear(in_dim // 2 // 2 // 2, 1)
        )

    def forward(self, x):
        out = self.model(x)
        return out


class Generator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            # layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_dim, in_dim * 2),
            *block(in_dim * 2, in_dim * 2 * 2),
            *block(in_dim * 2 * 2, in_dim * 2 * 2 * 2),
            *block(in_dim * 2 * 2 * 2, out_dim),
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
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.train_opt.lr_g,
                                            betas=(self.train_opt.beta1, self.train_opt.beta2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.train_opt.lr_d,
                                            betas=(self.train_opt.beta1, self.train_opt.beta2))
        self.losses_G = []
        self.losses_D = []

    def cal_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(real_data.size(0), 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(self.device)
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
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

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

    def C2ST(self):
        self.discriminator.eval()
        self.generator.eval()
        len = self.train_data.shape[0]
        fake_data = self.generator(self.noise(len)).detach().cpu()
        c2st_x = np.concatenate([fake_data, self.train_data])
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
        epoch_gps = []
        self.discriminator.train()
        self.generator.train()
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
                epoch_losses_D.append(loss_D)
                epoch_gps.append(gp)
            # generator training
            self.optimizer_G.zero_grad()
            noise = self.noise(self.train_opt.batch_size)
            z = self.generator(noise)
            loss_G = -torch.mean(self.discriminator(z))
            loss_G.backward()
            self.optimizer_G.step()
            epoch_losses_G.append(loss_G)
        acc, pric = self.C2ST()
        if abs(acc - 0.5) <= self.train_opt.stop_threshold:
            torch.save(self.generator.state_dict(),
                       os.path.join(self.g_save_dir,
                                    f'generator_n_{n_epoch}_acc_{acc:.3f}.pth'))
            torch.save(self.discriminator.state_dict(),
                       os.path.join(self.d_save_dir,
                                    f'discriminator_n_{n_epoch}_acc_{acc:.3f}.pth' % acc))
            print("\n[Epoch %d/%d] [D loss: %f] [G loss: %f] [GP: %f] [C2ST: %f] " % (n_epoch, self.train_opt.n_epochs,
                                                                                      torch.mean(
                                                                                          torch.FloatTensor(
                                                                                              epoch_losses_D)),
                                                                                      torch.mean(
                                                                                          torch.FloatTensor(
                                                                                              epoch_losses_G)),
                                                                                      torch.mean(torch.FloatTensor(
                                                                                          epoch_gps)),
                                                                                      acc,
                                                                                      ))
        if (n_epoch % 5 == 0):
            print("\n[Epoch %d/%d] [D loss: %f] [G loss: %f] [GP: %f] [C2ST: %f] " % (n_epoch, self.train_opt.n_epochs,
                                                                                      torch.mean(
                                                                                          torch.FloatTensor(
                                                                                              epoch_losses_D)),
                                                                                      torch.mean(
                                                                                          torch.FloatTensor(
                                                                                              epoch_losses_G)),
                                                                                      torch.mean(torch.FloatTensor(
                                                                                          epoch_gps)),
                                                                                      acc,
                                                                                      ))
        self.losses_D.append(torch.mean(torch.FloatTensor(epoch_losses_D)).cpu().numpy())
        self.losses_G.append(torch.mean(torch.FloatTensor(epoch_losses_G)).cpu().numpy())
