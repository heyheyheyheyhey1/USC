import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

# Define the MMD loss function
def mmd_loss(x, y):
    x_mean = torch.mean(x, dim=0)
    y_mean = torch.mean(y, dim=0)
    x_cov = torch.mm((x - x_mean).t(), (x - x_mean))
    y_cov = torch.mm((y - y_mean).t(), (y - y_mean))
    loss = torch.norm(x_cov - y_cov, p='fro')
    return loss
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 2 * latent_dim)  # 输出均值和标准差
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self,input_dim, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # 重构数据使用sigmoid激活函数
        )
    def forward(self,x):
        return self.model(x)

# Define the VAE model with MMD regularization
class MMDVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(MMDVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim,latent_dim)
        self.decoder = Decoder(input_dim,latent_dim)
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc_output = self.encoder(x)
        mu, log_var = enc_output[:, :self.latent_dim], enc_output[:, self.latent_dim:]
        z = self.reparameterize(mu, log_var)
        dec_output = self.decoder(z)
        return dec_output, mu, log_var

    # Training loop
    @staticmethod
    def train_one_epoch(model, x, optimizer, criterion):
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(x)
        reconstruction_loss = criterion(recon_batch, x)
        mmd_term = mmd_loss(mu, log_var)  # Calculate MMD loss
        loss = reconstruction_loss + mmd_term
        loss.backward()
        optimizer.step()
        return loss

    def generate_data(self,num_samples):
        z = torch.randn(num_samples, self.latent_dim).to("cuda:0")  # 从标准正态分布中采样潜在变量z
        with torch.no_grad():
            reconstructed_data = self.decoder(z).to("cpu")
        return reconstructed_data.cpu().numpy()

