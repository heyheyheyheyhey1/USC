import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


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
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # 重构数据使用sigmoid激活函数
        )

    def forward(self, x):
        return self.model(x)


# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, latent_dim)

        self.decoder = Decoder(input_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate_data(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim).to("cuda:0")  # 从标准正态分布中采样潜在变量z
        with torch.no_grad():
            reconstructed_data = self.decoder(z).to("cpu")
        return reconstructed_data.cpu().numpy()

    def forward(self, x):
        z_mean_logvar = self.encoder(x)
        z_mean = z_mean_logvar[:, :self.latent_dim]
        z_logvar = z_mean_logvar[:, self.latent_dim:]
        z = self.reparameterize(z_mean, z_logvar)  # 采样潜在变量z
        recon_x = self.decoder(z)  # 解码器重构数据
        return recon_x, z_mean, z_logvar

    @staticmethod
    def train_one_epoch(model, x, optimizer, criterion):
        optimizer.zero_grad()
        recon_x, z_mean, z_logvar = model(x)

        # 计算重构损失和KL散度正则项
        recon_loss = criterion(recon_x, x)
        kl_divergence = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        loss = recon_loss + kl_divergence

        loss.backward()
        optimizer.step()
        return loss