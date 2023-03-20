import torch
from torch.nn import Sequential, Linear, Softmax, ReLU, Sigmoid


class WeightNet(torch.nn.Module):

    def __init__(self, inDim=0) -> None:
        super().__init__()
        self.linear1 = Linear(inDim, 32)
        self.linear2 = Linear(32, inDim)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        x3 = self.softmax(x2)
        x4 = x3.mul(x)
        return x4 + x


class GEV(torch.nn.Module):
    def __init__(self, dimIn=0) -> None:
        super().__init__()
        self.encoder = Sequential(
            Linear(dimIn, 32),
            Linear(32, 16),
            Linear(16, 8),
            ReLU()
        )
        self.decoder = Sequential(
            Linear(8, 8),
            Linear(8, 16),
            Linear(16, 32),
            Linear(32, dimIn),
            Sigmoid()
        )
        self.weightnet = WeightNet(dimIn)

    def dist(self, x, x_dot):
        Euclid_dist = torch.nn.PairwiseDistance(p=2)
        euclid_dist = Euclid_dist(x, x_dot)
        return euclid_dist

    def gev(self,x):
        return torch.exp(-torch.exp(-x))

    def forward(self, x):
        weighted_v = self.weightnet(x)
        z = self.encoder(x)
        x_dot = self.decoder(z)
        d = self.dist(x, x_dot)
        concat = torch.cat([weighted_v, z, d.reshape(d.shape[0], -1)], 1)
        h1 = Linear(concat.shape[1], 32)(concat)
        h2 = Linear(32,2)(h1)
        return self.gev(h2)

x = torch.randn((100, 200), dtype=torch.float32)
gev = GEV(dimIn=200)
t = gev(x)
print(t.shape)