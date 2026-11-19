import torch
import torch.nn as nn


class InputNorm(nn.Module):
    def forward(self, x):
        power = x.pow(2).mean(1, keepdim=True)
        return torch.cat([x / power.sqrt(), torch.log10(power)], 1)


class BaseDOAModel(nn.Module):
    def __init__(self, d_model, input_norm=False):
        super().__init__()
        self.input_norm = InputNorm() if input_norm else nn.Identity()
        d_in = d_model + 1 if input_norm else d_model

        # Hidden layers (paper Fig. 1)
        self.net = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU())

        # Heads: number of sources, DOAs for K = 1..4
        self.classification_layer = nn.Sequential(nn.Linear(256, 4), nn.Softmax(dim=1))
        self.DOA_1 = nn.Sequential(nn.Linear(256, 1))
        self.DOA_2 = nn.Sequential(nn.Linear(256, 2))
        self.DOA_3 = nn.Sequential(nn.Linear(256, 3))
        self.DOA_4 = nn.Sequential(nn.Linear(256, 4))

    def forward(self, x):
        x = self.net(self.input_norm(x))
        return self.classification_layer(x), self.DOA_1(x), self.DOA_2(x), self.DOA_3(x), self.DOA_4(x)


class ResidualBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, d)
        self.bn = nn.BatchNorm1d(d)

    def forward(self, x):
        return x + torch.relu(self.bn(self.linear(x)))


class ResidualHeadModel(BaseDOAModel):
    def __init__(self, d_model, input_norm=False):
        super().__init__(d_model, input_norm)

        # DOA heads: 3 residual layers of 256, then K outputs
        self.DOA_1 = nn.Sequential(ResidualBlock(256), ResidualBlock(256), ResidualBlock(256), nn.Linear(256, 1))
        self.DOA_2 = nn.Sequential(ResidualBlock(256), ResidualBlock(256), ResidualBlock(256), nn.Linear(256, 2))
        self.DOA_3 = nn.Sequential(ResidualBlock(256), ResidualBlock(256), ResidualBlock(256), nn.Linear(256, 3))
        self.DOA_4 = nn.Sequential(ResidualBlock(256), ResidualBlock(256), ResidualBlock(256), nn.Linear(256, 4))
