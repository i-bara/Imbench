"""
imgagn generator
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, dim):
        super(Generator, self).__init__( )

        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, dim)
        self.fc4 = nn.Tanh()

        self.reg_params = list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.fc3.parameters())
        self.non_reg_params = self.fc4.parameters()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        x = (x+1)/2
        return x


def create_generator(dim):
    return Generator(dim=dim)


class GeneratorDRGCN(nn.Module):
    def __init__(self, n_cls):
        super(GeneratorDRGCN, self).__init__( )

        self.fc1 = nn.Linear(20 + n_cls, 100)
        self.fc2 = nn.Linear(100, n_cls)
        self.fc3 = nn.Tanh()

        self.reg_params = list(self.fc1.parameters())
        self.non_reg_params = list(self.fc2.parameters()) + list(self.fc3.parameters())

    def forward(self, z, y):
        x = F.relu(self.fc1(torch.cat((z, y), dim=1)))
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class DiscriminatorDRGCN(nn.Module):
    def __init__(self, n_cls):
        super(DiscriminatorDRGCN, self).__init__( )

        self.fc1 = nn.Linear(n_cls + n_cls, 100)
        self.fc2 = nn.Linear(100, 1)

        self.reg_params = list(self.fc1.parameters())
        self.non_reg_params = self.fc2.parameters()

    def forward(self, x, y):
        x = F.relu(self.fc1(torch.cat((x, y), dim=1)))
        x = self.fc2(x)
        return x


def create_gan_drgcn(n_cls):
    return GeneratorDRGCN(n_cls=n_cls), DiscriminatorDRGCN(n_cls=n_cls)
