from matplotlib import pyplot as plt
import torch.nn as nn
import torch
import numpy as np
import math
from mpl_toolkits.axes_grid1 import host_subplot


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


# Define loss function
def mse_loss(predict, target):
    loss = torch.nn.MSELoss()
    return loss(predict, target)


def xs_gen(cell_list, batch_size, random):
    if random == 1:
         np.random.shuffle(cell_list)
    steps = math.ceil(len(cell_list) / batch_size)
    for i in range(steps):
        batch_x = cell_list[i * batch_size: i * batch_size + batch_size]
        yield i, batch_x


class AE(torch.nn.Module):
    """
    This class implements an autoencoder
    """
    def __init__(self, in_dim, latent_dim, device='cuda'):
        super(AE, self).__init__()
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(256, 128),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(128, 64),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(64, 128),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(128, 256),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(256, in_dim)
        )

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        y = self.decoder(z)
        return y

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)
        return z, y



