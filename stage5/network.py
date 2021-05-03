import torch
import torch.nn as nn
import numpy as np


class Network0(nn.Module):
    def __init__(self, in_features, n_actions):
        super(Network0, self).__init__()

        # VERSION: architecture from 'Playing Atari with Deep Reinforcement Learning'
        self.conv = nn.Sequential(
            nn.Conv2d(in_features[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU()
        )

        conv_out = self.conv(torch.zeros(1, *in_features))
        conv_out_size = int(np.prod(conv_out.size()))

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    # forward pass combining conv set and lin set
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x


class Network1(nn.Module):
    def __init__(self, in_features, n_actions):
        super(Network1, self).__init__()
        self.in_features = in_features

        # VERSION: adapted architecture
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features[0], out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.Flatten()
        )

        conv_out = self.conv3(self.conv2(self.conv1(torch.zeros(1, *in_features))))
        conv_out_size = int(np.prod(conv_out.size()))

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    # forward pass combining conv set and lin set
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)

        return x
