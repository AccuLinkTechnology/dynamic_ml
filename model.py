import torch
import torch.nn as nn
import torch.nn.functional as F


class LaserNet(nn.Module):
    def __init__(self, in_ch: int = 6):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x
