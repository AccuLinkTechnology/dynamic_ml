import torch
import torch.nn as nn
import torch.nn.functional as F


class LaserNet(nn.Module):
    """
    Slightly stronger, better behaved than the original:
      - BatchNorm after each conv
      - more channels
      - still lightweight + fast
    """
    def __init__(self, in_channels=3):
        super().__init__()

        def block(cin, cout, k, s, p):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            block(in_channels, 32, 5, 2, 2),
            block(32, 64, 3, 2, 1),
            block(64, 128, 3, 2, 1),
            block(128, 128, 3, 1, 1),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x
