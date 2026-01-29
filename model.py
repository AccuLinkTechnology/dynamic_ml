import torch
import torch.nn as nn


class LaserNet(nn.Module):
    """
    Robotics-friendly small CNN:
      - GroupNorm instead of BatchNorm (stable for small batch sizes and domain shift)
      - modest depth/capacity
    """
    def __init__(self, in_channels=3, gn_groups=8):
        super().__init__()

        def gn(c):
            # groups must divide channels; fall back if needed
            g = gn_groups
            while c % g != 0 and g > 1:
                g //= 2
            return nn.GroupNorm(g, c)

        def block(cin, cout, k, s, p):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
                gn(cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
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
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x
