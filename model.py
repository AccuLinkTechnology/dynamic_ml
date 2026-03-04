import torch
import torch.nn as nn


class LaserNetSimple(nn.Module):
    """
    Simple CNN baseline (locked-in).

    - Input:  [B, 6, H, W]
    - Output: [B, out_dim] (default 4)

    Optional tanh bounding for safety; can be disabled.
    """
    def __init__(
        self,
        in_channels: int = 6,
        out_dim: int = 4,
        gn_groups: int = 8,
        use_tanh_bounding: bool = True,
        out_scale=(1.0, 1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.use_tanh_bounding = bool(use_tanh_bounding)
        self.register_buffer("out_scale", torch.tensor(out_scale, dtype=torch.float32))

        def gn(c):
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
            nn.Linear(128, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = self.pool(f).flatten(1)
        y = self.head(f)

        if self.use_tanh_bounding:
            y = torch.tanh(y) * self.out_scale

        return y