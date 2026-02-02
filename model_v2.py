# model_v2.py
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Simple residual block with GroupNorm."""
    def __init__(self, channels, gn_groups=8):
        super().__init__()
        g = gn_groups
        while channels % g != 0 and g > 1:
            g //= 2
        
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.gn1 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.gn2 = nn.GroupNorm(g, channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class LaserNetV2(nn.Module):
    """
    Improved architecture for visual servoing:
    - Deeper backbone with residual connections
    - Single command head (no pose supervision)
    - Spatial attention for better feature extraction
    - Bounded outputs for safety
    """
    def __init__(self, in_channels=6, gn_groups=8, out_scale=(2.6, 2.6), dropout=0.1):
        super().__init__()
        self.out_scale = torch.tensor(out_scale, dtype=torch.float32)
        
        def gn(c):
            g = gn_groups
            while c % g != 0 and g > 1:
                g //= 2
            return nn.GroupNorm(g, c)
        
        def conv_block(cin, cout, k, s, p):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
                gn(cout),
                nn.ReLU(inplace=True),
            )
        
        # Encoder with progressive downsampling
        self.stem = conv_block(in_channels, 32, 5, 2, 2)  # -> 32, H/2, W/2
        
        self.layer1 = nn.Sequential(
            conv_block(32, 64, 3, 2, 1),  # -> 64, H/4, W/4
            ResidualBlock(64, gn_groups),
        )
        
        self.layer2 = nn.Sequential(
            conv_block(64, 128, 3, 2, 1),  # -> 128, H/8, W/8
            ResidualBlock(128, gn_groups),
            ResidualBlock(128, gn_groups),
        )
        
        self.layer3 = nn.Sequential(
            conv_block(128, 256, 3, 2, 1),  # -> 256, H/16, W/16
            ResidualBlock(256, gn_groups),
        )
        
        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Command head
        self.cmd_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, 6, H, W] input (current + diff)
        Returns:
            cmd: [B, 2] bounded command predictions
        """
        # Feature extraction
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # [B, 256, H/16, W/16]
        
        # Spatial attention
        att = self.spatial_att(x)  # [B, 1, H/16, W/16]
        x = x * att
        
        # Global pooling
        x = self.pool(x).flatten(1)  # [B, 256]
        
        # Command prediction
        cmd_raw = self.cmd_head(x)  # [B, 2]
        
        # Bounded output for safety
        cmd = torch.tanh(cmd_raw) * self.out_scale.to(cmd_raw.device)
        
        return cmd


class LaserNetSimple(nn.Module):
    """
    Simpler baseline model (similar to original but without pose head).
    Use this if the deeper V2 model overfits.
    """
    def __init__(self, in_channels=6, gn_groups=8, out_scale=(2.6, 2.6)):
        super().__init__()
        self.out_scale = torch.tensor(out_scale, dtype=torch.float32)
        
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
        
        # Single command head
        self.cmd_head = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 2),
        )
        
    def forward(self, x):
        f = self.features(x)
        f = self.pool(f).flatten(1)  # [B, 128]
        cmd_raw = self.cmd_head(f)  # [B, 2]
        cmd = torch.tanh(cmd_raw) * self.out_scale.to(cmd_raw.device)
        return cmd
