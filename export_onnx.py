"""
Usage: python3 export_onnx.py <path/to/best_model.pt>
Auto-detects old (global pool) vs new (spatial pool) architecture from the checkpoint.
"""

import torch
import torch.nn as nn
import sys


def gn(c, gn_groups=8):
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


class LaserNetOld(nn.Module):
    """Original global-pool architecture."""
    def __init__(self, in_channels=6, out_dim=4, use_tanh_bounding=True, out_scale=(1.,1.,1.,1.)):
        super().__init__()
        self.use_tanh_bounding = use_tanh_bounding
        self.register_buffer("out_scale", torch.tensor(out_scale, dtype=torch.float32))
        self.features = nn.Sequential(
            block(in_channels, 32, 5, 2, 2),
            block(32, 64, 3, 2, 1),
            block(64, 128, 3, 2, 1),
            block(128, 128, 3, 1, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(nn.Linear(128, 96), nn.ReLU(inplace=True), nn.Linear(96, out_dim))

    def forward(self, x):
        f = self.pool(self.features(x)).flatten(1)
        y = self.head(f)
        return torch.tanh(y) * self.out_scale if self.use_tanh_bounding else y


class LaserNetNew(nn.Module):
    """Spatial-pool architecture."""
    def __init__(self, in_channels=6, out_dim=4, use_tanh_bounding=True, out_scale=(1.,1.,1.,1.)):
        super().__init__()
        self.use_tanh_bounding = use_tanh_bounding
        self.register_buffer("out_scale", torch.tensor(out_scale, dtype=torch.float32))
        self.features = nn.Sequential(
            block(in_channels, 32, 5, 2, 2),
            block(32, 64, 3, 2, 1),
            block(64, 128, 3, 2, 1),
            block(128, 128, 3, 1, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 7))
        self.head = nn.Sequential(nn.Linear(128*4*7, 256), nn.ReLU(inplace=True), nn.Linear(256, out_dim))

    def forward(self, x):
        f = self.pool(self.features(x)).flatten(1)
        y = self.head(f)
        return torch.tanh(y) * self.out_scale if self.use_tanh_bounding else y


def detect_arch(state_dict):
    w = state_dict["head.0.weight"]
    if w.shape == (96, 128):
        return "old"
    elif w.shape == (256, 3584):
        return "new"
    else:
        raise ValueError(f"Unrecognised head shape: {w.shape}")


if __name__ == "__main__":
    from config import IN_CHANNELS, OUT_DIM, USE_TANH_BOUNDING, OUT_SCALE

    weights = sys.argv[1]
    ckpt = torch.load(weights, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt

    arch = detect_arch(state)
    print(f"Detected architecture: {arch}")

    kwargs = dict(in_channels=IN_CHANNELS, out_dim=OUT_DIM,
                  use_tanh_bounding=USE_TANH_BOUNDING, out_scale=OUT_SCALE)

    model = (LaserNetOld(**kwargs) if arch == "old" else LaserNetNew(**kwargs))
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy = torch.zeros(1, IN_CHANNELS, 180, 320)
    out = weights.replace(".pt", ".onnx")
    torch.onnx.export(model, dummy, out, opset_version=13,
                      input_names=["input"], output_names=["output"])
    print(f"Exported: {out}")