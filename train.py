import os
import time
import random
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import LaserDataset
from model import LaserNetSimple
from config import (
    DATA_ROOT, DEFAULT_IMG_SIZE_HW,
    IN_CHANNELS, OUT_DIM, USE_TANH_BOUNDING, OUT_SCALE,
)

VAL_SEQS = os.environ.get("VAL_SEQS", "seq5,seq23,seq20").split(",")
VAL_SEQS = [s.strip() for s in VAL_SEQS if s.strip()]

SEED = int(os.environ.get("SEED", "123"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
EPOCHS = int(os.environ.get("EPOCHS", "50"))
LR = float(os.environ.get("LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
PIN_MEMORY = os.environ.get("PIN_MEMORY", "1") == "1"

RUN_DIR = os.path.join("runs_train3", time.strftime("%Y%m%d_%H%M%S"))
os.makedirs(RUN_DIR, exist_ok=True)


def list_seqs(root: str) -> List[str]:
    return sorted([d for d in os.listdir(root) if d.startswith("seq") and os.path.isdir(os.path.join(root, d))])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    total_loss = 0.0
    n = 0
    for x, y, _seq, _path in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(1, n)


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    all_seqs = list_seqs(DATA_ROOT)
    train_seqs = [s for s in all_seqs if s not in set(VAL_SEQS)]
    print(f"Found {len(all_seqs)} seqs. Train={len(train_seqs)} Val={len(VAL_SEQS)}")

    train_ds = LaserDataset(DATA_ROOT, train_seqs, img_size_hw=DEFAULT_IMG_SIZE_HW, strict=False)
    val_ds = LaserDataset(DATA_ROOT, VAL_SEQS, img_size_hw=DEFAULT_IMG_SIZE_HW, strict=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    model = LaserNetSimple(
        in_channels=IN_CHANNELS,
        out_dim=OUT_DIM,
        use_tanh_bounding=USE_TANH_BOUNDING,
        out_scale=OUT_SCALE,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    best_val = float("inf")
    best_path = os.path.join(RUN_DIR, "best_model.pt")

    torch.save(
        dict(
            data_root=DATA_ROOT,
            img_size_hw=DEFAULT_IMG_SIZE_HW,
            in_channels=IN_CHANNELS,
            out_dim=OUT_DIM,
            use_tanh_bounding=USE_TANH_BOUNDING,
            out_scale=OUT_SCALE,
            val_seqs=VAL_SEQS,
            seed=SEED,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        ),
        os.path.join(RUN_DIR, "config.pt"),
    )

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        n = 0
        t0 = time.time()

        for x, y, _seq, _path in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = total / max(1, n)
        val_loss = evaluate(model, val_loader, device)

        dt = time.time() - t0
        print(f"E{epoch:03d} | train={train_loss:.5f} val={val_loss:.5f} | {dt:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss},
                best_path,
            )
            print(f"  ✓ Best val: {best_val:.5f}")

    print(f"Done. Best model: {best_path}")


if __name__ == "__main__":
    main()
