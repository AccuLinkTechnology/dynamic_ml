import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset

from dataset2 import LaserDataset
from model import LaserNet


# -------------------------
# CONFIG
# -------------------------
DATA_ROOT = "/home/nvidia/Documents/dynamic_ml/train2"
SEQS = ["seq1", "seq2", "seq3", "seq4", "seq5"]

# Validation strategy:
#   "holdout_seq"   : train on TRAIN_SEQS, validate on VAL_SEQS (honest; )
#   "per_seq_split" : 80/20 split inside each seq then concat (fast; optimistic esp. raster)
VAL_MODE = "holdout_seq"

# With seq1 much larger, a good first test is: train on seq1-4, validate on seq5.
TRAIN_SEQS = ["seq1", "seq2", "seq3", "seq4"]
VAL_SEQS = ["seq5"]

LABEL_NORM = "global"  # "global" | "per_seq" | "none"

BATCH_SIZE = 8
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
SEED = 42


def split_per_seq(full_ds: LaserDataset, val_frac: float = 0.2) -> Tuple[ConcatDataset, ConcatDataset]:
    """
    Split indices within each sequence so both train/val contain all sequences.
    NOTE: for raster data, this can be overly optimistic due to near-duplicates.
    """
    g = torch.Generator().manual_seed(SEED)
    train_parts, val_parts = [], []

    idx_by_seq: Dict[str, List[int]] = {s: [] for s in full_ds.seqs}
    for i, (_path, _label, seq) in enumerate(full_ds.samples):
        idx_by_seq[seq].append(i)

    for seq, idxs in idx_by_seq.items():
        if len(idxs) == 0:
            continue
        n_total = len(idxs)
        n_val = max(1, int(val_frac * n_total))
        perm = torch.randperm(n_total, generator=g).tolist()

        val_idx = [idxs[j] for j in perm[:n_val]]
        tr_idx = [idxs[j] for j in perm[n_val:]]

        train_parts.append(Subset(full_ds, tr_idx))
        val_parts.append(Subset(full_ds, val_idx))

    return ConcatDataset(train_parts), ConcatDataset(val_parts)


def save_label_stats(stats_ds: LaserDataset, out_path: str = "label_stats.pt") -> dict:
    payload = {
        "label_norm": stats_ds.label_norm,
        "global": {"mean": stats_ds.global_stats.mean, "std": stats_ds.global_stats.std},
        "per_seq": {k: {"mean": v.mean, "std": v.std} for k, v in stats_ds.seq_stats.items()},
    }
    torch.save(payload, out_path)
    print(f"Saved {out_path}")
    return payload


def unnormalize(pred_norm: torch.Tensor, seq: List[str], stats: dict) -> torch.Tensor:
    """
    pred_norm: [B,2] CPU tensor
    returns: [B,2] CPU tensor in raw motor units
    """
    mode = stats["label_norm"]
    if mode == "none":
        return pred_norm
    if mode == "global":
        mean = stats["global"]["mean"]
        std = stats["global"]["std"]
        return pred_norm * std + mean
    if mode == "per_seq":
        out = []
        for i, s in enumerate(seq):
            m = stats["per_seq"][s]["mean"]
            sd = stats["per_seq"][s]["std"]
            out.append(pred_norm[i] * sd + m)
        return torch.stack(out, dim=0)
    raise ValueError(f"Unknown label_norm={mode}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if VAL_MODE == "holdout_seq":
        train_ds = LaserDataset(DATA_ROOT, TRAIN_SEQS, label_norm=LABEL_NORM)
        val_ds = LaserDataset(DATA_ROOT, VAL_SEQS, label_norm=LABEL_NORM)
        stats_ds = LaserDataset(DATA_ROOT, TRAIN_SEQS, label_norm=LABEL_NORM)  # stats from training distribution
        print("Train seqs:", TRAIN_SEQS, "n=", len(train_ds))
        print("Val seqs:", VAL_SEQS, "n=", len(val_ds))
    elif VAL_MODE == "per_seq_split":
        full_ds = LaserDataset(DATA_ROOT, SEQS, label_norm=LABEL_NORM)
        train_ds, val_ds = split_per_seq(full_ds, val_frac=0.2)
        stats_ds = full_ds
        print("Full seqs:", SEQS, "n=", len(full_ds))
        print("Train n=", len(train_ds), "Val n=", len(val_ds))
    else:
        raise ValueError(f"Unknown VAL_MODE={VAL_MODE}")

    # Save normalization stats used for inference and for MAE reporting
    stats = save_label_stats(stats_ds, "label_stats.pt")

    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

    model = LaserNet().to(device)
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")

    for epoch in range(EPOCHS):
        t0 = time.time()

        # ---- train ----
        model.train()
        train_loss = 0.0
        for x, y_norm, _seq, _y_raw in train_loader:
            x = x.to(device, non_blocking=True)
            y_norm = y_norm.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y_norm)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        mae_sum = torch.zeros(2)
        n = 0

        with torch.no_grad():
            for x, y_norm, seq, y_raw in val_loader:
                x = x.to(device, non_blocking=True)
                y_norm = y_norm.to(device, non_blocking=True)

                pred_norm = model(x)
                loss = criterion(pred_norm, y_norm)
                val_loss += loss.item() * x.size(0)

                pred_raw = unnormalize(pred_norm.cpu(), seq, stats)
                mae_sum += (pred_raw - y_raw).abs().sum(dim=0)
                n += x.size(0)

        val_loss /= len(val_loader.dataset)
        mae = (mae_sum / max(1, n)).tolist()

        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"MAE(az,el)=({mae[0]:.3f}, {mae[1]:.3f}) | {dt:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "laser_net_best.pt")
            print("  saved laser_net_best.pt (best val)")

    print("Done.")


if __name__ == "__main__":
    main()
