import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset2 import LaserDatasetRef
from model import LaserNet


def pick_data_root() -> str:
    candidates = [
        "/workspace/dynamic_ml/train2",
        "/home/nvidia/Documents/dynamic_ml/train2",
        "/home/nvidia/Documents/kam_ml/train2",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    if os.path.isdir("train2"):
        return os.path.abspath("train2")
    raise FileNotFoundError("Could not find train2 directory in known locations.")


# -------------------------
# CONFIG
# -------------------------
DATA_ROOT = pick_data_root()

# Multi-seq validation, and exclude seq3 from training
VAL_SEQS = ["seq5", "seq3"]
TRAIN_SEQS = ["seq1", "seq2", "seq4", "seq6", "seq7", "seq8", "seq9", "seq10", "seq11", "seq12", "seq13"]

# IMPORTANT: IMG_SIZE IS (H, W)
IMG_SIZE = (180, 320)

MODE = "diff"  # keep diff to avoid “sequence identity” cues as much as possible

# Label settings
LABEL_NORM = "global"
BASELINE_STRATEGY = "first_k"   # "first_k" recommended for your runtime semantics
BASELINE_K = 5                  # first 5 frames approximate “baseline still”

AUGMENT = False

BATCH_SIZE = 8
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

RUN_DIR = os.path.join("runs_ref", time.strftime("%Y%m%d_%H%M%S_ref"))
os.makedirs(RUN_DIR, exist_ok=True)


def save_label_stats(train_ds: LaserDatasetRef, path: str):
    # Save stats + per-seq baselines used to construct command labels (for audit/debug)
    baselines = {k: v.detach().cpu() for k, v in train_ds.baseline_by_seq.items()}
    payload = {
        "label_norm": train_ds.label_norm,
        "img_size": train_ds.img_size,
        "mode": MODE,
        "train_seqs": TRAIN_SEQS,
        "val_seqs": VAL_SEQS,
        "baseline_strategy": BASELINE_STRATEGY,
        "baseline_k": BASELINE_K,
        "global": {
            "mean": train_ds.global_stats.mean.detach().cpu(),
            "std": train_ds.global_stats.std.detach().cpu(),
        },
        "baseline_by_seq": baselines,
    }
    torch.save(payload, path)


def unnormalize(pred_norm: torch.Tensor, stats: dict) -> torch.Tensor:
    """Undo global normalization, returning predictions in COMMAND space."""
    if stats.get("label_norm", "none") == "none":
        return pred_norm
    mean = stats["global"]["mean"]
    std = stats["global"]["std"]
    return pred_norm * std + mean


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("DATA_ROOT:", DATA_ROOT)
    print("Run dir:", RUN_DIR)
    print("IMG_SIZE:", IMG_SIZE, "| AUGMENT:", AUGMENT)
    print("MODE:", MODE)
    print("LABEL_NORM:", LABEL_NORM)
    print("BASELINE_STRATEGY:", BASELINE_STRATEGY, "| BASELINE_K:", BASELINE_K)
    print("TRAIN_SEQS:", TRAIN_SEQS)
    print("VAL_SEQS:", VAL_SEQS)

    # Train dataset computes baseline + stats in COMMAND space
    train_ds = LaserDatasetRef(
        DATA_ROOT, TRAIN_SEQS,
        img_size=IMG_SIZE,
        label_norm=LABEL_NORM,
        augment=AUGMENT,
        mode=MODE,
        baseline_strategy=BASELINE_STRATEGY,
        baseline_k=BASELINE_K,
    )

    # Val uses its own baseline to define command labels, but MUST use TRAIN stats for normalization
    val_ds = LaserDatasetRef(
        DATA_ROOT, VAL_SEQS,
        img_size=IMG_SIZE,
        label_norm=LABEL_NORM,
        augment=False,
        mode=MODE,
        baseline_strategy=BASELINE_STRATEGY,
        baseline_k=BASELINE_K,
        stats_override=train_ds.global_stats,
    )

    print("Train n =", len(train_ds))
    print("Val   n =", len(val_ds))

    stats_path = os.path.join(RUN_DIR, "label_stats.pt")
    save_label_stats(train_ds, stats_path)

    # Safe-ish load (handles older torch too)
    try:
        stats = torch.load(stats_path, weights_only=True, map_location="cpu")
    except TypeError:
        stats = torch.load(stats_path, map_location="cpu")

    print("Saved", stats_path)
    print("Train CMD mean:", stats["global"]["mean"].tolist())
    print("Train CMD std :", stats["global"]["std"].tolist())

    pin = (device == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
    )

    in_ch = 3 if MODE in ["single", "diff"] else 6
    model = LaserNet(in_channels=in_ch).to(device)

    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Optional but usually beneficial for the jitter you’ve seen:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    best_val = float("inf")
    best_path = os.path.join(RUN_DIR, "laser_net_best.pt")

    for epoch in range(EPOCHS):
        t0 = time.time()

        # ---- train ----
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for x, y_norm, _seq, _y_cmd_raw, _y_raw in train_loader:
            x = x.to(device, non_blocking=pin)
            y_norm = y_norm.to(device, non_blocking=pin)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y_norm)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            n_train += x.size(0)

        train_loss = train_loss_sum / max(1, n_train)

        # ---- validate (COMMAND SPACE) ----
        model.eval()
        val_loss_sum = 0.0
        mae_sum = torch.zeros(2)
        err_sum = torch.zeros(2)
        n_val = 0

        with torch.no_grad():
            for x, y_norm, seqs, y_cmd_raw, _y_raw in val_loader:
                x = x.to(device, non_blocking=pin)
                y_norm = y_norm.to(device, non_blocking=pin)

                pred_norm = model(x)
                loss = criterion(pred_norm, y_norm)
                val_loss_sum += loss.item() * x.size(0)

                pred_cmd = unnormalize(pred_norm.cpu(), stats)  # [B,2] in COMMAND space
                diff = (pred_cmd - y_cmd_raw)

                mae_sum += diff.abs().sum(dim=0)
                err_sum += diff.sum(dim=0)
                n_val += x.size(0)

        val_loss = val_loss_sum / max(1, n_val)
        mae = (mae_sum / max(1, n_val)).tolist()
        bias = (err_sum / max(1, n_val)).tolist()

        scheduler.step(val_loss)

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | lr={lr_now:.2e} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"MAE_CMD(az,el)=({mae[0]:.3f}, {mae[1]:.3f}) | "
            f"BIAS_CMD(az,el)=({bias[0]:.3f}, {bias[1]:.3f}) | {dt:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print("  saved", best_path)

    print("Done. Best model:", best_path)


if __name__ == "__main__":
    main()
