import os
import time
from collections import defaultdict

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

# Keep seq-level split (this is correct). We will now print per-seq validation metrics too.
VAL_SEQS = ["seq5", "seq3"]
TRAIN_SEQS = ["seq1", "seq2", "seq4", "seq6", "seq7", "seq8", "seq9", "seq10", "seq11", "seq12", "seq13"]

IMG_SIZE = (180, 320)  # (H, W)

# (A/B) Default to concat now (helps domain shift / geometry differences)
MODE = "concat"        # "diff" or "concat" (try concat first)

LABEL_NORM = "global"

# Baseline = stable frames closest to ref
BASELINE_STRATEGY = "stable_m"
BASELINE_M = 10
BASELINE_K = 5

# Realistic zero-at-stable auxiliary loss (NO all-zero batches)
ZERO_LOSS_W = 0.20     # keep 0.2 for now; can tune 0.1–0.3
ZERO_STABLE_IDX = 0    # use the most stable frame (0). You can try 1 or 2.

AUGMENT = False

BATCH_SIZE = 8
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# Bias-aware checkpointing (robot-safe)
BIAS_LAMBDA = 0.25     # score = val_loss + lambda*(|bias_az|+|bias_el|)

RUN_DIR = os.path.join("runs_ref", time.strftime("%Y%m%d_%H%M%S_ref"))
os.makedirs(RUN_DIR, exist_ok=True)


def save_label_stats(train_ds: LaserDatasetRef, path: str):
    baselines = {k: v.detach().cpu() for k, v in train_ds.baseline_by_seq.items()}
    payload = {
        "label_norm": train_ds.label_norm,
        "img_size": train_ds.img_size,
        "mode": MODE,
        "train_seqs": TRAIN_SEQS,
        "val_seqs": VAL_SEQS,
        "baseline_strategy": BASELINE_STRATEGY,
        "baseline_m": BASELINE_M,
        "baseline_k": BASELINE_K,
        "global": {
            "mean": train_ds.global_stats.mean.detach().cpu(),
            "std": train_ds.global_stats.std.detach().cpu(),
        },
        "baseline_by_seq": baselines,
    }
    torch.save(payload, path)


def unnormalize(pred_norm: torch.Tensor, stats: dict) -> torch.Tensor:
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
    print("BASELINE_STRATEGY:", BASELINE_STRATEGY, "| BASELINE_M:", BASELINE_M)
    print("ZERO_LOSS_W:", ZERO_LOSS_W, "| ZERO_STABLE_IDX:", ZERO_STABLE_IDX)
    print("TRAIN_SEQS:", TRAIN_SEQS)
    print("VAL_SEQS:", VAL_SEQS)

    train_ds = LaserDatasetRef(
        DATA_ROOT, TRAIN_SEQS,
        img_size=IMG_SIZE,
        label_norm=LABEL_NORM,
        augment=AUGMENT,
        mode=MODE,
        baseline_strategy=BASELINE_STRATEGY,
        baseline_m=BASELINE_M,
        baseline_k=BASELINE_K,
    )

    val_ds = LaserDatasetRef(
        DATA_ROOT, VAL_SEQS,
        img_size=IMG_SIZE,
        label_norm=LABEL_NORM,
        augment=False,
        mode=MODE,
        baseline_strategy=BASELINE_STRATEGY,
        baseline_m=BASELINE_M,
        baseline_k=BASELINE_K,
        stats_override=train_ds.global_stats,  # IMPORTANT
    )

    print("Train n =", len(train_ds))
    print("Val   n =", len(val_ds))

    stats_path = os.path.join(RUN_DIR, "label_stats.pt")
    save_label_stats(train_ds, stats_path)
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    best_score = float("inf")
    best_path = os.path.join(RUN_DIR, "laser_net_best.pt")

    for epoch in range(EPOCHS):
        t0 = time.time()

        # ---- train ----
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for x, y_norm, seqs, _y_cmd_raw, _y_raw in train_loader:
            x = x.to(device, non_blocking=pin)
            y_norm = y_norm.to(device, non_blocking=pin)

            optimizer.zero_grad(set_to_none=True)

            pred = model(x)
            loss_main = criterion(pred, y_norm)

            # (2) Realistic "zero" loss using stable frames per seq (in-distribution)
            seqs_list = list(seqs)
            x_stable = torch.stack(
                [train_ds.get_stable_input(s, stable_idx=ZERO_STABLE_IDX) for s in seqs_list],
                dim=0
            ).to(device, non_blocking=pin)
            y0 = torch.zeros((x_stable.size(0), 2), device=device, dtype=pred.dtype)

            pred_stable = model(x_stable)
            loss_zero = criterion(pred_stable, y0)

            loss = loss_main + ZERO_LOSS_W * loss_zero
            loss.backward()
            optimizer.step()

            train_loss_sum += loss_main.item() * x.size(0)
            n_train += x.size(0)

        train_loss = train_loss_sum / max(1, n_train)

        # ---- validate (COMMAND space) ----
        model.eval()
        val_loss_sum = 0.0

        mae_sum = torch.zeros(2)
        err_sum = torch.zeros(2)
        n_val = 0

        # per-seq accumulators
        per_seq_mae = defaultdict(lambda: torch.zeros(2))
        per_seq_err = defaultdict(lambda: torch.zeros(2))
        per_seq_n = defaultdict(int)

        with torch.no_grad():
            for x, y_norm, seqs, y_cmd_raw, _y_raw in val_loader:
                x = x.to(device, non_blocking=pin)
                y_norm = y_norm.to(device, non_blocking=pin)

                pred_norm = model(x)
                loss = criterion(pred_norm, y_norm)
                val_loss_sum += loss.item() * x.size(0)

                pred_cmd = unnormalize(pred_norm.cpu(), stats)  # COMMAND space
                diff = (pred_cmd - y_cmd_raw)                  # COMMAND error

                mae_sum += diff.abs().sum(dim=0)
                err_sum += diff.sum(dim=0)
                n_val += x.size(0)

                # per-seq
                for i, s in enumerate(seqs):
                    s = str(s)
                    per_seq_mae[s] += diff[i].abs()
                    per_seq_err[s] += diff[i]
                    per_seq_n[s] += 1

        val_loss = val_loss_sum / max(1, n_val)
        mae = (mae_sum / max(1, n_val)).tolist()
        bias = (err_sum / max(1, n_val)).tolist()

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        score = val_loss + BIAS_LAMBDA * (abs(bias[0]) + abs(bias[1]))

        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d} | lr={lr_now:.2e} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"MAE_CMD(az,el)=({mae[0]:.3f}, {mae[1]:.3f}) | "
            f"BIAS_CMD(az,el)=({bias[0]:.3f}, {bias[1]:.3f}) | score={score:.4f} | {dt:.1f}s"
        )

        # print per-seq val metrics every epoch (small val set)
        parts = []
        for s in sorted(per_seq_n.keys()):
            n = per_seq_n[s]
            if n == 0:
                continue
            mae_s = (per_seq_mae[s] / n).tolist()
            bias_s = (per_seq_err[s] / n).tolist()
            parts.append(f"{s}: MAE=({mae_s[0]:.3f},{mae_s[1]:.3f}) BIAS=({bias_s[0]:.3f},{bias_s[1]:.3f}) n={n}")
        print("  Val per-seq | " + " | ".join(parts))

        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)
            print("  saved", best_path)

    print("Done. Best model:", best_path)


if __name__ == "__main__":
    main()
