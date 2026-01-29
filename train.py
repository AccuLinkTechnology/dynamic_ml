import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset2 import LaserDatasetRef, DEFAULT_IMG_SIZE


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

# Move seq3 into validation (as you requested)
VAL_SEQS = ["seq5", "seq3"]
TRAIN_SEQS = ["seq1", "seq2", "seq4", "seq6", "seq7", "seq8", "seq9", "seq10", "seq11", "seq12", "seq13"]

# IMPORTANT: IMG_SIZE IS (H, W)
IMG_SIZE = (180, 320)        # (H, W)
# IMG_SIZE = DEFAULT_IMG_SIZE

LABEL_NORM = "global"
LABEL_CENTER = "seq"         # <-- C2 enabled: "seq" centering
AUGMENT = False              # keep OFF for now

BATCH_SIZE = 8
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
MODE = "diff"                # "single" | "concat" | "diff"

RUN_DIR = os.path.join("runs_ref", time.strftime("%Y%m%d_%H%M%S_ref"))
os.makedirs(RUN_DIR, exist_ok=True)


def save_label_stats(train_ds: LaserDatasetRef, path: str):
    # Save seq means too (for inference / analysis)
    seq_means_payload = {k: v.detach().cpu() for k, v in getattr(train_ds, "seq_means", {}).items()}

    payload = {
        "label_norm": train_ds.label_norm,
        "label_center": train_ds.label_center,
        "global": {
            "mean": train_ds.global_stats.mean.detach().cpu(),
            "std": train_ds.global_stats.std.detach().cpu(),
        },
        "seq_means": seq_means_payload,   # { "seq1": tensor([..,..]), ... }
        "img_size": train_ds.img_size,
        "mode": MODE,
        "train_seqs": TRAIN_SEQS,
        "val_seqs": VAL_SEQS,
    }
    torch.save(payload, path)


def unnormalize(pred_norm: torch.Tensor, stats: dict) -> torch.Tensor:
    """Undo global normalization (returns CENTERED labels if label_center='seq')."""
    if stats.get("label_norm", "none") == "none":
        return pred_norm
    mean = stats["global"]["mean"]
    std = stats["global"]["std"]
    return pred_norm * std + mean


def uncenter(centered: torch.Tensor, seqs, stats: dict) -> torch.Tensor:
    """Undo seq-centering (returns RAW labels)."""
    if stats.get("label_center", "none") != "seq":
        return centered

    seq_means = stats.get("seq_means", {})
    # seqs is a list/tuple of strings with batch size length
    means = []
    for s in seqs:
        m = seq_means.get(s, None)
        if m is None:
            m = torch.zeros(2)
        means.append(m)
    mean_t = torch.stack(means, dim=0)  # [B,2]
    return centered + mean_t


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("DATA_ROOT:", DATA_ROOT)
    print("Run dir:", RUN_DIR)
    print("IMG_SIZE:", IMG_SIZE, "| AUGMENT:", AUGMENT)
    print("LABEL_CENTER:", LABEL_CENTER, "| LABEL_NORM:", LABEL_NORM)
    print("TRAIN_SEQS:", TRAIN_SEQS)
    print("VAL_SEQS:", VAL_SEQS)

    # Build train first (its stats + seq_means are the source of truth)
    train_ds = LaserDatasetRef(
        DATA_ROOT, TRAIN_SEQS,
        img_size=IMG_SIZE,
        label_norm=LABEL_NORM,
        label_center=LABEL_CENTER,
        augment=AUGMENT,
        mode=MODE,
    )

    # Build val using TRAIN stats + TRAIN seq means
    val_ds = LaserDatasetRef(
        DATA_ROOT, VAL_SEQS,
        img_size=IMG_SIZE,
        label_norm=LABEL_NORM,
        label_center=LABEL_CENTER,
        augment=False,
        mode=MODE,
        stats_override=train_ds.global_stats,
        seq_center_override=getattr(train_ds, "seq_means", None),
    )

    print("Train n =", len(train_ds))
    print("Val   n =", len(val_ds))

    # Save stats for inference + raw-unit metrics
    stats_path = os.path.join(RUN_DIR, "label_stats.pt")
    save_label_stats(train_ds, stats_path)

    # Safer torch.load usage (silences warning on newer torch)
    try:
        stats = torch.load(stats_path, weights_only=True, map_location="cpu")
    except TypeError:
        stats = torch.load(stats_path, map_location="cpu")

    print("Saved", stats_path)
    print("Train centered-mean:", stats["global"]["mean"].tolist())
    print("Train centered-std :", stats["global"]["std"].tolist())
    if stats.get("label_center") == "seq":
        print("Seq means saved for:", sorted(list(stats["seq_means"].keys())))

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

    best_val = float("inf")
    best_path = os.path.join(RUN_DIR, "laser_net_best.pt")

    for epoch in range(EPOCHS):
        t0 = time.time()

        # ---- train ----
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for x, y_norm, _seq, _y_raw in train_loader:
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

        # ---- validate ----
        model.eval()
        val_loss_sum = 0.0
        mae_sum = torch.zeros(2)
        err_sum = torch.zeros(2)  # <-- C1: bias logging
        n_val = 0

        with torch.no_grad():
            for x, y_norm, seqs, y_raw in val_loader:
                x = x.to(device, non_blocking=pin)
                y_norm = y_norm.to(device, non_blocking=pin)

                pred_norm = model(x)
                loss = criterion(pred_norm, y_norm)
                val_loss_sum += loss.item() * x.size(0)

                # pred_norm -> centered -> raw
                pred_centered = unnormalize(pred_norm.cpu(), stats)            # [B,2] centered
                pred_raw = uncenter(pred_centered, seqs, stats)               # [B,2] raw

                diff = (pred_raw - y_raw)
                mae_sum += diff.abs().sum(dim=0)
                err_sum += diff.sum(dim=0)  # signed error for bias
                n_val += x.size(0)

        val_loss = val_loss_sum / max(1, n_val)
        mae = (mae_sum / max(1, n_val)).tolist()
        bias = (err_sum / max(1, n_val)).tolist()

        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"MAE(az,el)=({mae[0]:.3f}, {mae[1]:.3f}) | "
            f"BIAS(az,el)=({bias[0]:.3f}, {bias[1]:.3f}) | {dt:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print("  saved", best_path)

    print("Done. Best model:", best_path)


if __name__ == "__main__":
    main()
