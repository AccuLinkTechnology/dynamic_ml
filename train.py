import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset2 import LaserDatasetRef, DEFAULT_IMG_SIZE
from model import LaserNet


def pick_data_root() -> str:
    # Prefer container-mounted path if present
    candidates = [
        "/workspace/dynamic_ml/train2",
        "/home/nvidia/Documents/dynamic_ml/train2",
        "/home/nvidia/Documents/kam_ml/train2",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    # fallback: current working directory relative
    if os.path.isdir("train2"):
        return os.path.abspath("train2")
    raise FileNotFoundError("Could not find train2 directory in known locations.")


# -------------------------
# CONFIG
# -------------------------
DATA_ROOT = pick_data_root()

TRAIN_SEQS = ["seq1", "seq2", "seq3", "seq4"]
VAL_SEQS = ["seq5"]

IMG_SIZE = (640, 360)        # your current preference
# IMG_SIZE = DEFAULT_IMG_SIZE  # or go back to 320x180

LABEL_NORM = "global"
AUGMENT = False              # keep OFF for now

BATCH_SIZE = 8
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

RUN_DIR = os.path.join("runs_ref", time.strftime("%Y%m%d_%H%M%S_ref"))
os.makedirs(RUN_DIR, exist_ok=True)


def save_label_stats(ds: LaserDatasetRef, path: str):
    payload = {
        "label_norm": ds.label_norm,
        "global": {
            "mean": ds.global_stats.mean,
            "std": ds.global_stats.std,
        },
        "img_size": ds.img_size,
        "mode": "ref_concat_6ch",
        "train_seqs": TRAIN_SEQS,
        "val_seqs": VAL_SEQS,
    }
    torch.save(payload, path)


def unnormalize(pred_norm: torch.Tensor, stats: dict) -> torch.Tensor:
    if stats["label_norm"] == "none":
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

    train_ds = LaserDatasetRef(
        DATA_ROOT, TRAIN_SEQS, img_size=IMG_SIZE, label_norm=LABEL_NORM, augment=AUGMENT
    )
    val_ds = LaserDatasetRef(
        DATA_ROOT, VAL_SEQS, img_size=IMG_SIZE, label_norm=LABEL_NORM, augment=False
    )

    print("Train seqs:", TRAIN_SEQS, "n=", len(train_ds))
    print("Val seqs:", VAL_SEQS, "n=", len(val_ds))

    # Save stats for inference + real-unit MAE
    stats_path = os.path.join(RUN_DIR, "label_stats.pt")
    save_label_stats(train_ds, stats_path)
    stats = torch.load(stats_path)
    print("Saved", stats_path)

    pin = (device == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin
    )

    model = LaserNet(in_ch=6).to(device)
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
        n_val = 0

        with torch.no_grad():
            for x, y_norm, _seq, y_raw in val_loader:
                x = x.to(device, non_blocking=pin)
                y_norm = y_norm.to(device, non_blocking=pin)

                pred_norm = model(x)
                loss = criterion(pred_norm, y_norm)
                val_loss_sum += loss.item() * x.size(0)

                pred_raw = unnormalize(pred_norm.cpu(), stats)
                mae_sum += (pred_raw - y_raw).abs().sum(dim=0)
                n_val += x.size(0)

        val_loss = val_loss_sum / max(1, n_val)
        mae = (mae_sum / max(1, n_val)).tolist()

        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"MAE(az,el)=({mae[0]:.3f}, {mae[1]:.3f}) | {dt:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print("  saved", best_path)

    print("Done. Best model:", best_path)


if __name__ == "__main__":
    main()
