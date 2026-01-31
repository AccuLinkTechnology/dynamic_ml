import os
import time
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler

from dataset2 import LaserDatasetRef, DEFAULT_IMG_SIZE_HW
from model import LaserNet
from cross_roi import debug_overlay
from PIL import Image


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

VAL_SEQS = ["seq5", "seq20"]  # adjust as you like
ALL_SEQS = ["seq1", "seq2", "seq3", "seq4", "seq5",
            "seq6", "seq7", "seq8", "seq9", "seq10", "seq11", "seq12", "seq13", "seq14",
            "seq15", "seq16", "seq17", "seq18", "seq19", "seq20"]
TRAIN_SEQS = [s for s in ALL_SEQS if s not in set(VAL_SEQS)]

IMG_SIZE_HW = DEFAULT_IMG_SIZE_HW     # (H,W) = (180,320)
AUGMENT = False

MODE = "concat"                       # "single" | "concat" | "diff"
LABEL_NORM = "global"

BASELINE_STRATEGY = "stable_m"
BASELINE_M = 10

BATCH_SIZE = 8
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4

# DataLoader performance
NUM_WORKERS = 4
PIN_MEMORY = True

# Zero-loss (stable should command (0,0))
ZERO_LOSS_W = 0.2
ZERO_STABLE_IDX = 0

# Make training sequence-balanced
SEQ_BALANCED = True
SEED = 123

# ROI
USE_ROI = True
ROI_BG_WEIGHT = 0.20
ROI_STRENGTH = 1.00
ROI_FROM = "ref"   # IMPORTANT: "ref" gives stable masking and avoids per-frame failures

RUN_DIR = os.path.join("runs_ref", time.strftime("%Y%m%d_%H%M%S_ref"))
os.makedirs(RUN_DIR, exist_ok=True)

DEBUG_DIR = os.path.join(RUN_DIR, "debug_roi")
os.makedirs(DEBUG_DIR, exist_ok=True)


def save_label_stats(ds: LaserDatasetRef, path: str):
    payload = {
        "label_norm": ds.label_norm,
        "global": {"mean": ds.global_stats.mean, "std": ds.global_stats.std},
        "img_size_hw": ds.img_size_hw,
        "mode": MODE,
        "baseline_strategy": ds.baseline_strategy,
        "baseline_m": ds.baseline_m,
        "baseline_by_seq": ds.baseline_by_seq,
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


class SeqBalancedBatchSampler(Sampler):
    """
    Yields batches by sampling sequences uniformly, then sampling a random index within each sequence.
    Prevents large sequences from dominating training.
    """
    def __init__(self, ds: LaserDatasetRef, batch_size: int, seed: int = 0):
        self.ds = ds
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        seq_to_indices = defaultdict(list)
        for i, (_path, _y_raw, seq, _pic) in enumerate(ds.samples):
            seq_to_indices[seq].append(i)

        self.seq_to_indices = dict(seq_to_indices)
        self.seqs = sorted(self.seq_to_indices.keys())
        if len(self.seqs) == 0:
            raise RuntimeError("No sequences found for sampler.")

        self.num_batches = max(1, len(ds) // self.batch_size)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + 1000 * self.epoch)
        for _ in range(self.num_batches):
            batch = []
            for _j in range(self.batch_size):
                seq = rng.choice(self.seqs)
                idx = rng.choice(self.seq_to_indices[seq])
                batch.append(idx)
            yield batch


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("DATA_ROOT:", DATA_ROOT)
    print("Run dir:", RUN_DIR)
    print("IMG_SIZE:", IMG_SIZE_HW, "| AUGMENT:", AUGMENT)
    print("MODE:", MODE)
    print("LABEL_NORM:", LABEL_NORM)
    print("BASELINE_STRATEGY:", BASELINE_STRATEGY, "| BASELINE_M:", BASELINE_M)
    print("ZERO_LOSS_W:", ZERO_LOSS_W, "| ZERO_STABLE_IDX:", ZERO_STABLE_IDX)
    print("USE_ROI:", USE_ROI, "| ROI_FROM:", ROI_FROM, "| ROI_BG_WEIGHT:", ROI_BG_WEIGHT, "| ROI_STRENGTH:", ROI_STRENGTH)
    print("TRAIN_SEQS:", TRAIN_SEQS)
    print("VAL_SEQS:", VAL_SEQS)

    train_ds = LaserDatasetRef(
        DATA_ROOT,
        TRAIN_SEQS,
        img_size_hw=IMG_SIZE_HW,
        label_norm=LABEL_NORM,
        augment=AUGMENT,
        mode=MODE,
        baseline_strategy=BASELINE_STRATEGY,
        baseline_m=BASELINE_M,
        stable_pool=3,
        use_cross_roi=USE_ROI,
        roi_bg_weight=ROI_BG_WEIGHT,
        roi_strength=ROI_STRENGTH,
        roi_from=ROI_FROM,
    )

    # IMPORTANT: val_ds uses train_ds stats for normalization correctness
    val_ds = LaserDatasetRef(
        DATA_ROOT,
        VAL_SEQS,
        img_size_hw=IMG_SIZE_HW,
        label_norm=LABEL_NORM,
        augment=False,
        mode=MODE,
        baseline_strategy=BASELINE_STRATEGY,
        baseline_m=BASELINE_M,
        stable_pool=3,
        use_cross_roi=USE_ROI,
        roi_bg_weight=ROI_BG_WEIGHT,
        roi_strength=ROI_STRENGTH,
        roi_from=ROI_FROM,
        external_stats=train_ds.global_stats,
    )

    print("Train n =", len(train_ds))
    print("Val   n =", len(val_ds))

    stats_path = os.path.join(RUN_DIR, "label_stats.pt")
    save_label_stats(train_ds, stats_path)

    # Avoid pickle warning: keep tensors only, no need to load arbitrary objects
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    print("Saved", stats_path)
    print("Train CMD mean:", stats["global"]["mean"].tolist())
    print("Train CMD std :", stats["global"]["std"].tolist())

    pin = PIN_MEMORY and (device == "cuda")

    if SEQ_BALANCED:
        batch_sampler = SeqBalancedBatchSampler(train_ds, batch_size=BATCH_SIZE, seed=SEED)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=pin,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=pin,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )

    in_ch = 3 if MODE in ["single", "diff"] else 6
    model = LaserNet(in_channels=in_ch).to(device)

    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    best_score = float("inf")
    best_path = os.path.join(RUN_DIR, "laser_net_best.pt")

    # For stable->zero loss in normalized space:
    mean = train_ds.global_stats.mean.to(device)
    std = train_ds.global_stats.std.to(device)
    zero_target_base = (-mean / std).unsqueeze(0)

    for epoch in range(EPOCHS):
        t0 = time.time()

        if SEQ_BALANCED:
            train_loader.batch_sampler.set_epoch(epoch)

        # ---- train ----
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for x, y_norm, seqs, y_cmd, _y_raw, img_paths in train_loader:
            x = x.to(device, non_blocking=pin)
            y_norm = y_norm.to(device, non_blocking=pin)

            # cached stable batch (no disk I/O)
            x_stable = torch.stack(
                [train_ds.get_stable_input(s, stable_idx=ZERO_STABLE_IDX) for s in seqs],
                dim=0,
            ).to(device, non_blocking=pin)

            # debug overlay: first batch only
            if epoch % 5 == 0 and n_train == 0:
                for k in range(min(4, len(img_paths))):
                    try:
                        pil = Image.open(img_paths[k]).convert("RGB")
                        ov = debug_overlay(pil, out_hw=IMG_SIZE_HW, bg_weight=ROI_BG_WEIGHT)
                        out_path = os.path.join(DEBUG_DIR, f"e{epoch:03d}_k{k}_{seqs[k]}.png")
                        ov.save(out_path)
                    except Exception as e:
                        print(f"[WARN] debug overlay failed for {img_paths[k]}: {e}")

            optimizer.zero_grad(set_to_none=True)

            pred = model(x)
            loss_main = criterion(pred, y_norm)

            pred_stable = model(x_stable)
            zero_target = zero_target_base.expand_as(pred_stable)
            loss_zero = criterion(pred_stable, zero_target)

            loss = loss_main + (ZERO_LOSS_W * loss_zero)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            n_train += x.size(0)

        train_loss = train_loss_sum / max(1, n_train)

        # ---- validate ----
        model.eval()
        val_loss_sum = 0.0
        mae_sum = torch.zeros(2)
        bias_sum = torch.zeros(2)
        n_val = 0

        per_seq = {}
        with torch.no_grad():
            for x, y_norm, seqs, y_cmd, _y_raw, _img_paths in val_loader:
                x = x.to(device, non_blocking=pin)
                y_norm = y_norm.to(device, non_blocking=pin)

                pred_norm = model(x)
                loss = criterion(pred_norm, y_norm)
                val_loss_sum += loss.item() * x.size(0)

                pred_cmd = unnormalize(pred_norm.cpu(), stats)
                err = (pred_cmd - y_cmd).cpu()

                mae_sum += err.abs().sum(dim=0)
                bias_sum += err.sum(dim=0)
                n_val += x.size(0)

                for i, s in enumerate(seqs):
                    if s not in per_seq:
                        per_seq[s] = {"mae": torch.zeros(2), "bias": torch.zeros(2), "n": 0}
                    per_seq[s]["mae"] += err[i].abs()
                    per_seq[s]["bias"] += err[i]
                    per_seq[s]["n"] += 1

        val_loss = val_loss_sum / max(1, n_val)
        mae = (mae_sum / max(1, n_val)).tolist()
        bias = (bias_sum / max(1, n_val)).tolist()

        # Bias-aware score (safer for closed-loop)
        score = (abs(bias[0]) + abs(bias[1])) + 0.5 * (mae[0] + mae[1])

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | lr={lr_now:.2e} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"MAE_CMD(az,el)=({mae[0]:.3f}, {mae[1]:.3f}) | "
            f"BIAS_CMD(az,el)=({bias[0]:.3f}, {bias[1]:.3f}) | score={score:.4f} | {dt:.1f}s"
        )

        if len(per_seq) > 0:
            parts = []
            for s in sorted(per_seq.keys()):
                n = per_seq[s]["n"]
                mae_s = (per_seq[s]["mae"] / max(1, n)).tolist()
                bias_s = (per_seq[s]["bias"] / max(1, n)).tolist()
                parts.append(f"{s}: MAE=({mae_s[0]:.3f},{mae_s[1]:.3f}) BIAS=({bias_s[0]:.3f},{bias_s[1]:.3f}) n={n}")
            print("  Val per-seq | " + " | ".join(parts))

        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)
            print("  saved", best_path)

        scheduler.step()

    print("Done. Best model:", best_path)


if __name__ == "__main__":
    main()
