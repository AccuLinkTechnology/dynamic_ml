# train_v2_fast.py
# PERFORMANCE OPTIMIZED VERSION
# - Uses LaserNetSimple (180K params, same as original)
# - Simplified validation (no per-sample denorm)
# - Maintains all improvements (no pose head, better losses, curriculum)

import os
import time
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler

from dataset3 import LaserDatasetV3, DEFAULT_IMG_SIZE_HW
from model_v2 import LaserNetSimple


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
# CONFIG - OPTIMIZED FOR SPEED
# -------------------------
DATA_ROOT = pick_data_root()

VAL_SEQS = ["seq5", "seq23", "seq20"]
ALL_SEQS = sorted([d for d in os.listdir(DATA_ROOT) 
                   if d.startswith("seq") and os.path.isdir(os.path.join(DATA_ROOT, d))])
TRAIN_SEQS = [s for s in ALL_SEQS if s not in set(VAL_SEQS)]

IMG_SIZE_HW = DEFAULT_IMG_SIZE_HW
AUGMENT = True

BASELINE_STRATEGY = "stable_median"
BASELINE_M = 10

BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3  # Slightly higher for faster convergence
WEIGHT_DECAY = 1e-4  # Less regularization for speed

WARMUP_EPOCHS = 3  # Shorter warmup
WARMUP_LR = 2e-4

NUM_WORKERS = 4
PIN_MEMORY = True

# Simplified loss (Huber is slower than SmoothL1)
LOSS_TYPE = "smooth_l1"
HUBER_DELTA = 0.5

DEADZONE_INITIAL = (0.15, 0.15)
DEADZONE_FINAL = (0.05, 0.05)
DEADZONE_EPOCHS = 40

ZERO_LOSS_W = 0.3
ZERO_STABLE_IDX = 0
BIAS_W = 0.2  # REDUCED from 0.4 - was overcorrecting

OUT_SCALE = (2.6, 2.6)

SEQ_BALANCED = True
SEED = 123

NORMALIZE_LABELS = False  # DISABLED - causes bias drift

RUN_DIR = os.path.join("runs_v2", time.strftime("%Y%m%d_%H%M%S_fast"))
os.makedirs(RUN_DIR, exist_ok=True)


def save_config(path: str, ds_train, ds_val):
    config = {
        "data_root": DATA_ROOT,
        "train_seqs": TRAIN_SEQS,
        "val_seqs": VAL_SEQS,
        "img_size_hw": IMG_SIZE_HW,
        "model_type": "simple",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "out_scale": OUT_SCALE,
        "normalize_labels": NORMALIZE_LABELS,
        "train_baseline_by_seq": ds_train.baseline_by_seq,
        "val_baseline_by_seq": ds_val.baseline_by_seq,
        "train_seq_stats": ds_train.seq_stats if NORMALIZE_LABELS else None,
        "val_seq_stats": ds_val.seq_stats if NORMALIZE_LABELS else None,
    }
    torch.save(config, path)


class SeqBalancedBatchSampler(Sampler):
    def __init__(self, ds: LaserDatasetV3, batch_size: int, seed: int = 0):
        self.ds = ds
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        seq_to_indices = defaultdict(list)
        for i, (_path, _y_raw, seq, _pic) in enumerate(ds.samples):
            seq_to_indices[seq].append(i)

        self.seq_to_indices = dict(seq_to_indices)
        self.seqs = sorted(self.seq_to_indices.keys())
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


def get_adaptive_deadzone(epoch: int) -> tuple:
    if epoch >= DEADZONE_EPOCHS:
        return DEADZONE_FINAL
    
    alpha = epoch / DEADZONE_EPOCHS
    dz_az = DEADZONE_INITIAL[0] * (1 - alpha) + DEADZONE_FINAL[0] * alpha
    dz_el = DEADZONE_INITIAL[1] * (1 - alpha) + DEADZONE_FINAL[1] * alpha
    return (dz_az, dz_el)


def deadzone_loss(pred: torch.Tensor, target: torch.Tensor, dz=(0.1, 0.1)) -> torch.Tensor:
    """Fast deadzone + SmoothL1 loss."""
    err = pred - target
    dz_t = torch.tensor(dz, device=err.device, dtype=err.dtype).view(1, 2)
    err_shrunk = torch.sign(err) * torch.clamp(err.abs() - dz_t, min=0.0)
    return nn.functional.smooth_l1_loss(err_shrunk, torch.zeros_like(err_shrunk), beta=1.0)


@torch.no_grad()
def compute_calibration_offset(model, ds: LaserDatasetV3, device: str) -> dict:
    model.eval()
    offsets = {}
    for s in ds.seqs:
        x_stable = ds.get_stable_input(s, stable_idx=ZERO_STABLE_IDX).unsqueeze(0).to(device)
        cmd = model(x_stable)
        offsets[s] = cmd.squeeze(0).detach().cpu()
    return offsets


@torch.no_grad()
def validate(model, val_loader, device, ds_val, deadzone, cal_offsets):
    """Optimized validation - minimal denormalization."""
    model.eval()
    
    val_loss = 0.0
    mae_sum = torch.zeros(2)
    bias_sum = torch.zeros(2)
    n = 0
    
    per_seq = {}
    
    for x, y_cmd, seqs, _ in val_loader:
        x = x.to(device, non_blocking=True)
        y_cmd_gpu = y_cmd.to(device, non_blocking=True)
        
        pred_cmd = model(x)
        
        # Apply calibration
        offs = torch.stack([cal_offsets[s] for s in seqs], dim=0).to(device)
        pred_cal = pred_cmd - offs
        
        # Loss in normalized space
        loss = deadzone_loss(pred_cal, y_cmd_gpu, dz=deadzone)
        val_loss += loss.item() * x.size(0)
        
        # Metrics: denormalize on CPU for speed
        pred_cpu = pred_cal.cpu()
        y_cpu = y_cmd  # Already on CPU
        
        if ds_val.normalize_labels:
            for i, s in enumerate(seqs):
                stats = ds_val.seq_stats[s]
                pred_cpu[i] = pred_cpu[i] * stats.std + stats.mean
                y_cpu[i] = y_cpu[i] * stats.std + stats.mean
        
        err = pred_cpu - y_cpu
        mae_sum += err.abs().sum(dim=0)
        bias_sum += err.sum(dim=0)
        n += x.size(0)
        
        # Per-seq
        for i, s in enumerate(seqs):
            if s not in per_seq:
                per_seq[s] = {"mae": torch.zeros(2), "bias": torch.zeros(2), "n": 0}
            per_seq[s]["mae"] += err[i].abs()
            per_seq[s]["bias"] += err[i]
            per_seq[s]["n"] += 1
    
    val_loss /= max(1, n)
    mae = (mae_sum / max(1, n)).tolist()
    bias = (bias_sum / max(1, n)).tolist()
    score = 1.5 * (abs(bias[0]) + abs(bias[1])) + 0.5 * (mae[0] + mae[1])
    
    return val_loss, mae, bias, score, per_seq


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("LASER CONTROL - FAST TRAINING (Simple Model)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Workers: {NUM_WORKERS} | Model: LaserNetSimple (~180K params)")
    print("=" * 80)

    train_ds = LaserDatasetV3(
        DATA_ROOT, TRAIN_SEQS,
        img_size_hw=IMG_SIZE_HW,
        augment=AUGMENT,
        baseline_strategy=BASELINE_STRATEGY,
        baseline_m=BASELINE_M,
        stable_pool=5,
        normalize_labels=NORMALIZE_LABELS,
    )
    
    val_ds = LaserDatasetV3(
        DATA_ROOT, VAL_SEQS,
        img_size_hw=IMG_SIZE_HW,
        augment=False,
        baseline_strategy=BASELINE_STRATEGY,
        baseline_m=BASELINE_M,
        stable_pool=5,
        normalize_labels=NORMALIZE_LABELS,
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    save_config(os.path.join(RUN_DIR, "config.pt"), train_ds, val_ds)

    loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    if NUM_WORKERS > 0:
        loader_kwargs.update(dict(prefetch_factor=2, persistent_workers=True))

    if SEQ_BALANCED:
        batch_sampler = SeqBalancedBatchSampler(train_ds, batch_size=BATCH_SIZE, seed=SEED)
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    model = LaserNetSimple(in_channels=6, out_scale=OUT_SCALE).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-5)

    best_score = float("inf")
    best_path = os.path.join(RUN_DIR, "best_model.pt")
    
    print("Starting training...")
    print("=" * 80)

    for epoch in range(EPOCHS):
        t0 = time.time()
        
        deadzone = get_adaptive_deadzone(epoch)
        
        # Warmup
        if epoch < WARMUP_EPOCHS:
            lr_scale = (epoch + 1) / WARMUP_EPOCHS
            lr_current = WARMUP_LR + (LR - WARMUP_LR) * lr_scale
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_current
        
        if SEQ_BALANCED:
            train_loader.batch_sampler.set_epoch(epoch)

        # Train
        model.train()
        train_loss = 0.0
        n_train = 0

        for x, y_cmd, seqs, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y_cmd = y_cmd.to(device, non_blocking=True)

            x_stable = torch.stack([
                train_ds.get_stable_input(s, stable_idx=ZERO_STABLE_IDX) 
                for s in seqs
            ], dim=0).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred_cmd = model(x)
            loss_cmd = deadzone_loss(pred_cmd, y_cmd, dz=deadzone)

            pred_stable = model(x_stable)
            loss_zero = deadzone_loss(pred_stable, torch.zeros_like(pred_stable), dz=deadzone)

            bias = (pred_cmd - y_cmd).mean(dim=0).abs().sum()

            loss = loss_cmd + ZERO_LOSS_W * loss_zero + BIAS_W * bias
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            n_train += x.size(0)

        train_loss /= max(1, n_train)

        # Validate
        cal_offsets = compute_calibration_offset(model, val_ds, device)
        val_loss, mae, bias, score, per_seq = validate(
            model, val_loader, device, val_ds, deadzone, cal_offsets
        )

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        
        print(
            f"E{epoch:03d} | lr={lr_now:.1e} | t={train_loss:.4f} v={val_loss:.4f} | "
            f"MAE=({mae[0]:.3f},{mae[1]:.3f}) BIAS=({bias[0]:.3f},{bias[1]:.3f}) | "
            f"score={score:.3f} | {dt:.1f}s"
        )

        if score < best_score:
            best_score = score
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 
                       'score': score}, best_path)
            print(f"  ✓ Best: {score:.3f}")

        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

    print(f"Done! Best: {best_path}")


if __name__ == "__main__":
    main()