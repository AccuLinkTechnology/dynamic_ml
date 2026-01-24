import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset

from dataset2 import LaserDataset
from model import LaserNet

DATA_ROOT = "/home/nvidia/Documents/kam_ml"
SEQS = ["seq2", "seq3", "seq4"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Build dataset per sequence (so we can split within each sequence)
    per_seq_ds = [LaserDataset(DATA_ROOT, [seq]) for seq in SEQS]

    # Save stats so inference can un-normalize (each dataset has exactly one seq)
    seq_stats = {}
    for ds, seq in zip(per_seq_ds, SEQS):
        seq_stats[seq] = ds.seq_stats[seq]

    torch.save(seq_stats, "seq_stats.pt")
    print("Saved seq_stats.pt")

    # Split EACH sequence into train/val and then combine
    train_parts, val_parts = [], []
    g = torch.Generator().manual_seed(42)

    for ds in per_seq_ds:
        n_total = len(ds)
        n_val = max(1, int(0.2 * n_total))
        n_train = n_total - n_val
        tr, va = random_split(ds, [n_train, n_val], generator=g)
        train_parts.append(tr)
        val_parts.append(va)

    train_ds = ConcatDataset(train_parts)
    val_ds = ConcatDataset(val_parts)

    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=pin)

    model = LaserNet().to(device)

    # Huber loss is more stable than MSE when you have occasional outliers
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val = float("inf")

    for epoch in range(80):
        t0 = time.time()

        # -----------------
        # Train
        # -----------------
        model.train()
        train_loss = 0.0

        for x, y_norm, _seq in train_loader:
            x = x.to(device, non_blocking=pin)
            y_norm = y_norm.to(device, non_blocking=pin)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y_norm)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -----------------
        # Validate
        # -----------------
        model.eval()
        val_loss = 0.0
        mae_az = 0.0
        mae_el = 0.0
        n = 0

        with torch.no_grad():
            for x, y_norm, seq_list in val_loader:
                x = x.to(device, non_blocking=pin)
                y_norm = y_norm.to(device, non_blocking=pin)

                pred_norm = model(x)

                # loss in normalized space (stays on device)
                val_loss += criterion(pred_norm, y_norm).item()

                # move to CPU for un-normalization / MAE in real units
                pred_norm_cpu = pred_norm.cpu()
                y_norm_cpu = y_norm.cpu()

                for i, seq in enumerate(seq_list):
                    mean = seq_stats[seq]["mean"]
                    std = seq_stats[seq]["std"]

                    pred = pred_norm_cpu[i] * std + mean
                    true = y_norm_cpu[i] * std + mean

                    mae_az += float(torch.abs(pred[0] - true[0]).item())
                    mae_el += float(torch.abs(pred[1] - true[1]).item())
                    n += 1

        val_loss /= len(val_loader)
        mae_az /= max(1, n)
        mae_el /= max(1, n)

        dt = time.time() - t0

        print(
            f"Epoch {epoch:02d} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"MAE_az {mae_az:.3f} | "
            f"MAE_el {mae_el:.3f} | "
            f"{dt:.2f}s/epoch"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "laser_net_best.pt")
            print("  ✓ New best checkpoint")

    print("Training complete — best model saved to laser_net_best.pt")


if __name__ == "__main__":
    main()
