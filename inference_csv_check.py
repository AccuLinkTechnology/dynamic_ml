"""
Usage: python3 inference_csv_check.py <seq_name>
Example: python3 inference_csv_check.py seq1

Runs the model on every frame in the sequence, writes predictions to
train3/seqX/seqX_comparison.csv, then prints a per-output error summary.
"""

import sys
import os
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from model import LaserNetSimple
from config import (
    DATA_ROOT, IMG_EXT, REF_SUFFIX,
    DEFAULT_IMG_SIZE_HW, IN_CHANNELS, OUT_DIM,
    USE_TANH_BOUNDING, OUT_SCALE,
)

WEIGHTS = "/home/acculink/Documents/dynamic_ml/runs_train3/20260305_094954/best_model.pt"

transform = T.Compose([
    T.Resize(DEFAULT_IMG_SIZE_HW, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 inference_csv_check.py <seq_name>")
        sys.exit(1)

    seq = sys.argv[1]
    seq_dir = os.path.join(DATA_ROOT, seq)
    csv_path = os.path.join(seq_dir, f"{seq}.csv")
    out_path = os.path.join(seq_dir, f"{seq}_comparison.csv")

    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f"Sequence folder not found: {seq_dir}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    scale = torch.tensor(OUT_SCALE, dtype=torch.float32)

    model = LaserNetSimple(IN_CHANNELS, OUT_DIM, use_tanh_bounding=USE_TANH_BOUNDING, out_scale=OUT_SCALE)
    ckpt = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded: {WEIGHTS}")

    ref_path = os.path.join(seq_dir, f"{seq}{REF_SUFFIX}{IMG_EXT}")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference image not found: {ref_path}")
    ref = transform(Image.open(ref_path).convert("RGB"))

    df = pd.read_csv(csv_path, skipinitialspace=True)
    ref_row = df[df["pic_number"].astype(int) == 0].iloc[0]
    ref_label = {c: float(ref_row[c]) for c in ["x", "y", "rotation", "zoom"]}
    df = df[df["pic_number"].astype(int) != 0].copy()

    rows = []
    for _, row in df.iterrows():
        pic = int(row["pic_number"])
        img_path = os.path.join(seq_dir, f"{seq}_{pic}{IMG_EXT}")
        if not os.path.exists(img_path):
            print(f"[SKIP] Missing: {img_path}")
            continue

        cur = transform(Image.open(img_path).convert("RGB"))
        x_in = torch.cat([cur, cur - ref], dim=0).unsqueeze(0)

        with torch.no_grad():
            pred = (model(x_in)[0] * scale).tolist()

        # Predictions are displacements ? add ref label back to get absolute coords
        pred_x    = pred[0] + ref_label["x"]
        pred_y    = pred[1] + ref_label["y"]
        pred_rot  = pred[2] + ref_label["rotation"]
        pred_zoom = pred[3] + ref_label["zoom"]

        rows.append({
            "pic_number":  pic,
            "gt_x":        float(row["x"]),
            "gt_y":        float(row["y"]),
            "gt_rotation": float(row["rotation"]),
            "gt_zoom":     float(row["zoom"]),
            "pred_x":      round(pred_x, 3),
            "pred_y":      round(pred_y, 3),
            "pred_rotation": round(pred_rot, 3),
            "pred_zoom":   round(pred_zoom, 3),
            "err_x":       round(abs(float(row["x"]) - pred_x), 3),
            "err_y":       round(abs(float(row["y"]) - pred_y), 3),
            "err_rotation": round(abs(float(row["rotation"]) - pred_rot), 3),
            "err_zoom":    round(abs(float(row["zoom"]) - pred_zoom), 3),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"\nComparison written to: {out_path}")
    print(f"Frames evaluated: {len(out_df)}\n")

    print("=== Mean Absolute Error ===")
    for col in ["err_x", "err_y", "err_rotation", "err_zoom"]:
        print(f"  {col[4:]:>10}: {out_df[col].mean():.3f}  (max {out_df[col].max():.3f})")

    print("\n=== Worst predictions ===")
    for metric, label in [("err_x", "x"), ("err_y", "y"), ("err_rotation", "rotation"), ("err_zoom", "zoom")]:
        worst = out_df.loc[out_df[metric].idxmax()]
        print(f"  {label:>10}: pic {int(worst['pic_number']):>3} | "
              f"gt={worst[f'gt_{label}']:>8.2f} pred={worst[f'pred_{label}']:>8.2f} err={worst[metric]:.3f}")


if __name__ == "__main__":
    main()
