# dataset3.py
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as T

DEFAULT_IMG_SIZE_HW: Tuple[int, int] = (180, 320)  # (H, W)


@dataclass
class Stats:
    mean: torch.Tensor  # [2]
    std: torch.Tensor   # [2]


class LaserDatasetV3(Dataset):
    """
    Simplified dataset without pose supervision.
    Returns:
      inp:      [6,H,W]      = cat([x, x-ref]) where x is current normalized [-1,1]
      y_cmd:    [2]          command label (baseline-subtracted, optionally normalized)
      seq:      str          sequence name
      img_path: str          path to image
    """

    def __init__(
        self,
        root: str,
        seqs: Iterable[str],
        img_size_hw: Tuple[int, int] = DEFAULT_IMG_SIZE_HW,
        csv_root: Optional[str] = None,
        img_root: Optional[str] = None,
        strict: bool = False,
        augment: bool = False,
        # Label processing:
        baseline_strategy: str = "stable_median",  # "none" | "stable_mean" | "stable_median"
        baseline_m: int = 10,
        stable_pool: int = 5,
        # Normalization:
        normalize_labels: bool = True,  # per-sequence z-score normalization
    ):
        self.root = root
        self.seqs = list(seqs)
        self.csv_root = csv_root or root
        self.img_root = img_root or root
        self.img_size_hw = img_size_hw
        self.strict = strict
        self.augment = augment
        self.baseline_strategy = baseline_strategy
        self.baseline_m = int(baseline_m)
        self.stable_pool = int(stable_pool)
        self.normalize_labels = normalize_labels

        H, W = self.img_size_hw

        # ---------------- transforms ----------------
        base_tf = [
            T.Resize((H, W), interpolation=T.InterpolationMode.BILINEAR),
        ]

        if augment:
            # Reduced augmentation - avoid breaking cross detection
            base_tf += [
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
                T.RandomAffine(
                    degrees=1.5,
                    translate=(0.02, 0.02),
                    scale=(0.97, 1.03),
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
            ]

        base_tf += [
            T.ToTensor(),  # [0,1]
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -> [-1,1]
        ]

        self.transform = T.Compose(base_tf)
        # -------------------------------------------

        self.ref_tensor_by_seq: Dict[str, torch.Tensor] = {}
        self.baseline_by_seq: Dict[str, torch.Tensor] = {}
        self.stable_inputs_by_seq: Dict[str, List[torch.Tensor]] = {}
        self.seq_stats: Dict[str, Stats] = {}  # per-sequence label statistics

        # (img_path, y_raw, seq, pic_number)
        self.samples: List[Tuple[str, torch.Tensor, str, int]] = []

        per_seq_rows: Dict[str, List[Tuple[int, str, torch.Tensor]]] = {}
        
        # First pass: load all data
        for seq in self.seqs:
            csv_path = os.path.join(self.csv_root, f"{seq}.csv")
            seq_dir = os.path.join(self.img_root, seq)

            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV not found: {csv_path}")
            if not os.path.isdir(seq_dir):
                raise FileNotFoundError(f"Sequence folder not found: {seq_dir}")

            ref_path = os.path.join(seq_dir, f"{seq}_start 1.tga")
            if not os.path.exists(ref_path):
                raise FileNotFoundError(f"Reference start image not found: {ref_path}")

            ref_img = Image.open(ref_path).convert("RGB")
            self.ref_tensor_by_seq[seq] = self.transform(ref_img)

            df = pd.read_csv(csv_path)
            required = {"pic_number", "delta_azimuth", "delta_elevation"}
            if not required.issubset(df.columns):
                raise ValueError(f"{csv_path} missing columns {required}")

            rows: List[Tuple[int, str, torch.Tensor]] = []
            for _, row in df.iterrows():
                pic = int(row["pic_number"])
                img_path = os.path.join(seq_dir, f"{seq}_{pic} 1.tga")

                if not os.path.exists(img_path):
                    msg = f"[WARN] Missing image, skipping: {img_path}"
                    if strict:
                        raise FileNotFoundError(msg)
                    print(msg)
                    continue

                az = float(row["delta_azimuth"])
                el = float(row["delta_elevation"])
                y_raw = torch.tensor([az, el], dtype=torch.float32)
                rows.append((pic, img_path, y_raw))

            rows.sort(key=lambda x: x[0])
            per_seq_rows[seq] = rows

            for pic, path, y_raw in rows:
                self.samples.append((path, y_raw, seq, pic))

        if len(self.samples) == 0:
            raise RuntimeError("No samples found.")

        # ---------------- Compute baselines ----------------
        for seq, rows in per_seq_rows.items():
            if self.baseline_strategy == "none":
                self.baseline_by_seq[seq] = torch.zeros(2, dtype=torch.float32)
            elif self.baseline_strategy == "stable_mean":
                if len(rows) == 0:
                    self.baseline_by_seq[seq] = torch.zeros(2, dtype=torch.float32)
                    continue
                m = min(self.baseline_m, len(rows))
                y_stack = torch.stack([rows[i][2] for i in range(m)], dim=0)  # [m,2]
                self.baseline_by_seq[seq] = y_stack.mean(dim=0)
            elif self.baseline_strategy == "stable_median":
                if len(rows) == 0:
                    self.baseline_by_seq[seq] = torch.zeros(2, dtype=torch.float32)
                    continue
                m = min(self.baseline_m, len(rows))
                y_stack = torch.stack([rows[i][2] for i in range(m)], dim=0)  # [m,2]
                self.baseline_by_seq[seq] = y_stack.median(dim=0)[0]
            else:
                raise ValueError(f"Unknown baseline_strategy: {self.baseline_strategy}")

        # ---------------- Per-sequence label statistics ----------------
        if self.normalize_labels:
            for seq, rows in per_seq_rows.items():
                if len(rows) == 0:
                    self.seq_stats[seq] = Stats(
                        mean=torch.zeros(2, dtype=torch.float32),
                        std=torch.ones(2, dtype=torch.float32)
                    )
                    continue
                
                # Compute stats on baseline-subtracted commands
                y_cmd_list = []
                baseline = self.baseline_by_seq[seq]
                for _, _, y_raw in rows:
                    y_cmd = y_raw - baseline
                    y_cmd_list.append(y_cmd)
                
                if len(y_cmd_list) > 0:
                    y_cmd_stack = torch.stack(y_cmd_list, dim=0)  # [N,2]
                    self.seq_stats[seq] = Stats(
                        mean=y_cmd_stack.mean(dim=0),
                        std=y_cmd_stack.std(dim=0).clamp_min(1e-3)
                    )
                else:
                    self.seq_stats[seq] = Stats(
                        mean=torch.zeros(2, dtype=torch.float32),
                        std=torch.ones(2, dtype=torch.float32)
                    )

        # ---------------- Stable cached inputs ----------------
        for seq, rows in per_seq_rows.items():
            ref = self.ref_tensor_by_seq[seq]
            stable_list: List[torch.Tensor] = []

            k = min(self.stable_pool, len(rows))
            for i in range(k):
                _pic, img_path, _y_raw = rows[i]
                img_pil = Image.open(img_path).convert("RGB")
                x = self.transform(img_pil)  # [-1,1]
                inp = torch.cat([x, x - ref], dim=0)  # [6,H,W]
                stable_list.append(inp)

            if len(stable_list) == 0:
                stable_list = [torch.cat([ref, ref - ref], dim=0)]

            self.stable_inputs_by_seq[seq] = stable_list

        # ---------------- Global label stats (for reference) ----------------
        y_cmd_all = []
        for _path, y_raw, seq, _pic in self.samples:
            y_cmd = y_raw - self.baseline_by_seq[seq]
            if self.normalize_labels:
                y_cmd = (y_cmd - self.seq_stats[seq].mean) / self.seq_stats[seq].std
            y_cmd_all.append(y_cmd)

        if len(y_cmd_all) > 0:
            t_all = torch.stack(y_cmd_all, dim=0)  # [N,2]
            self.global_stats = Stats(
                mean=t_all.mean(dim=0),
                std=t_all.std(dim=0).clamp_min(1e-6),
            )
        else:
            self.global_stats = Stats(
                mean=torch.zeros(2, dtype=torch.float32),
                std=torch.ones(2, dtype=torch.float32)
            )

    def __len__(self):
        return len(self.samples)

    def get_stable_input(self, seq: str, stable_idx: int = 0) -> torch.Tensor:
        """Get a stable input for calibration/zero-command testing."""
        pool = self.stable_inputs_by_seq[seq]
        stable_idx = int(stable_idx) % len(pool)
        return pool[stable_idx]

    def denormalize_cmd(self, y_norm: torch.Tensor, seq: str) -> torch.Tensor:
        """Convert normalized command back to real units."""
        if not self.normalize_labels:
            return y_norm
        stats = self.seq_stats[seq]
        # Move stats to same device as y_norm
        std = stats.std.to(y_norm.device)
        mean = stats.mean.to(y_norm.device)
        return y_norm * std + mean

    def __getitem__(self, idx: int):
        img_path, y_raw, seq, _pic = self.samples[idx]

        img_pil = Image.open(img_path).convert("RGB")
        x = self.transform(img_pil)          # [-1,1]
        ref = self.ref_tensor_by_seq[seq]    # [-1,1]

        inp = torch.cat([x, x - ref], dim=0)  # [6,H,W]

        # Baseline subtraction
        y_cmd = y_raw - self.baseline_by_seq[seq]

        # Optional per-sequence normalization
        if self.normalize_labels:
            stats = self.seq_stats[seq]
            y_cmd = (y_cmd - stats.mean) / stats.std

        return inp, y_cmd, seq, img_path