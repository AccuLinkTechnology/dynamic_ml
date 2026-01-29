import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as T

# IMPORTANT: IMG_SIZE IS (H, W)
DEFAULT_IMG_SIZE: Tuple[int, int] = (180, 320)  # (H, W)


@dataclass
class Stats:
    mean: torch.Tensor  # [2]
    std: torch.Tensor   # [2]


class LaserDatasetRef(Dataset):
    """
    Command-space dataset for incremental control.

    We build a per-sequence baseline y0 and train on:
        y_cmd(t) = y_raw(t) - y0(seq)

    Baseline options:
      - "stable_m": choose M frames whose images are closest to the ref image (lowest mean abs diff),
                    and average their labels as y0.
      - "first_k":  choose first K smallest pic_number and average their labels as y0.
      - "seq_mean": average all labels in the sequence as y0.

    Returns:
      inp, y_norm, seq, y_cmd_raw, y_raw
    """
    def __init__(
        self,
        root: str,
        seqs: Iterable[str],
        img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
        csv_root: Optional[str] = None,
        img_root: Optional[str] = None,
        label_norm: str = "global",          # "global" | "none"
        strict: bool = False,
        augment: bool = False,
        mode: str = "diff",                  # "single" | "concat" | "diff"
        stats_override: Optional[Stats] = None,

        baseline_strategy: str = "stable_m", # "stable_m" | "first_k" | "seq_mean"
        baseline_k: int = 5,                 # used by first_k
        baseline_m: int = 10,                # used by stable_m
    ):
        self.root = root
        self.seqs = list(seqs)
        self.csv_root = csv_root or root
        self.img_root = img_root or root
        self.img_size = img_size
        self.label_norm = label_norm
        self.strict = strict
        self.augment = augment
        self.mode = mode
        self.stats_override = stats_override

        self.baseline_strategy = baseline_strategy
        self.baseline_k = int(baseline_k)
        self.baseline_m = int(baseline_m)

        if self.baseline_strategy not in ("stable_m", "first_k", "seq_mean"):
            raise ValueError(f"baseline_strategy must be stable_m|first_k|seq_mean, got {baseline_strategy}")
        if self.baseline_k <= 0:
            raise ValueError("baseline_k must be >= 1")
        if self.baseline_m <= 0:
            raise ValueError("baseline_m must be >= 1")

        # ---------- Transform ----------
        tf_list = [
            T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR),
        ]
        # NOTE: augmentation disabled by default; if enabled later, should be paired for ref+current.
        if augment:
            tf_list += [T.ColorJitter(brightness=0.2, contrast=0.2)]
        tf_list += [
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
        ]
        self.transform = T.Compose(tf_list)
        # -----------------------------

        # Cache reference tensor per seq
        self.ref_tensor_by_seq: Dict[str, torch.Tensor] = {}

        # Samples: (img_path, pic_number, y_raw, seq)
        self.samples: List[Tuple[str, int, torch.Tensor, str]] = []

        # Per-seq list of (pic_number, img_path, y_raw)
        per_seq: Dict[str, List[Tuple[int, str, torch.Tensor]]] = {s: [] for s in self.seqs}

        # Load metadata + ref tensors
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

                self.samples.append((img_path, pic, y_raw, seq))
                per_seq[seq].append((pic, img_path, y_raw))

        if len(self.samples) == 0:
            raise RuntimeError("No samples found.")

        # ---------- Build baseline per seq ----------
        self.baseline_by_seq: Dict[str, torch.Tensor] = {}
        for seq in self.seqs:
            items = per_seq.get(seq, [])
            if len(items) == 0:
                self.baseline_by_seq[seq] = torch.zeros(2, dtype=torch.float32)
                continue

            if self.baseline_strategy == "seq_mean":
                ys = torch.stack([y for _pic, _path, y in items], dim=0)
                self.baseline_by_seq[seq] = ys.mean(dim=0)
                continue

            if self.baseline_strategy == "first_k":
                items_sorted = sorted(items, key=lambda t: t[0])
                k = min(self.baseline_k, len(items_sorted))
                ys0 = torch.stack([items_sorted[i][2] for i in range(k)], dim=0)
                self.baseline_by_seq[seq] = ys0.mean(dim=0)
                continue

            # stable_m: pick frames with smallest image diff magnitude to ref
            ref = self.ref_tensor_by_seq[seq]
            scored: List[Tuple[float, torch.Tensor]] = []
            for _pic, img_path, y in items:
                img = Image.open(img_path).convert("RGB")
                x = self.transform(img)
                score = (x - ref).abs().mean().item()  # scalar “how disturbed”
                scored.append((score, y))
            scored.sort(key=lambda t: t[0])
            m = min(self.baseline_m, len(scored))
            ys0 = torch.stack([scored[i][1] for i in range(m)], dim=0)
            self.baseline_by_seq[seq] = ys0.mean(dim=0)

        # ---------- Global stats computed in COMMAND space ----------
        cmd_all: List[torch.Tensor] = []
        for _img_path, _pic, y_raw, seq in self.samples:
            y0 = self.baseline_by_seq.get(seq, torch.zeros(2, dtype=torch.float32))
            cmd_all.append(y_raw - y0)

        t_all = torch.stack(cmd_all, dim=0)
        self.global_stats = Stats(
            mean=t_all.mean(dim=0),
            std=t_all.std(dim=0).clamp_min(1e-6),
        )

    def __len__(self):
        return len(self.samples)

    def _norm(self, y_cmd: torch.Tensor) -> torch.Tensor:
        if self.label_norm == "none":
            return y_cmd
        stats = self.stats_override if self.stats_override is not None else self.global_stats
        return (y_cmd - stats.mean) / stats.std

    def __getitem__(self, idx):
        img_path, _pic, y_raw, seq = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)

        ref = self.ref_tensor_by_seq[seq]

        if self.mode == "single":
            inp = x
        elif self.mode == "concat":
            inp = torch.cat([x, ref], dim=0)  # [6,H,W]
        elif self.mode == "diff":
            inp = x - ref                     # [3,H,W]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        y0 = self.baseline_by_seq.get(seq, torch.zeros(2, dtype=torch.float32))
        y_cmd = y_raw - y0
        y_norm = self._norm(y_cmd)

        return inp, y_norm, seq, y_cmd, y_raw
