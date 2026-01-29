import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as T

# IMPORTANT: IMG_SIZE IS (H, W) for torchvision Resize
DEFAULT_IMG_SIZE: Tuple[int, int] = (180, 320)  # (H, W)


@dataclass
class Stats:
    mean: torch.Tensor  # [2]
    std: torch.Tensor   # [2]


class LaserDatasetRef(Dataset):
    """
    Returns: inp, y_norm, seq, y_raw
      - inp: image input (single/concat/diff)
      - y_norm: normalized target (possibly seq-centered before normalization)
      - seq: sequence string (e.g., "seq5")
      - y_raw: original raw label (no centering, no normalization)
    """
    def __init__(
        self,
        root: str,
        seqs: Iterable[str],
        img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,  # (H, W)
        csv_root: Optional[str] = None,
        img_root: Optional[str] = None,
        label_norm: str = "global",        # "global" | "none"
        label_center: str = "none",        # "none" | "seq"
        strict: bool = False,
        augment: bool = False,
        mode: str = "single",              # "single" | "concat" | "diff"
        stats_override: Optional[Stats] = None,  # use train stats for val/test
        seq_center_override: Optional[Dict[str, torch.Tensor]] = None,  # use train seq means for val/test
    ):
        self.root = root
        self.seqs = list(seqs)
        self.csv_root = csv_root or root
        self.img_root = img_root or root
        self.img_size = img_size
        self.label_norm = label_norm
        self.label_center = label_center
        self.strict = strict
        self.augment = augment
        self.mode = mode
        self.stats_override = stats_override
        self.seq_center_override = seq_center_override

        if self.label_center not in ("none", "seq"):
            raise ValueError(f"label_center must be 'none' or 'seq', got {self.label_center}")

        # ---------------- Transform ----------------
        tf_list = [
            T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR),
        ]
        if augment:
            tf_list += [
                T.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        tf_list += [
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
        ]
        self.transform = T.Compose(tf_list)
        # --------------------------------------------------------

        # Cache reference tensors per sequence
        self.ref_tensor_by_seq: Dict[str, torch.Tensor] = {}

        # Samples: (problem_img_path, y_raw, seq)
        self.samples: List[Tuple[str, torch.Tensor, str]] = []

        # Collect raw labels per seq for seq-centering
        labels_by_seq: Dict[str, List[torch.Tensor]] = {}

        for seq in self.seqs:
            labels_by_seq[seq] = []

            csv_path = os.path.join(self.csv_root, f"{seq}.csv")
            seq_dir = os.path.join(self.img_root, seq)

            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV not found: {csv_path}")
            if not os.path.isdir(seq_dir):
                raise FileNotFoundError(f"Sequence folder not found: {seq_dir}")

            # Reference image
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
                problem_path = os.path.join(seq_dir, f"{seq}_{pic} 1.tga")

                if not os.path.exists(problem_path):
                    msg = f"[WARN] Missing image, skipping: {problem_path}"
                    if strict:
                        raise FileNotFoundError(msg)
                    print(msg)
                    continue

                az = float(row["delta_azimuth"])
                el = float(row["delta_elevation"])
                y_raw = torch.tensor([az, el], dtype=torch.float32)

                self.samples.append((problem_path, y_raw, seq))
                labels_by_seq[seq].append(y_raw)

        if len(self.samples) == 0:
            raise RuntimeError("No samples found.")

        # ---------------- Sequence means (for centering) ----------------
        if self.label_center == "seq":
            if self.seq_center_override is not None:
                # Use train-provided seq means (val/test path)
                self.seq_means = self.seq_center_override
            else:
                # Compute from this dataset (train path)
                self.seq_means: Dict[str, torch.Tensor] = {}
                for seq in self.seqs:
                    ys = labels_by_seq.get(seq, [])
                    if len(ys) == 0:
                        # no samples -> default zero mean
                        self.seq_means[seq] = torch.zeros(2, dtype=torch.float32)
                    else:
                        self.seq_means[seq] = torch.stack(ys, dim=0).mean(dim=0)
        else:
            self.seq_means = {}
        # ---------------------------------------------------------------

        # ---------------- Global stats (for normalization) --------------
        # IMPORTANT: if seq-centering is enabled, compute global stats on CENTERED labels
        centered_labels_all: List[torch.Tensor] = []
        if self.label_center == "seq":
            for _img_path, y_raw, seq in self.samples:
                centered_labels_all.append(y_raw - self.seq_means.get(seq, torch.zeros(2)))
        else:
            for _img_path, y_raw, _seq in self.samples:
                centered_labels_all.append(y_raw)

        t_all = torch.stack(centered_labels_all, dim=0)
        self.global_stats = Stats(
            mean=t_all.mean(dim=0),
            std=t_all.std(dim=0).clamp_min(1e-6),
        )
        # ---------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def _center_label(self, y_raw: torch.Tensor, seq: str) -> torch.Tensor:
        if self.label_center == "seq":
            return y_raw - self.seq_means.get(seq, torch.zeros(2, dtype=torch.float32))
        return y_raw

    def _norm_label(self, y_centered: torch.Tensor) -> torch.Tensor:
        if self.label_norm == "none":
            return y_centered
        stats = self.stats_override if self.stats_override is not None else self.global_stats
        return (y_centered - stats.mean) / stats.std

    def __getitem__(self, idx):
        img_path, y_raw, seq = self.samples[idx]

        # current image
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)

        # reference image (cached)
        ref = self.ref_tensor_by_seq[seq]

        if self.mode == "single":
            inp = x
        elif self.mode == "concat":
            inp = torch.cat([x, ref], dim=0)   # [6,H,W]
        elif self.mode == "diff":
            inp = x - ref                      # [3,H,W]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        y_centered = self._center_label(y_raw, seq)
        y_norm = self._norm_label(y_centered)

        return inp, y_norm, seq, y_raw
