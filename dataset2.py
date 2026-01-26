import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as T


# Easy toggle: use (320,180) or (640,360)
DEFAULT_IMG_SIZE: Tuple[int, int] = (320, 180)  # (W, H)


@dataclass
class Stats:
    mean: torch.Tensor  # [2]
    std: torch.Tensor   # [2]


class LaserDatasetRef(Dataset):
    def __init__(
        self,
        root: str,
        seqs: Iterable[str],
        img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
        csv_root: Optional[str] = None,
        img_root: Optional[str] = None,
        label_norm: str = "global",    # "global" | "none"
        strict: bool = False,
        augment: bool = False,
        mode: str = "single",          # "single" | "concat" | "diff"
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

        # ---------------- FIX #1: define transform ----------------
        base_tf = [
            T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ]

        if augment:
            base_tf = [
                T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
            ]

        base_tf.append(T.Normalize(mean=[0.5, 0.5, 0.5],
                                   std=[0.5, 0.5, 0.5]))

        self.transform = T.Compose(base_tf)
        # --------------------------------------------------------

        # Cache reference tensors per sequence
        self.ref_tensor_by_seq: Dict[str, torch.Tensor] = {}

        # Samples: (problem_img_path, y_raw, seq)
        self.samples: List[Tuple[str, torch.Tensor, str]] = []

        labels_all: List[List[float]] = []

        for seq in self.seqs:
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
                labels_all.append([az, el])

        if len(self.samples) == 0:
            raise RuntimeError("No samples found.")

        # Global label stats
        t_all = torch.tensor(labels_all, dtype=torch.float32)
        self.global_stats = Stats(
            mean=t_all.mean(dim=0),
            std=t_all.std(dim=0).clamp_min(1e-6),
        )

    def __len__(self):
        return len(self.samples)

    # ---------------- FIX #2: signature mismatch ----------------
    def _norm_label(self, y: torch.Tensor):
        if self.label_norm == "none":
            return y
        return (y - self.global_stats.mean) / self.global_stats.std
    # -----------------------------------------------------------

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
            inp = x - ref                     # [3,H,W]

        else:
            raise ValueError(f"Unknown mode {self.mode}")

        y_norm = self._norm_label(y_raw)
        return inp, y_norm, seq, y_raw
