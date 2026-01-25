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
    """
    Reference-image dataset:
      input = concat(problem_RGB, reference_RGB) => 6-channel tensor [6,H,W]
      label = [delta_azimuth, delta_elevation]

    Expected folder layout (train2):
      train2/
        seq1.csv ... seq5.csv
        seq1 >seq1_start 1.tga etc
        seq2> esq2_0 1.tga etc
    Returns:
      x: [6,H,W] normalized to [-1,1]
      y_norm: [2] normalized by label_norm
      seq: string
      y_raw: [2] raw label
    """

    def __init__(
        self,
        root: str,
        seqs: Iterable[str],
        img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
        csv_root: Optional[str] = None,
        img_root: Optional[str] = None,
        label_norm: str = "global",    # "global" | "none"
        strict: bool = False,
        augment: bool = False,         # keep OFF for now
        mode: str = "single",  # "single" | "concat" | "diff"
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

        # Base transform (same for ref + problem)
        self.base_transform = T.Compose([
            T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),  # [0,1]
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -> [-1,1]
        ])

        # Optional light augment on PROBLEM ONLY (kept off by default)
        # (No crops/rotations/flips.)
        self.problem_transform_aug = T.Compose([
            T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Cache reference tensors per sequence (already transformed)
        self.ref_tensor_by_seq: Dict[str, torch.Tensor] = {}

        # Samples: (problem_img_path, y_raw, seq)
        self.samples: List[Tuple[str, torch.Tensor, str]] = []

        # Collect all labels for global stats
        labels_all: List[List[float]] = []

        for seq in self.seqs:
            csv_path = os.path.join(self.csv_root, f"{seq}.csv")
            seq_dir = os.path.join(self.img_root, seq)

            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV not found: {csv_path}")
            if not os.path.isdir(seq_dir):
                raise FileNotFoundError(f"Sequence folder not found: {seq_dir}")

            # Load reference image for this seq
            ref_path = os.path.join(seq_dir, f"{seq}_start 1.tga")
            if not os.path.exists(ref_path):
                raise FileNotFoundError(f"Reference start image not found: {ref_path}")

            ref_img = Image.open(ref_path).convert("RGB")
            ref_x = self.base_transform(ref_img)  # [3,H,W]
            self.ref_tensor_by_seq[seq] = ref_x

            df = pd.read_csv(csv_path)
            required = {"pic_number", "delta_azimuth", "delta_elevation"}
            if not required.issubset(df.columns):
                raise ValueError(
                    f"{csv_path} missing columns. Found {list(df.columns)}, need {sorted(required)}"
                )

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
            raise RuntimeError("No samples found. Check paths, CSVs, and naming patterns.")

        # Global label stats
        t_all = torch.tensor(labels_all, dtype=torch.float32)
        self.global_stats = Stats(
            mean=t_all.mean(dim=0),
            std=t_all.std(dim=0).clamp_min(1e-6),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _norm_label(self, y: torch.Tensor) -> torch.Tensor:
        if self.label_norm == "none":
            return y
        # default: global
        return (y - self.global_stats.mean) / self.global_stats.std

    def __getitem__(self, idx: int):
        img_path, y_raw, seq = self.samples[idx]

        # load current frame
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)

        # reference image
        ref_path = os.path.join(self.img_root, seq, f"{seq}_start 1.tga")
        ref_img = Image.open(ref_path).convert("RGB")
        ref = self.transform(ref_img)

        if self.mode == "single":
            inp = x

        elif self.mode == "concat":
            inp = torch.cat([x, ref], dim=0)  # 6 channels

        elif self.mode == "diff":
            inp = x - ref  # 3 channels difference image

        else:
            raise ValueError(f"Unknown mode {self.mode}")

        y_norm = self._norm_label(y_raw, seq)
        return inp, y_norm, seq, y_raw

