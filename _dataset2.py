import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as T


@dataclass
class Stats:
    mean: torch.Tensor  # [2]
    std: torch.Tensor   # [2]


class LaserDataset(Dataset):
    """
    Loads (image -> delta_az, delta_el) regression samples.
    """

    def __init__(
        self,
        root: str,
        seqs: Iterable[str],
        img_size: Tuple[int, int] = (320,180),
        csv_root: Optional[str] = None,
        img_root: Optional[str] = None,
        label_norm: str = "global",  # "global" | "per_seq" | "none"
        strict: bool = False,
    ):
        self.root = root
        self.seqs = list(seqs)
        self.csv_root = csv_root or root
        self.img_root = img_root or root
        self.img_size = img_size
        self.label_norm = label_norm
        self.strict = strict

        self.transform = T.Compose(
            [
                T.Resize(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # samples: (img_path, label[2], seq)
        self.samples: List[Tuple[str, torch.Tensor, str]] = []

        # collect labels for stats
        labels_all: List[List[float]] = []
        labels_by_seq: Dict[str, List[List[float]]] = {s: [] for s in self.seqs}

        for seq in self.seqs:
            csv_path = os.path.join(self.csv_root, f"{seq}.csv")
            seq_dir = os.path.join(self.img_root, seq)

            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV not found: {csv_path}")
            if not os.path.isdir(seq_dir):
                raise FileNotFoundError(f"Sequence folder not found: {seq_dir}")

            df = pd.read_csv(csv_path)

            required = {"pic_number", "delta_azimuth", "delta_elevation"}
            if not required.issubset(df.columns):
                raise ValueError(
                    f"{csv_path} missing columns. Found {list(df.columns)}, need {sorted(required)}"
                )

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

                label = torch.tensor([az, el], dtype=torch.float32)
                self.samples.append((img_path, label, seq))

                labels_all.append([az, el])
                labels_by_seq[seq].append([az, el])

        if len(self.samples) == 0:
            raise RuntimeError("No samples found. Check root/seq list/path patterns.")

        # stats
        self.seq_stats: Dict[str, Stats] = {}
        for seq, arr in labels_by_seq.items():
            if len(arr) == 0:
                continue
            t = torch.tensor(arr, dtype=torch.float32)
            mean = t.mean(dim=0)
            std = t.std(dim=0).clamp_min(1e-6)
            self.seq_stats[seq] = Stats(mean=mean, std=std)

        t_all = torch.tensor(labels_all, dtype=torch.float32)
        self.global_stats = Stats(
            mean=t_all.mean(dim=0),
            std=t_all.std(dim=0).clamp_min(1e-6),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _norm_label(self, y: torch.Tensor, seq: str) -> torch.Tensor:
        if self.label_norm == "none":
            return y
        if self.label_norm == "per_seq":
            st = self.seq_stats[seq]
            return (y - st.mean) / st.std
        # default: global
        return (y - self.global_stats.mean) / self.global_stats.std

    def __getitem__(self, idx: int):
        img_path, y_raw, seq = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y_norm = self._norm_label(y_raw, seq)
        return x, y_norm, seq, y_raw
