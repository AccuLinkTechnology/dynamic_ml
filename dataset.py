import os
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as T

from config import DEFAULT_IMG_SIZE_HW, IMG_EXT, CSV_REQUIRED_COLUMNS, REF_SUFFIX


class LaserDataset(Dataset):

    def __init__(
        self,
        root: str,
        seqs: Iterable[str],
        img_size_hw: Tuple[int, int] = DEFAULT_IMG_SIZE_HW,
        csv_root: Optional[str] = None,
        img_root: Optional[str] = None,
        strict: bool = False,
        transform: Optional[torch.nn.Module] = None,
    ):
        self.root = root
        self.seqs = list(seqs)
        self.csv_root = csv_root or root
        self.img_root = img_root or root
        self.img_size_hw = img_size_hw
        self.strict = strict

        H, W = self.img_size_hw

        # Minimal + consistent preprocessing (option 1): Resize + [-1,1]
        self.transform = transform or T.Compose([
            T.Resize((H, W), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.ref_tensor_by_seq: Dict[str, torch.Tensor] = {}
        self.samples: List[Tuple[str, torch.Tensor, str, int]] = []

        for seq in self.seqs:
            csv_path = os.path.join(self.csv_root, f"{seq}.csv")
            seq_dir = os.path.join(self.img_root, seq)

            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV not found: {csv_path}")
            if not os.path.isdir(seq_dir):
                raise FileNotFoundError(f"Sequence folder not found: {seq_dir}")

            ref_path = os.path.join(seq_dir, f"{seq}{REF_SUFFIX}{IMG_EXT}")
            if not os.path.exists(ref_path):
                raise FileNotFoundError(f"Reference image not found: {ref_path}")

            ref_img = Image.open(ref_path).convert("RGB")
            self.ref_tensor_by_seq[seq] = self.transform(ref_img)

            df = pd.read_csv(csv_path)
            missing = [c for c in CSV_REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(f"{csv_path} missing columns: {missing}")

            df = df.dropna(subset=CSV_REQUIRED_COLUMNS)
            skipped = 0
            for _, row in df.iterrows():
                pic = int(row["pic_number"])
                img_path = os.path.join(seq_dir, f"{seq}_{pic}{IMG_EXT}")

                if not os.path.exists(img_path):
                    if strict:
                        raise FileNotFoundError(f"Missing image: {img_path}")
                    skipped += 1
                    continue

                y = torch.tensor(
                    [
                        float(row["azimuth"]),
                        float(row["elevation"]),
                        float(row["rotation"]),
                        float(row["distance"]),
                    ],
                    dtype=torch.float32,
                )
                self.samples.append((img_path, y, seq, pic))

            loaded = len(df) - skipped
            print(f"[{seq}] loaded={loaded} skipped={skipped}")

        if len(self.samples) == 0:
            raise RuntimeError("No samples found.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, y, seq, _pic = self.samples[idx]
        img_pil = Image.open(img_path).convert("RGB")
        x = self.transform(img_pil)
        ref = self.ref_tensor_by_seq[seq]
        inp = torch.cat([x, x - ref], dim=0)
        return inp, y, seq, img_path