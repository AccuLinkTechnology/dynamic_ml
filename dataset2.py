import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as T

class LaserDataset(Dataset):
    """
    Returns:
      x: image tensor [3, H, W] normalized to [-1,1]
      y_norm: normalized label tensor [2] (per-sequence z-score)
      seq: sequence name string (e.g., "seq2")
    Also exposes:
      seq_stats: dict {seq: {"mean": tensor([2]), "std": tensor([2])}}
    """

    def __init__(self, data_root, seqs, img_size=(320, 180)):
        self.samples = []
        self.seqs = list(seqs)

        self.transform = T.Compose([
            T.Resize(img_size),               # PIL uses (W,H) internally; torchvision handles this
            T.ToTensor(),                     # -> [0,1]
            T.Normalize(mean=[0.5, 0.5, 0.5],  # -> [-1,1]
                        std=[0.5, 0.5, 0.5])
        ])

        # Collect labels per sequence to compute mean/std
        per_seq_labels = {seq: [] for seq in self.seqs}

        for seq in self.seqs:
            csv_path = os.path.join(data_root, f"{seq}.csv")
            seq_dir = os.path.join(data_root, seq)

            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                pic = int(row["pic_number"])
                img_path = os.path.join(seq_dir, f"{seq}_{pic} 1.tga")

                if not os.path.exists(img_path):
                    print(f"[WARN] Missing image, skipping: {img_path}")
                    continue

                az = float(row["delta_azimuth"])
                el = float(row["delta_elevation"])

                per_seq_labels[seq].append([az, el])
                label = torch.tensor([az, el], dtype=torch.float32)

                self.samples.append((img_path, label, seq))

        # Compute per-sequence mean/std
        self.seq_stats = {}
        for seq, arr in per_seq_labels.items():
            t = torch.tensor(arr, dtype=torch.float32)
            mean = t.mean(dim=0)
            std = t.std(dim=0).clamp_min(1e-6)
            self.seq_stats[seq] = {"mean": mean, "std": std}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, seq = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)

        mean = self.seq_stats[seq]["mean"]
        std = self.seq_stats[seq]["std"]
        y_norm = (label - mean) / std

        return x, y_norm, seq
