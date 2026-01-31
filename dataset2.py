import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as T

from cross_roi import build_cross_weight_mask, apply_cross_roi_weight, debug_overlay

# NOTE: torchvision Resize expects (H, W)
DEFAULT_IMG_SIZE_HW: Tuple[int, int] = (180, 320)  # (H, W)


@dataclass
class Stats:
    mean: torch.Tensor  # [2]
    std: torch.Tensor   # [2]


class LaserDatasetRef(Dataset):
    """
    Returns:
      inp:      [C,H,W]  (C=3 for diff/single, C=6 for concat)
      y_norm:   [2]      normalized command label (baseline-subtracted)
      seq:      str
      y_cmd:    [2]      command label in real units (baseline-subtracted)
      y_raw:    [2]      raw label from CSV (original)
      img_path: str      path (for debugging/overlays)
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
        mode: str = "concat",  # "single" | "concat" | "diff"
        # Label processing:
        label_norm: str = "global",  # "global" | "none"
        baseline_strategy: str = "stable_m",  # "none" | "stable_m"
        baseline_m: int = 10,               # use first M frames for baseline estimate
        # Stable cache:
        stable_pool: int = 3,               # how many stable frames to cache per seq
        # ROI emphasis:
        use_cross_roi: bool = True,
        roi_bg_weight: float = 0.20,        # background weight in [0..1], lower = stronger focus
        roi_strength: float = 1.00,         # 1.0 = use mask as-is; >1 boosts ROI contrast slightly
        # IMPORTANT: make val use train stats
        external_stats: Optional[Stats] = None,
        # IMPORTANT: make ROI stable per sequence
        roi_from: str = "ref",              # "ref" | "frame" (ref is recommended)
    ):
        self.root = root
        self.seqs = list(seqs)
        self.csv_root = csv_root or root
        self.img_root = img_root or root
        self.img_size_hw = img_size_hw
        self.label_norm = label_norm
        self.strict = strict
        self.augment = augment
        self.mode = mode
        self.baseline_strategy = baseline_strategy
        self.baseline_m = int(baseline_m)
        self.stable_pool = int(stable_pool)

        self.use_cross_roi = bool(use_cross_roi)
        self.roi_bg_weight = float(roi_bg_weight)
        self.roi_strength = float(roi_strength)
        self.external_stats = external_stats
        self.roi_from = str(roi_from)

        if self.roi_from not in ("ref", "frame"):
            raise ValueError(f"roi_from must be 'ref' or 'frame', got: {self.roi_from}")

        # ---------------- transforms ----------------
        base_tf = [
            T.Resize(self.img_size_hw_allow_hw(), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),  # [0,1]
        ]

        if augment:
            # mild physically-meaningful augments
            base_tf = [
                T.Resize(self.img_size_hw_allow_hw(), interpolation=T.InterpolationMode.BILINEAR),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
                T.RandomAffine(
                    degrees=0.0,
                    translate=(0.02, 0.02),
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.ToTensor(),
            ]

        # normalize to [-1,1]
        base_tf.append(
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
        )
        self.transform = T.Compose(base_tf)
        # -------------------------------------------

        # Cached reference tensors per sequence (normalized [-1,1], ROI NOT baked in)
        self.ref_tensor_by_seq: Dict[str, torch.Tensor] = {}

        # Optional cached ROI masks per sequence: [1,H,W] float in [bg..1]
        self.roi_mask_by_seq: Dict[str, Optional[torch.Tensor]] = {}

        # Baseline per seq in REAL units (az, el)
        self.baseline_by_seq: Dict[str, torch.Tensor] = {}

        # Cached stable inputs per seq: list of tensors shaped like model input (diff/single/concat)
        self.stable_inputs_by_seq: Dict[str, List[torch.Tensor]] = {}

        # Samples store: (img_path, y_raw, seq, pic_number)
        self.samples: List[Tuple[str, torch.Tensor, str, int]] = []

        # Build all samples and per-seq label lists for baseline
        per_seq_rows: Dict[str, List[Tuple[int, str, torch.Tensor]]] = {}  # seq -> [(pic, path, y_raw)]
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

            # Cache ROI mask per sequence (recommended: from reference)
            if self.use_cross_roi and self.roi_from == "ref":
                self.roi_mask_by_seq[seq] = self._roi_weight_from_pil(ref_img)
            else:
                self.roi_mask_by_seq[seq] = None

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

        # ---------------- baseline per seq ----------------
        for seq, rows in per_seq_rows.items():
            if self.baseline_strategy == "none":
                self.baseline_by_seq[seq] = torch.zeros(2, dtype=torch.float32)
                continue

            if self.baseline_strategy == "stable_m":
                if len(rows) == 0:
                    self.baseline_by_seq[seq] = torch.zeros(2, dtype=torch.float32)
                    continue
                m = min(self.baseline_m, len(rows))
                y_stack = torch.stack([rows[i][2] for i in range(m)], dim=0)  # [m,2]
                self.baseline_by_seq[seq] = y_stack.mean(dim=0)
            else:
                raise ValueError(f"Unknown baseline_strategy: {self.baseline_strategy}")

        # ---------------- build cached stable inputs ----------------
        for seq, rows in per_seq_rows.items():
            ref = self.ref_tensor_by_seq[seq]
            stable_list: List[torch.Tensor] = []

            k = min(self.stable_pool, len(rows))
            for i in range(k):
                _pic, img_path, _y_raw = rows[i]
                img_pil = Image.open(img_path).convert("RGB")

                # x is normalized [-1,1]
                x = self.transform(img_pil)

                # ROI weighting (stable per-seq, or from each frame if configured)
                if self.use_cross_roi:
                    if self.roi_from == "ref":
                        w = self.roi_mask_by_seq[seq]
                        if w is not None:
                            x2 = apply_cross_roi_weight(x, w)
                            ref2 = apply_cross_roi_weight(ref, w)
                        else:
                            x2, ref2 = x, ref
                    else:
                        w = self._roi_weight_from_pil(img_pil)
                        x2 = apply_cross_roi_weight(x, w)
                        ref2 = apply_cross_roi_weight(ref, w)
                else:
                    x2, ref2 = x, ref

                inp = self._make_input(x2, ref2)
                stable_list.append(inp)

            if len(stable_list) == 0:
                # fallback
                if self.mode == "single":
                    stable_list = [ref]
                elif self.mode == "concat":
                    stable_list = [torch.cat([ref, ref], dim=0)]
                elif self.mode == "diff":
                    stable_list = [ref - ref]
                else:
                    raise ValueError(f"Unknown mode {self.mode}")

            self.stable_inputs_by_seq[seq] = stable_list

        # ---------------- global label stats on COMMAND ----------------
        y_cmd_all = []
        for _path, y_raw, seq, _pic in self.samples:
            y_cmd = y_raw - self.baseline_by_seq[seq]
            y_cmd_all.append(y_cmd)

        t_all = torch.stack(y_cmd_all, dim=0)  # [N,2]

        if self.external_stats is not None:
            # Use train stats for val (critical for correctness)
            self.global_stats = self.external_stats
        else:
            self.global_stats = Stats(
                mean=t_all.mean(dim=0),
                std=t_all.std(dim=0).clamp_min(1e-6),
            )

    def img_size_hw_allow_hw(self) -> Tuple[int, int]:
        # torchvision uses (H, W); keep helper in case you want to swap later
        return self.img_size_hw

    def __len__(self):
        return len(self.samples)

    def _roi_weight_from_pil(self, img_pil: Image.Image) -> torch.Tensor:
        """
        Returns [1,H,W] weight map in [roi_bg_weight..1], optionally sharpened by roi_strength.
        """
        rr = build_cross_weight_mask(img_pil, out_hw=self.img_size_hw, bg_weight=self.roi_bg_weight)
        w = rr.mask_hw  # already in [bg..1]

        # Optional "strength": increase contrast between ROI and bg
        if self.roi_strength != 1.0:
            bg = self.roi_bg_weight
            raw = (w - bg) / max(1e-6, (1.0 - bg))
            raw = raw.clamp(0.0, 1.0)
            raw = raw.pow(1.0 / max(1e-6, self.roi_strength))
            w = bg + (1.0 - bg) * raw

        return w

    def _make_input(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if self.mode == "single":
            return x
        if self.mode == "concat":
            return torch.cat([x, ref], dim=0)  # [6,H,W]
        if self.mode == "diff":
            return x - ref
        raise ValueError(f"Unknown mode {self.mode}")

    def _norm_label(self, y_cmd: torch.Tensor) -> torch.Tensor:
        if self.label_norm == "none":
            return y_cmd
        return (y_cmd - self.global_stats.mean) / self.global_stats.std

    def get_stable_input(self, seq: str, stable_idx: int = 0) -> torch.Tensor:
        pool = self.stable_inputs_by_seq[seq]
        if len(pool) == 0:
            raise RuntimeError(f"No stable pool for seq={seq}")
        stable_idx = int(stable_idx) % len(pool)
        return pool[stable_idx]

    def __getitem__(self, idx: int):
        img_path, y_raw, seq, _pic = self.samples[idx]

        img_pil = Image.open(img_path).convert("RGB")
        x = self.transform(img_pil)          # [-1,1]
        ref = self.ref_tensor_by_seq[seq]    # [-1,1]

        if self.use_cross_roi:
            if self.roi_from == "ref":
                w = self.roi_mask_by_seq[seq]
                if w is not None:
                    x = apply_cross_roi_weight(x, w)
                    ref = apply_cross_roi_weight(ref, w)
            else:
                w = self._roi_weight_from_pil(img_pil)
                x = apply_cross_roi_weight(x, w)
                ref = apply_cross_roi_weight(ref, w)

        inp = self._make_input(x, ref)

        y_cmd = y_raw - self.baseline_by_seq[seq]
        y_norm = self._norm_label(y_cmd)

        return inp, y_norm, seq, y_cmd, y_raw, img_path


# ---------------- Debug helper ----------------
if __name__ == "__main__":
    # Example:
    # python3 dataset2.py /workspace/dynamic_ml/train2/seq2/seq2_0\ 1.tga
    import sys
    if len(sys.argv) >= 2:
        p = sys.argv[1]
        img = Image.open(p).convert("RGB")
        out = debug_overlay(img, out_hw=DEFAULT_IMG_SIZE_HW)
        out.save("cross_debug_overlay.png")
        print("wrote cross_debug_overlay.png")
    else:
        print("usage: python3 dataset2.py /path/to/frame.tga")
