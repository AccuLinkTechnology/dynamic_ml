# cross_roi.py
# Minimal "math/CV" to bias the model toward the cross pattern:
# - build a soft mask from bright pixels (white lines) on dark background
# - multiply image tensors by that mask before normalization / model

from __future__ import annotations

import numpy as np
import cv2
import torch
from PIL import Image


def build_cross_soft_mask_bgr(
    bgr: np.ndarray,
    out_size: tuple[int, int] = (320, 180),
    *,
    percentile: float = 97.0,
    thr_floor: float = 0.60,
    thr_scale: float = 0.80,
    min_area_frac: float = 0.0008,
    close_ksize: int = 3,
    dilate_ksize: int = 5,
    blur_sigma: float = 3.0,
) -> np.ndarray:
    """
    Returns float32 mask in [0,1] shape (H,W) where cross-like bright structures get higher weight.
    This intentionally avoids fragile Hough-line logic; it's robust and minimal.

    out_size is (W,H) to match your IMG_SIZE=(180,320) -> (H,W).
    """
    out_w, out_h = out_size
    img = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    # adaptive threshold based on bright tail, but never below thr_floor
    thr = max(thr_floor, float(np.percentile(v, percentile)) * thr_scale)
    bin_ = (v > thr).astype(np.uint8)

    # remove tiny bright speckles (glints/compression noise)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_, connectivity=8)
    keep = np.zeros_like(bin_, dtype=np.uint8)
    min_area = int(min_area_frac * out_w * out_h)

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            keep[labels == i] = 1

    keep = (keep * 255).astype(np.uint8)

    # connect broken line segments and widen slightly so gradients survive downsampling
    if close_ksize > 1:
        keep = cv2.morphologyEx(
            keep,
            cv2.MORPH_CLOSE,
            np.ones((close_ksize, close_ksize), np.uint8),
            iterations=1,
        )

    if dilate_ksize > 1:
        keep = cv2.dilate(keep, np.ones((dilate_ksize, dilate_ksize), np.uint8), iterations=1)

    # blur -> soft attention rather than hard cutoff
    if blur_sigma > 0:
        keep = cv2.GaussianBlur(keep, (0, 0), sigmaX=blur_sigma)

    mask = keep.astype(np.float32) / 255.0
    if mask.max() > 1e-6:
        mask /= mask.max()
    return mask


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def preprocess_with_cross_mask(
    pil_img: Image.Image,
    out_hw: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    """
    Returns torch float tensor (3,H,W) normalized, after applying the cross soft mask.

    out_hw is (H,W) like your IMG_SIZE=(180,320).
    mean/std are for your image normalization (whatever you're using now).
    """
    H, W = out_hw
    bgr = pil_to_bgr(pil_img)
    mask = build_cross_soft_mask_bgr(bgr, out_size=(W, H))  # (H,W) float32 [0,1]

    # resize image to match
    bgr_rs = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr_rs, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # (H,W,3)

    # apply mask (broadcast over channels)
    rgb *= mask[:, :, None]

    # to torch (3,H,W)
    x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

    # normalize
    mean_t = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
    x = (x - mean_t) / std_t
    return x


def debug_overlay(
    pil_img: Image.Image,
    out_path: str,
    out_hw: tuple[int, int] = (180, 320),
) -> None:
    """Writes an overlay image so you can sanity-check what the mask is selecting."""
    H, W = out_hw
    bgr = pil_to_bgr(pil_img)
    mask = build_cross_soft_mask_bgr(bgr, out_size=(W, H))
    img = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)

    heat = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.7, heat, 0.3, 0.0)
    cv2.imwrite(out_path, overlay)
