# cross_roi.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

# OpenCV is best here (fast, robust)
import cv2
from PIL import Image


@dataclass
class CrossRoiConfig:
    out_hw: Tuple[int, int] = (180, 320)  # (H,W)
    # soft attention: out = img * (base + strength*mask)
    base: float = 0.20
    strength: float = 0.80

    # card detection constraints
    card_area_min_frac: float = 0.01
    card_area_max_frac: float = 0.50
    card_aspect_min: float = 1.2
    card_aspect_max: float = 3.0

    # edge detection for card
    canny_lo: int = 40
    canny_hi: int = 120

    # cross threshold (within card ROI)
    cross_v_percentile: float = 92.0
    cross_v_clip_min: int = 80
    cross_v_clip_max: int = 220

    # component filtering (within card)
    comp_area_min_frac_of_full: float = 0.0003
    comp_area_max_frac_of_roi: float = 0.15

    # mask thickening
    dilate_kernel: int = 5


def _find_card_bbox(rgb: np.ndarray, cfg: CrossRoiConfig) -> Tuple[int, int, int, int]:
    """
    Find a near-center quadrilateral (the dark-blue card) via edges/contours.
    Returns (x,y,w,h) in resized image coordinates.
    """
    H, W = cfg.out_hw
    img = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, cfg.canny_lo, cfg.canny_hi)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0.0
    full_area = float(H * W)

    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < cfg.card_area_min_frac * full_area or area > cfg.card_area_max_frac * full_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h + 1e-6)
        if ar < cfg.card_aspect_min or ar > cfg.card_aspect_max:
            continue

        cx = x + 0.5 * w
        cy = y + 0.5 * h
        center_dist = ((cx - W / 2) ** 2 + (cy - H / 2) ** 2) ** 0.5

        # prefer large + near center
        score = area / (1.0 + center_dist)
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        # fallback: central box
        w = int(0.55 * W)
        h = int(0.35 * H)
        x = (W - w) // 2
        y = (H - h) // 2
        best = (x, y, w, h)

    return best


def _cross_mask_uint8(rgb: np.ndarray, cfg: CrossRoiConfig) -> np.ndarray:
    """
    Returns uint8 mask (H,W) values {0,255} in resized coordinates.
    """
    H, W = cfg.out_hw
    img = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

    x, y, w, h = _find_card_bbox(rgb, cfg)
    roi = img[y : y + h, x : x + w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    _, s, v = cv2.split(hsv)

    v_blur = cv2.GaussianBlur(v, (5, 5), 0)
    thr_v = int(np.clip(np.percentile(v_blur, cfg.cross_v_percentile), cfg.cross_v_clip_min, cfg.cross_v_clip_max))

    # inside the card, don't rely on saturation (lighting changes)
    mask = (v_blur >= thr_v).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    # filter connected components to remove huge blobs and tiny noise
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)

    full_area = float(H * W)
    area_min = cfg.comp_area_min_frac_of_full * full_area
    area_max = cfg.comp_area_max_frac_of_roi * float(w * h)

    for i in range(1, num):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area_min <= area <= area_max:
            keep[lab == i] = 255

    d = cfg.dilate_kernel
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)), iterations=1)

    full = np.zeros((H, W), dtype=np.uint8)
    full[y : y + h, x : x + w] = keep
    return full


def preprocess_with_cross_mask(
    img_pil: Image.Image,
    out_hw: Tuple[int, int] = (180, 320),
    base: float = 0.20,
    strength: float = 0.80,
    return_mask: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """
    Drop-in preprocessor:
      - resizes
      - finds cross ROI mask
      - applies soft attention to RGB
      - returns torch float tensor (C,H,W) in [0,1]

    Output is compatible with your existing model (still 3ch per frame).
    """
    cfg = CrossRoiConfig(out_hw=out_hw, base=base, strength=strength)

    img_rgb = np.array(img_pil.convert("RGB"))
    H, W = out_hw

    # compute mask (uint8 0/255) and normalize to 0..1
    m_u8 = _cross_mask_uint8(img_rgb, cfg)
    m = (m_u8.astype(np.float32) / 255.0)  # (H,W)

    # resize image to match
    img_rs = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0  # (H,W,3)

    # soft attention
    alpha = np.clip(cfg.base + cfg.strength * m, 0.0, 1.0).astype(np.float32)  # (H,W)
    img_att = img_rs * alpha[..., None]  # (H,W,3)

    x = torch.from_numpy(img_att).permute(2, 0, 1).contiguous()  # (3,H,W)

    if return_mask:
        m_t = torch.from_numpy(m).unsqueeze(0).contiguous()  # (1,H,W)
        return x, m_t
    return x


def debug_overlay(img_pil: Image.Image, out_hw=(180, 320)) -> Image.Image:
    """
    Handy: returns an RGB PIL image with mask overlayed in red.
    """
    img_rgb = np.array(img_pil.convert("RGB"))
    cfg = CrossRoiConfig(out_hw=out_hw)
    H, W = out_hw

    m_u8 = _cross_mask_uint8(img_rgb, cfg)
    img_rs = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)

    overlay = img_rs.copy()
    overlay[m_u8 > 0] = (255, 0, 0)  # red highlight
    out = cv2.addWeighted(img_rs, 0.80, overlay, 0.20, 0)
    return Image.fromarray(out)
