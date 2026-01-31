# cross_roi.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2
import torch
from PIL import Image


@dataclass
class RoiResult:
    mask_hw: torch.Tensor          # [1,H,W] float32 in {bg_weight..1} (soft)
    quad_xy: Optional[np.ndarray]  # [4,2] float32 (TL,TR,BR,BL) in resized coords (bbox)
    score: float                   # higher = more confident


def _bbox_to_quad(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Convert bbox corners into quad TL,TR,BR,BL."""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def _soft_weight_from_bbox(
    H: int,
    W: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    bg_weight: float,
    dilate_px: int,
    blur_px: int,
) -> torch.Tensor:
    hard = np.zeros((H, W), dtype=np.uint8)
    x0 = int(np.clip(x0, 0, W - 1))
    x1 = int(np.clip(x1, 0, W - 1))
    y0 = int(np.clip(y0, 0, H - 1))
    y1 = int(np.clip(y1, 0, H - 1))
    if x1 <= x0 or y1 <= y0:
        # invalid box: return ones
        return torch.ones((1, H, W), dtype=torch.float32)

    hard[y0:y1 + 1, x0:x1 + 1] = 255

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        hard = cv2.dilate(hard, k, iterations=1)

    soft = hard.astype(np.float32) / 255.0

    if blur_px and blur_px > 1:
        if blur_px % 2 == 0:
            blur_px += 1
        soft = cv2.GaussianBlur(soft, (blur_px, blur_px), 0)

    # map [0..1] -> [bg_weight..1]
    soft = bg_weight + (1.0 - bg_weight) * soft
    return torch.from_numpy(soft).unsqueeze(0).float()


def _detect_white_stroke_cluster_bbox(
    bgr: np.ndarray,
) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    """
    Detect a cluster of white "stroke-like" components (the crosses) and return a bbox (x0,y0,x1,y1).

    Robust to warm lighting because it works in grayscale with contrast enhancement.
    """

    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Contrast normalize for lighting variation (warm cast, shadows)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    # Slight blur to reduce sensor noise / salt-pepper
    g = cv2.GaussianBlur(g, (5, 5), 0)

    # Threshold for bright/white strokes.
    # Using Otsu on many scenes works well; if the scene is very bright, Otsu can fail,
    # but your assumption is: crosses are the brightest, so this is usually fine.
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean speckles and join fragmented strokes slightly
    bw = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )
    bw = cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )

    # Connected components so we can filter by geometry (stroke-like)
    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)

    # Collect candidate components
    comps = []
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]

        # Basic area filtering: remove tiny noise and huge bright regions
        if area < 20:
            continue
        if area > 0.02 * H * W:
            continue

        # Aspect ratio: strokes can be vertical or horizontal but not insanely long
        ar = ww / (hh + 1e-6)
        if ar < 0.15 or ar > 6.5:
            continue

        # Stroke-likeness via compactness: perimeter^2 / area
        comp_mask = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = cnts[0]
        per = float(cv2.arcLength(c, True))
        a = float(cv2.contourArea(c))
        if a < 1e-6:
            continue
        note = (per * per) / (4.0 * np.pi * a + 1e-6)

        # blobs ~1-2, strokes bigger; clamp to reasonable range
        if note < 2.0 or note > 120.0:
            continue

        comps.append((i, x, y, ww, hh, area, note))

    if len(comps) < 2:
        return None, -1.0

    # Build a bbox union of all candidates (or you could cluster, but usually not needed here)
    xs0 = [c[1] for c in comps]
    ys0 = [c[2] for c in comps]
    xs1 = [c[1] + c[3] for c in comps]
    ys1 = [c[2] + c[4] for c in comps]

    x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)

    # Expand bbox margin a bit (cross cluster usually needs some padding)
    # Use image-relative padding so it scales across resolutions.
    pad = int(0.04 * max(H, W))  # ~4% of max dimension
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)

    # Confidence score:
    # - More components is better
    # - More total bright area is better
    # - Tighter bbox (relative to image) is better
    n = len(comps)
    total_area = float(sum(c[5] for c in comps))
    bbox_area = float((x1 - x0 + 1) * (y1 - y0 + 1))
    tightness = float(total_area / (bbox_area + 1e-6))  # higher when candidates fill bbox

    # Penalize bboxes that cover too much of the image
    frac = bbox_area / (H * W + 1e-6)
    size_pen = float(np.clip(1.0 - frac / 0.40, 0.0, 1.0))  # ok up to ~40% image

    score = (n * 1500.0) + (total_area * 0.5) + (tightness * 3000.0) + (size_pen * 2000.0)
    return (x0, y0, x1, y1), float(score)


def build_cross_weight_mask(
    pil_rgb: Image.Image,
    out_hw: Tuple[int, int],
    bg_weight: float = 0.20,
    dilate_px: int = 10,
    blur_px: int = 11,
) -> RoiResult:
    """
    Returns a soft mask [1,H,W] in [bg_weight..1].
    If detection fails, returns all-ones (no masking).

    This version detects the white cross strokes directly (lighting-robust).
    """
    H_out, W_out = out_hw

    # Resize to model input size
    pil_rs = pil_rgb.resize((W_out, H_out), Image.BILINEAR)
    rgb = np.array(pil_rs, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    bbox, score = _detect_white_stroke_cluster_bbox(bgr)

    if bbox is None:
        mask = torch.ones((1, H_out, W_out), dtype=torch.float32)
        return RoiResult(mask_hw=mask, quad_xy=None, score=-1.0)

    x0, y0, x1, y1 = bbox
    quad = _bbox_to_quad(x0, y0, x1, y1)
    mask = _soft_weight_from_bbox(
        H_out, W_out, x0, y0, x1, y1,
        bg_weight=bg_weight,
        dilate_px=dilate_px,
        blur_px=blur_px,
    )
    return RoiResult(mask_hw=mask, quad_xy=quad, score=score)


def apply_cross_roi_weight(x_chw: torch.Tensor, w_1hw: torch.Tensor) -> torch.Tensor:
    """
    Multiply an image tensor [3,H,W] or [6,H,W] by weight map [1,H,W].
    """
    if w_1hw.dtype != x_chw.dtype:
        w_1hw = w_1hw.to(dtype=x_chw.dtype)
    return x_chw * w_1hw


def preprocess_with_cross_mask(
    pil_rgb: Image.Image,
    out_hw: Tuple[int, int],
    x_tensor: Optional[torch.Tensor] = None,
    bg_weight: float = 0.20,
) -> Tuple[torch.Tensor, RoiResult]:
    roi = build_cross_weight_mask(pil_rgb, out_hw=out_hw, bg_weight=bg_weight)
    if x_tensor is None:
        return roi.mask_hw, roi
    x2 = apply_cross_roi_weight(x_tensor, roi.mask_hw)
    return x2, roi


def debug_overlay(img_pil: Image.Image, out_hw=(180, 320), bg_weight: float = 0.20) -> Image.Image:
    """
    Debug overlay: red = ROI, green box = detected bbox (quad).
    Uses raw mask thresholding for visualization (not weight thresholding).
    """
    img_rgb = np.array(img_pil.convert("RGB"))
    H, W = out_hw

    roi = build_cross_weight_mask(img_pil, out_hw=out_hw, bg_weight=bg_weight)
    img_rs = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)

    m = roi.mask_hw.squeeze(0).numpy()
    raw = (m - bg_weight) / (1.0 - bg_weight + 1e-6)
    raw = np.clip(raw, 0.0, 1.0)
    m_bin = (raw > 0.5).astype(np.uint8)

    overlay = img_rs.copy()
    overlay[m_bin > 0] = (255, 0, 0)
    out = cv2.addWeighted(img_rs, 0.80, overlay, 0.20, 0)

    if roi.quad_xy is not None:
        q = roi.quad_xy.astype(np.int32)
        cv2.polylines(out, [q], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(
            out,
            f"score={roi.score:.0f}",
            (q[0][0], max(0, q[0][1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            out,
            "ROI FAIL",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return Image.fromarray(out)
