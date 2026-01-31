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
    quad_xy: Optional[np.ndarray]  # [4,2] float32 (TL,TR,BR,BL) in resized coords
    score: float                   # higher = more confident


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points TL,TR,BR,BL."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)[:, 0]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _poly_mask(h: int, w: int, quad: np.ndarray) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(m, quad.astype(np.int32), 255)
    return m


def _mean_hsv_in_poly(hsv: np.ndarray, quad: np.ndarray) -> Tuple[float, float, float]:
    """Return mean (H,S,V) inside quad polygon."""
    h, w = hsv.shape[:2]
    pm = _poly_mask(h, w, quad)
    ys, xs = np.where(pm > 0)
    if len(xs) < 50:
        return 0.0, 0.0, 0.0
    roi = hsv[ys, xs, :]
    Hm = float(np.mean(roi[:, 0]))
    Sm = float(np.mean(roi[:, 1]))
    Vm = float(np.mean(roi[:, 2]))
    return Hm, Sm, Vm


def _count_white_strokes_in_warp(warp_bgr: np.ndarray) -> int:
    """
    Heuristic validation: in the plate warp, white cross strokes produce
    multiple thin/high-perimeter components. Count candidates.
    """
    gray = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, -5
    )

    bw = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )

    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    H, W = gray.shape[:2]
    cands = 0

    for i in range(1, num):
        x, y, ww, hh, aa = stats[i]

        if aa < 25 or aa > 0.15 * H * W:
            continue

        ar = ww / (hh + 1e-6)
        if ar < 0.25 or ar > 4.0:
            continue

        comp = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        c = cnts[0]
        area = float(cv2.contourArea(c))
        if area <= 1e-6:
            continue
        per = float(cv2.arcLength(c, True))

        compact = (per * per) / (4.0 * np.pi * area + 1e-6)
        if compact < 2.0 or compact > 60.0:
            continue

        cands += 1

    return cands


def _detect_plate_quad(resized_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """
    Find the dark blue/black rectangle by:
      1) color mask -> candidates
      2) rectangular contour check
      3) HSV stats inside quad (dark + bluish)
      4) center prior (plate is usually central)
      5) cross-stroke validation in perspective warp

    Returns (quad_xy, score). quad ordered TL,TR,BR,BL.
    """
    h, w = resized_bgr.shape[:2]
    hsv = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
    Hc = hsv[:, :, 0]
    Sc = hsv[:, :, 1]
    Vc = hsv[:, :, 2]

    blueish = ((Hc >= 75) & (Hc <= 155) & (Sc > 25) & (Vc < 245))
    dark_neutral = ((Vc < 60) & (Sc < 90))
    mask = (blueish | dark_neutral).astype(np.uint8) * 255

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    )

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_score = -1.0

    img_cx, img_cy = 0.5 * w, 0.5 * h
    img_diag = float(np.sqrt(w * w + h * h))

    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < 0.002 * h * w:
            continue

        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), _ang = rect
        if rw < 20 or rh < 20:
            continue

        ar = max(rw, rh) / (min(rw, rh) + 1e-6)
        if ar < 1.15 or ar > 6.0:
            continue

        box = cv2.boxPoints(rect).astype(np.float32)
        quad = _order_quad(box)

        Hm, Sm, Vm = _mean_hsv_in_poly(hsv, quad)
        if Vm > 165:
            continue

        is_bluish = (75 <= Hm <= 155 and Sm >= 18)
        is_blackish = (Sm < 18 and Vm < 95)
        if not (is_bluish or is_blackish):
            continue

        rect_area = float(rw * rh)
        fill = area / (rect_area + 1e-6)
        if fill < 0.35:
            continue

        out_w = int(max(rw, rh))
        out_h = int(min(rw, rh))
        if out_h > out_w:
            out_w, out_h = out_h, out_w
        out_w = max(out_w, 80)
        out_h = max(out_h, 50)

        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(quad, dst)
        warp = cv2.warpPerspective(resized_bgr, M, (out_w, out_h))

        stroke_count = _count_white_strokes_in_warp(warp)
        if stroke_count < 3:
            continue

        d_center = float(np.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2))
        center_score = 1.0 - (d_center / (0.55 * img_diag + 1e-6))
        center_score = float(np.clip(center_score, 0.0, 1.0))

        score = (
            stroke_count * 2000.0
            + area * fill * 0.5
            + center_score * 5000.0
            + (165.0 - Vm) * 20.0
        )

        if score > best_score:
            best_score = score
            best_quad = quad

    return best_quad, best_score


def build_cross_weight_mask(
    pil_rgb: Image.Image,
    out_hw: Tuple[int, int],
    bg_weight: float = 0.15,
    dilate_px: int = 6,
    blur_px: int = 11,
) -> RoiResult:
    """
    Returns a soft mask [1,H,W] in [bg_weight..1].
    If detection fails, returns all-ones (no masking).
    """
    H_out, W_out = out_hw

    pil_rs = pil_rgb.resize((W_out, H_out), Image.BILINEAR)
    rgb = np.array(pil_rs, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    quad, score = _detect_plate_quad(bgr)

    if quad is None:
        mask = torch.ones((1, H_out, W_out), dtype=torch.float32)
        return RoiResult(mask_hw=mask, quad_xy=None, score=-1.0)

    hard = np.zeros((H_out, W_out), dtype=np.uint8)
    cv2.fillConvexPoly(hard, quad.astype(np.int32), 255)

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        hard = cv2.dilate(hard, k, iterations=1)

    soft = hard.astype(np.float32) / 255.0

    if blur_px and blur_px > 1:
        if blur_px % 2 == 0:
            blur_px += 1
        soft = cv2.GaussianBlur(soft, (blur_px, blur_px), 0)

    soft = bg_weight + (1.0 - bg_weight) * soft
    soft_t = torch.from_numpy(soft).unsqueeze(0).float()

    return RoiResult(mask_hw=soft_t, quad_xy=quad, score=float(score))


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
    bg_weight: float = 0.15,
) -> Tuple[torch.Tensor, RoiResult]:
    roi = build_cross_weight_mask(pil_rgb, out_hw=out_hw, bg_weight=bg_weight)
    if x_tensor is None:
        return roi.mask_hw, roi
    x2 = apply_cross_roi_weight(x_tensor, roi.mask_hw)
    return x2, roi


def debug_overlay(img_pil: Image.Image, out_hw=(180, 320), bg_weight: float = 0.15) -> Image.Image:
    """
    Debug overlay: red = ROI, green polyline = detected quad.

    IMPORTANT: mask values are weights in [bg_weight..1], so we convert back to raw [0..1]
    before thresholding for a meaningful overlay.
    """
    img_rgb = np.array(img_pil.convert("RGB"))
    H, W = out_hw

    roi = build_cross_weight_mask(img_pil, out_hw=out_hw, bg_weight=bg_weight)
    img_rs = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)

    m = roi.mask_hw.squeeze(0).numpy()
    raw = (m - bg_weight) / (1.0 - bg_weight + 1e-6)  # back to [0..1]
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

    return Image.fromarray(out)
