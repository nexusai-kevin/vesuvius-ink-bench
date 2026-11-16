"""3D Hann-window overlap blending, smooth Z-Hann aggregation, and metrics."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


def compute_hann_window_3d(
    tile_shape: Tuple[int, int, int],
    *,
    min_weight: float = 1e-6,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Construct a strictly separable 3D raised-Hann window of shape ``(D, H, W)``.

    Each axis is raised by ``min_weight`` *before* taking the outer product so
    the Hann taper along the remaining axes is preserved even at ``z=0`` or
    ``z=D-1``, preventing flat unblended tile footprints at volume borders.
    """
    d, h, w = tile_shape
    wz = torch.hann_window(d, periodic=False, dtype=dtype, device=device)
    wy = torch.hann_window(h, periodic=False, dtype=dtype, device=device)
    wx = torch.hann_window(w, periodic=False, dtype=dtype, device=device)

    if min_weight > 0.0:
        wz = min_weight + (1.0 - min_weight) * wz
        wy = min_weight + (1.0 - min_weight) * wy
        wx = min_weight + (1.0 - min_weight) * wx

    return wz[:, None, None] * wy[None, :, None] * wx[None, None, :]


def project_surface_z_hann(
    volume_3d: np.ndarray,
    *,
    min_weight: float = 1e-3,
) -> np.ndarray:
    """Collapse a 3D depth-slab prediction volume ``(D, H, W)`` using Z-Hann weights.

    Replaces hard ``max(axis=0)`` across Z-tiles with smooth Z-Hann depth
    aggregation so no Z-slab boundary discontinuities appear in 2D exports.
    """
    depth = volume_3d.shape[0]
    if depth == 1:
        return volume_3d[0].astype(np.float32, copy=False)
    wz = np.hanning(depth).astype(np.float32)
    if min_weight > 0.0:
        wz = min_weight + (1.0 - min_weight) * wz
    wz = wz / np.maximum(wz.sum(), 1e-8)
    return np.tensordot(wz, volume_3d, axes=(0, 0)).astype(np.float32, copy=False)


def compute_tile_grid(
    volume_shape: Tuple[int, int, int],
    tile_shape: Tuple[int, int, int],
    stride: Tuple[int, int, int],
) -> list[Tuple[int, int, int]]:
    """Compute tile start coordinates ``(z, y, x)`` covering ``volume_shape``."""
    vd, vh, vw = volume_shape
    td, th, tw = tile_shape
    sd, sh, sw = stride

    def _axis_starts(v_len: int, t_len: int, step: int) -> list[int]:
        if v_len <= t_len:
            return [0]
        starts = list(range(0, v_len - t_len + 1, step))
        if starts[-1] != v_len - t_len:
            starts.append(v_len - t_len)
        return starts

    zs = _axis_starts(vd, td, sd)
    ys = _axis_starts(vh, th, sh)
    xs = _axis_starts(vw, tw, sw)
    return [(z, y, x) for z in zs for y in ys for x in xs]


def compute_seam_metrics(
    blended_2d: np.ndarray,
    unblended_2d: np.ndarray,
    tile_shape: Tuple[int, int],
    stride: Tuple[int, int],
) -> Dict[str, float]:
    """Quantify tile-seam artifacts on 2D ink surface projections."""
    h, w = blended_2d.shape
    _, _ = tile_shape
    sy, sx = stride

    seam_ys = [y for y in range(sy, h - 1, sy)]
    seam_xs = [x for x in range(sx, w - 1, sx)]

    def _boundary_stats(img: np.ndarray) -> Tuple[float, float, float]:
        dy = np.abs(np.diff(img, axis=0))
        dx = np.abs(np.diff(img, axis=1))

        y_mask = np.zeros(h - 1, dtype=bool)
        for y in seam_ys:
            if 0 <= y - 1 < h - 1:
                y_mask[y - 1] = True

        x_mask = np.zeros(w - 1, dtype=bool)
        for x in seam_xs:
            if 0 <= x - 1 < w - 1:
                x_mask[x - 1] = True

        seam_grads = []
        interior_grads = []
        if np.any(y_mask):
            seam_grads.append(dy[y_mask, :].ravel())
            interior_grads.append(dy[~y_mask, :].ravel())
        if np.any(x_mask):
            seam_grads.append(dx[:, x_mask].ravel())
            interior_grads.append(dx[:, ~x_mask].ravel())

        if not seam_grads:
            return 0.0, 0.0, 1.0

        seam_mean = float(np.mean(np.concatenate(seam_grads)))
        interior_mean = float(np.mean(np.concatenate(interior_grads)))
        ratio = seam_mean / max(interior_mean, 1e-8)
        return seam_mean, interior_mean, ratio

    u_seam, u_int, u_ratio = _boundary_stats(unblended_2d)
    b_seam, b_int, b_ratio = _boundary_stats(blended_2d)

    reduction_pct = (
        100.0 * (u_seam - b_seam) / max(u_seam, 1e-8) if u_seam > 0 else 0.0
    )
    return {
        "unblended_seam_gradient": round(u_seam, 6),
        "unblended_interior_gradient": round(u_int, 6),
        "unblended_seam_to_interior_ratio": round(u_ratio, 4),
        "hann_seam_gradient": round(b_seam, 6),
        "hann_interior_gradient": round(b_int, 6),
        "hann_seam_to_interior_ratio": round(b_ratio, 4),
        "seam_discontinuity_reduction_pct": round(reduction_pct, 2),
    }


def compute_ink_accuracy_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute F0.5, Dice/F1, IoU, and ROC-AUC against ground-truth ink labels."""
    if mask is not None:
        valid = mask > 0
        yt = (y_true[valid] > 0).astype(np.uint8)
        yp = y_prob[valid].astype(np.float64)
    else:
        yt = (y_true.ravel() > 0).astype(np.uint8)
        yp = y_prob.ravel().astype(np.float64)

    def _scores_at(th: float) -> Tuple[float, float, float]:
        pred_bin = yp >= th
        tp = float(np.logical_and(yt == 1, pred_bin).sum())
        fp = float(np.logical_and(yt == 0, pred_bin).sum())
        fn = float(np.logical_and(yt == 1, ~pred_bin).sum())
        f1 = (2.0 * tp) / max(2.0 * tp + fp + fn, 1e-9)
        f05 = (1.25 * tp) / max(1.25 * tp + 0.25 * fn + fp, 1e-9)
        iou = tp / max(tp + fp + fn, 1e-9)
        return f05, f1, iou

    f05, f1, iou = _scores_at(threshold)

    best_f05 = f05
    best_th = threshold
    for th in np.linspace(0.1, 0.9, 17):
        cand_f05, _, _ = _scores_at(float(th))
        if cand_f05 > best_f05:
            best_f05 = cand_f05
            best_th = float(th)
    _, best_th_f1, best_th_iou = _scores_at(best_th)

    n_pos = float(yt.sum())
    n_neg = float(len(yt) - n_pos)
    if n_pos > 0 and n_neg > 0:
        order = np.argsort(yp, kind="mergesort")
        y_sorted = yt[order]
        ranks = np.arange(1, len(y_sorted) + 1, dtype=np.float64)
        roc_auc = float(
            (ranks[y_sorted == 1].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
        )
    else:
        roc_auc = 0.0

    return {
        "threshold": round(threshold, 4),
        "f0_5": round(f05, 4),
        "dice_f1": round(f1, 4),
        "iou": round(iou, 4),
        "roc_auc": round(roc_auc, 4),
        "best_f0_5_threshold": round(best_th, 4),
        "best_f0_5": round(best_f05, 4),
        "dice_f1_at_best_f0_5": round(best_th_f1, 4),
        "iou_at_best_f0_5": round(best_th_iou, 4),
    }
