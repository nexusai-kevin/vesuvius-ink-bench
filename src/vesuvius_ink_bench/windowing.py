"""3D Hann-window overlap blending and quantitative tile-seam evaluation."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


def compute_hann_window_3d(
    tile_shape: Tuple[int, int, int],
    *,
    min_weight: float = 1e-3,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Construct a separable 3D raised-Hann window of shape ``(D, H, W)``.

    A small positive floor ``min_weight`` prevents zero-division at global volume
    boundaries where no neighboring tile overlaps, while preserving smooth C1
    tapering in all interior overlap regions.
    """
    d, h, w = tile_shape
    wz = torch.hann_window(d, periodic=False, dtype=dtype, device=device)
    wy = torch.hann_window(h, periodic=False, dtype=dtype, device=device)
    wx = torch.hann_window(w, periodic=False, dtype=dtype, device=device)

    win = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
    if min_weight > 0.0:
        win = min_weight + (1.0 - min_weight) * win
    return win


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
    """Quantify tile-seam artifacts on 2D ink surface projections.

    Compares the gradient jump across tile boundary lines against interior
    pixels for both unblended (hard-cut/overwrite) and 3D Hann-blended outputs.
    """
    h, w = blended_2d.shape
    _, _ = tile_shape
    sy, sx = stride

    seam_ys = [y for y in range(sy, h - 1, sy)]
    seam_xs = [x for x in range(sx, w - 1, sx)]

    def _boundary_stats(img: np.ndarray) -> Tuple[float, float, float]:
        dy = np.abs(np.diff(img, axis=0))  # (h-1, w)
        dx = np.abs(np.diff(img, axis=1))  # (h, w-1)

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
