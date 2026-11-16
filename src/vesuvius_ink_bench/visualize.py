"""2D ink surface & seam-blending PNG visualization exports."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def export_ink_visualizations(
    ct_volume: np.ndarray,
    blended_mip_2d: np.ndarray,
    blended_mean_2d: np.ndarray,
    unblended_mip_2d: np.ndarray,
    output_dir: Path,
    stride_yx: Tuple[int, int],
) -> Dict[str, str]:
    """Export predicted 2D ink surface PNGs and seam-blending comparisons."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ct_mid = ct_volume[ct_volume.shape[0] // 2].astype(np.float32) / 255.0

    # 1. Predicted 2D ink surface visualization (3-panel)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=160)
    axes[0].imshow(ct_mid, cmap="gray", vmin=0.0, vmax=max(0.4, float(np.percentile(ct_mid, 99.5))))
    axes[0].set_title("Herculaneum CT Mid-Slice (Z/2)")
    axes[0].axis("off")

    im1 = axes[1].imshow(blended_mip_2d, cmap="magma")
    axes[1].set_title("Predicted 2D Ink Surface (3D Hann MIP)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(ct_mid, cmap="gray")
    axes[2].imshow(blended_mean_2d, cmap="inferno", alpha=0.45)
    axes[2].set_title("Ink Surface Overlay on Papyrus")
    axes[2].axis("off")

    fig.tight_layout()
    ink_path = output_dir / "ink_surface_prediction.png"
    fig.savefig(ink_path, bbox_inches="tight")
    plt.close(fig)

    # 2. Seam-blending comparison (Unblended vs 3D Hann-Windowed)
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), dpi=160)
    vmin = float(min(unblended_mip_2d.min(), blended_mip_2d.min()))
    vmax = float(max(unblended_mip_2d.max(), blended_mip_2d.max()))

    axes2[0].imshow(unblended_mip_2d, cmap="magma", vmin=vmin, vmax=vmax)
    sy, sx = stride_yx
    for y in range(sy, unblended_mip_2d.shape[0], sy):
        axes2[0].axhline(y - 0.5, color="cyan", linestyle="--", linewidth=0.5, alpha=0.55)
    for x in range(sx, unblended_mip_2d.shape[1], sx):
        axes2[0].axvline(x - 0.5, color="cyan", linestyle="--", linewidth=0.5, alpha=0.55)
    axes2[0].set_title("Unblended Tiling (Seam Grid Shown)")
    axes2[0].axis("off")

    axes2[1].imshow(blended_mip_2d, cmap="magma", vmin=vmin, vmax=vmax)
    axes2[1].set_title("3D Hann-Window Overlap Blending")
    axes2[1].axis("off")

    diff = np.abs(unblended_mip_2d - blended_mip_2d)
    im_diff = axes2[2].imshow(diff, cmap="viridis")
    axes2[2].set_title("|Unblended - Hann| Seam Suppression")
    axes2[2].axis("off")
    fig2.colorbar(im_diff, ax=axes2[2], fraction=0.046, pad=0.04)

    fig2.tight_layout()
    seam_path = output_dir / "seam_blending_comparison.png"
    fig2.savefig(seam_path, bbox_inches="tight")
    plt.close(fig2)

    return {
        "ink_surface_prediction_png": str(ink_path),
        "seam_blending_comparison_png": str(seam_path),
    }
