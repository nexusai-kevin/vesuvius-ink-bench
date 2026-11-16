"""2D ink surface, seam-blending, and ground-truth PNG visualization exports."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def export_ink_visualizations(
    ct_volume: np.ndarray,
    blended_surface_2d: np.ndarray,
    blended_mean_2d: np.ndarray,
    unblended_surface_2d: np.ndarray,
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

    im1 = axes[1].imshow(blended_surface_2d, cmap="magma", vmin=0.0, vmax=max(0.5, float(blended_surface_2d.max())))
    axes[1].set_title("Predicted 2D Ink Surface (3D + Z-Hann)")
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
    vmin = float(min(unblended_surface_2d.min(), blended_surface_2d.min()))
    vmax = float(max(unblended_surface_2d.max(), blended_surface_2d.max()))

    axes2[0].imshow(unblended_surface_2d, cmap="magma", vmin=vmin, vmax=vmax)
    sy, sx = stride_yx
    for y in range(sy, unblended_surface_2d.shape[0], sy):
        axes2[0].axhline(y - 0.5, color="cyan", linestyle="--", linewidth=0.5, alpha=0.55)
    for x in range(sx, unblended_surface_2d.shape[1], sx):
        axes2[0].axvline(x - 0.5, color="cyan", linestyle="--", linewidth=0.5, alpha=0.55)
    axes2[0].set_title("Unblended Tiling (Seam Grid Shown)")
    axes2[0].axis("off")

    axes2[1].imshow(blended_surface_2d, cmap="magma", vmin=vmin, vmax=vmax)
    axes2[1].set_title("3D Hann-Window Overlap Blending")
    axes2[1].axis("off")

    diff = np.abs(unblended_surface_2d - blended_surface_2d)
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


def export_ground_truth_validation(
    ct_volume: np.ndarray,
    gt_inklabels: np.ndarray,
    pred_surface_2d: np.ndarray,
    metrics: Dict[str, float],
    output_path: Path,
) -> str:
    """Export side-by-side ground-truth inklabels vs. vesuvius-infer prediction PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ct_mid = ct_volume[ct_volume.shape[0] // 2].astype(np.float32) / 255.0
    gt_bin = (gt_inklabels > 0).astype(np.uint8)
    pred_bin = (pred_surface_2d >= 0.5).astype(np.uint8)

    # RGB error map: TP=green, FP=red, FN=blue
    overlay = np.zeros((*gt_bin.shape, 3), dtype=np.float32)
    overlay[(gt_bin == 1) & (pred_bin == 1)] = [0.15, 0.85, 0.35]  # TP
    overlay[(gt_bin == 0) & (pred_bin == 1)] = [0.95, 0.25, 0.20]  # FP
    overlay[(gt_bin == 1) & (pred_bin == 0)] = [0.20, 0.45, 0.95]  # FN

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), dpi=160)
    axes[0].imshow(ct_mid, cmap="gray")
    axes[0].set_title("PHercParis4 2.4µm CT Mid-Slice")
    axes[0].axis("off")

    axes[1].imshow(gt_bin, cmap="gray")
    axes[1].set_title("Ground Truth (inklabels.zarr)")
    axes[1].axis("off")

    im2 = axes[2].imshow(pred_surface_2d, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("vesuvius-infer 2D Surface Prob.")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(overlay)
    axes[3].set_title("TP (Green) / FP (Red) / FN (Blue)")
    axes[3].axis("off")

    fig.suptitle(
        f"PHercParis4 (20230702185753) Validation — "
        f"ROC-AUC: {metrics['roc_auc']:.4f} | "
        f"Dice/F1: {metrics['dice_f1']:.4f} | "
        f"IoU: {metrics['iou']:.4f} | "
        f"F0.5: {metrics['f0_5']:.4f} (best {metrics['best_f0_5']:.4f})",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)
