"""End-to-end verification, ground-truth validation, and reproducible benchmark runner."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import zarr

from vesuvius_ink_bench.config import (
    BenchmarkConfig,
    CacheConfig,
    InferenceConfig,
    ZarrSourceConfig,
)
from vesuvius_ink_bench.infer import run_inference
from vesuvius_ink_bench.model import VillaInkResNet3D
from vesuvius_ink_bench.pipeline import MixedPrecisionInkPipeline
from vesuvius_ink_bench.visualize import (
    export_ground_truth_validation,
    export_ink_visualizations,
)
from vesuvius_ink_bench.windowing import compute_ink_accuracy_metrics
from vesuvius_ink_bench.zarr_loader import StreamingZarrSubvolumeLoader, open_zarr_group


GT_SURFACE_URI = (
    "s3://vesuvius-challenge-open-data/"
    "PHercParis4/segments/20230702185753/surface-volumes/"
    "2.4um-0.22m-78keV-volume-20260411134726.zarr"
)
GT_INKLABELS_URI = (
    "s3://vesuvius-challenge-open-data/"
    "PHercParis4/segments/20230702185753/ink-labels/"
    "2.4um-volume-20260411134726/20260918/inklabels.zarr"
)
GT_SUPERVISION_URI = (
    "s3://vesuvius-challenge-open-data/"
    "PHercParis4/segments/20230702185753/ink-labels/"
    "2.4um-volume-20260411134726/20260918/supervision.zarr"
)
GT_ROI_ORIGIN = (0, 46080, 9728)
GT_ROI_SHAPE = (109, 512, 512)


def _candidate_origins_for_shape(
    vol_shape: Tuple[int, int, int],
    roi_shape: Tuple[int, int, int],
    preferred_origin: Tuple[int, int, int],
) -> List[Tuple[int, int, int]]:
    cz = max(0, min(vol_shape[0] - roi_shape[0], vol_shape[0] // 2 - roi_shape[0] // 2))
    cy = max(0, min(vol_shape[1] - roi_shape[1], vol_shape[1] // 2 - roi_shape[1] // 2))
    cx = max(0, min(vol_shape[2] - roi_shape[2], vol_shape[2] // 2 - roi_shape[2] // 2))
    candidates = [
        preferred_origin,
        (cz, cy, cx),
        (cz, max(0, cy - roi_shape[1]), cx),
        (cz, cy, max(0, cx - roi_shape[2])),
    ]
    seen = set()
    unique: List[Tuple[int, int, int]] = []
    for c in candidates:
        c_clamped = (
            max(0, min(vol_shape[0] - roi_shape[0], c[0])),
            max(0, min(vol_shape[1] - roi_shape[1], c[1])),
            max(0, min(vol_shape[2] - roi_shape[2], c[2])),
        )
        if c_clamped not in seen:
            seen.add(c_clamped)
            unique.append(c_clamped)
    return unique


def run_ground_truth_validation(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    hf_repo_id: str,
    batch_size: int = 8,
    amp_dtype: str = "bfloat16",
    pin_staging_buffer: bool = True,
    use_channels_last_3d: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run vesuvius-infer on the labeled PHercParis4 (2.4 um) segment and score vs inklabels."""
    _, y0, x0 = GT_ROI_ORIGIN
    _, h, w = GT_ROI_SHAPE

    with tempfile.TemporaryDirectory() as tmp_dir:
        infer_summary = run_inference(
            uri=GT_SURFACE_URI,
            level="0",
            origin=GT_ROI_ORIGIN,
            shape=GT_ROI_SHAPE,
            output_dir=Path(tmp_dir),
            batch_size=batch_size,
            amp_dtype=amp_dtype,
            checkpoint_path=checkpoint_path,
            hf_repo_id=hf_repo_id,
            pin_staging_buffer=pin_staging_buffer,
            use_channels_last_3d=use_channels_last_3d,
            seed=seed,
        )
        pred_group = zarr.open_group(infer_summary["exports"]["ome_zarr"], mode="r")
        pred_surface = np.asarray(pred_group["surface_prob_2d"][:], dtype=np.float32)

    lbl_group = open_zarr_group(GT_INKLABELS_URI, anon=True)
    sup_group = open_zarr_group(GT_SUPERVISION_URI, anon=True)
    gt_labels = np.asarray(lbl_group["0"][y0 : y0 + h, x0 : x0 + w], dtype=np.uint8)
    sup_mask = np.asarray(sup_group["0"][y0 : y0 + h, x0 : x0 + w], dtype=np.uint8)

    surf_loader = StreamingZarrSubvolumeLoader(
        ZarrSourceConfig(uri=GT_SURFACE_URI, level="0", roi_origin=GT_ROI_ORIGIN, roi_shape=GT_ROI_SHAPE),
        CacheConfig(pin_staging_buffer=pin_staging_buffer),
    )
    ct_volume = surf_loader.read_subvolume()

    acc_metrics = compute_ink_accuracy_metrics(gt_labels, pred_surface, mask=sup_mask, threshold=0.5)
    gt_png = export_ground_truth_validation(
        ct_volume=ct_volume,
        gt_inklabels=gt_labels,
        pred_surface_2d=pred_surface,
        metrics=acc_metrics,
        output_path=output_dir / "ground_truth_validation.png",
    )

    return {
        "segment": "PHercParis4/segments/20230702185753 (2.4 um)",
        "surface_uri": GT_SURFACE_URI,
        "inklabels_uri": GT_INKLABELS_URI,
        "roi_origin": list(GT_ROI_ORIGIN),
        "roi_shape": list(GT_ROI_SHAPE),
        "supervised_fraction": round(float((sup_mask > 0).mean()), 4),
        "ground_truth_ink_fraction": round(float((gt_labels > 0).mean()), 4),
        "accuracy": acc_metrics,
        "inference_metrics": {
            "gpu_seconds": infer_summary["metrics"]["gpu_seconds"],
            "throughput_tile_voxels_per_sec": infer_summary["metrics"]["throughput_tile_voxels_per_sec"],
            "peak_vram_allocated_mib": infer_summary["metrics"]["peak_vram_allocated_mib"],
            "peak_vram_reserved_mib": infer_summary["metrics"]["peak_vram_reserved_mib"],
        },
        "ground_truth_validation_png": gt_png,
    }


def run_benchmark(config: BenchmarkConfig, *, include_gt_validation: bool = True) -> Dict[str, Any]:
    """Execute streaming Zarr load, DDR5 cache verification, RTX 5080 inference, and reporting."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = StreamingZarrSubvolumeLoader(config.source, config.cache)
    zarr_meta = loader.metadata_summary()

    # 1. Cold streaming pull from AWS Open Data OME-Zarr into DDR5 RAM cache
    candidates = _candidate_origins_for_shape(
        loader.shape, config.source.roi_shape, config.source.roi_origin
    )
    t_cold0 = time.perf_counter()
    origin, volume = loader.find_nonempty_roi(
        candidates, config.source.roi_shape, min_nonzero_fraction=0.05
    )
    cold_seconds = time.perf_counter() - t_cold0
    cold_stats = loader.cache.stats.to_dict()

    # 2. Warm read from host DDR5 RAM cache
    t_warm0 = time.perf_counter()
    _ = loader.read_subvolume(origin=origin, roi_shape=config.source.roi_shape)
    warm_seconds = time.perf_counter() - t_warm0
    warm_stats = loader.cache.stats.to_dict()

    nonzero_frac = float(np.count_nonzero(volume)) / float(volume.size)
    roi_bytes = int(volume.nbytes)

    # 3. Mixed-precision PyTorch inference with pretrained ScrollPrize checkpoint & 3D Hann blending
    ckpt_path = Path(config.inference.checkpoint_path)
    if not ckpt_path.exists():
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=config.inference.hf_repo_id,
            local_dir=str(ckpt_path.parent),
        )
    model, ckpt_meta = VillaInkResNet3D.from_villa_checkpoint(ckpt_path, strict=True)
    pipeline = MixedPrecisionInkPipeline(
        model,
        config.inference,
        pin_staging_buffer=config.cache.pin_staging_buffer,
    )
    result = pipeline.run(volume, warmup_batches=config.warmup_batches)

    surf_2d = result.blended_surface_2d
    ink_prob_stats = {
        "min": round(float(surf_2d.min()), 6),
        "max": round(float(surf_2d.max()), 6),
        "mean": round(float(surf_2d.mean()), 6),
        "std": round(float(surf_2d.std()), 6),
        "p50": round(float(np.percentile(surf_2d, 50)), 6),
        "p99": round(float(np.percentile(surf_2d, 99)), 6),
    }

    # 4. Export 2D ink surface PNGs
    png_paths = export_ink_visualizations(
        ct_volume=volume,
        blended_surface_2d=result.blended_surface_2d,
        blended_mean_2d=result.blended_mean_2d,
        unblended_surface_2d=result.unblended_surface_2d,
        output_dir=output_dir,
        stride_yx=(config.inference.stride[1], config.inference.stride[2]),
    )

    gt_validation: Dict[str, Any] | None = None
    if include_gt_validation:
        gt_validation = run_ground_truth_validation(
            output_dir=output_dir,
            checkpoint_path=ckpt_path,
            hf_repo_id=config.inference.hf_repo_id,
            batch_size=config.inference.batch_size,
            amp_dtype=config.inference.amp_dtype,
            pin_staging_buffer=config.cache.pin_staging_buffer,
            use_channels_last_3d=config.inference.use_channels_last_3d,
            seed=config.inference.seed,
        )
        png_paths["ground_truth_validation_png"] = gt_validation["ground_truth_validation_png"]

    # 5. Assemble reproducible JSON & Markdown reports
    report: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "zarr_source": {
            **zarr_meta,
            "selected_roi_origin": list(origin),
            "selected_roi_shape": list(config.source.roi_shape),
            "roi_nonzero_fraction": round(nonzero_frac, 4),
            "roi_bytes": roi_bytes,
            "input_intensity_scaling": "[0, 1] (uint8 / 255.0)",
        },
        "ddr5_cache": {
            "budget_gib": round(config.cache.max_bytes / (1024**3), 2),
            "pin_staging_buffer": config.cache.pin_staging_buffer,
            "cold_stream_seconds": round(cold_seconds, 4),
            "warm_ddr5_read_seconds": round(warm_seconds, 6),
            "warm_ddr5_bandwidth_gb_s": round((roi_bytes / 1e9) / max(warm_seconds, 1e-9), 2),
            "after_cold_stats": cold_stats,
            "after_warm_stats": warm_stats,
        },
        "pretrained_checkpoint": {
            "hf_repo_id": config.inference.hf_repo_id,
            "backbone": "ResNet3D-152 + 3D FPN Decoder (i3d)",
            "weights_only": True,
            **ckpt_meta,
        },
        "inference_benchmark": {
            **result.metrics,
            "ink_probability_stats_2d_surface": ink_prob_stats,
        },
        "ground_truth_validation": gt_validation,
        "artifacts": png_paths,
    }

    json_path = output_dir / "benchmark_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = output_dir / "benchmark_report.md"
    md_path.write_text(_render_markdown_report(report), encoding="utf-8")

    return report


def _render_markdown_report(report: Dict[str, Any]) -> str:
    env = report["environment"]
    zarr_src = report["zarr_source"]
    cache = report["ddr5_cache"]
    ckpt = report["pretrained_checkpoint"]
    inf = report["inference_benchmark"]
    prob = inf["ink_probability_stats_2d_surface"]
    seam = inf["seam_quality"]
    gt = report.get("ground_truth_validation")
    arts = report["artifacts"]

    gt_section = ""
    if gt is not None:
        acc = gt["accuracy"]
        gt_inf = gt["inference_metrics"]
        gt_section = f"""
## 5. Ground-Truth Validation (`{gt['segment']}`)

- **Surface Volume URI**: `{gt['surface_uri']}`
- **Ground-Truth Labels URI**: `{gt['inklabels_uri']}`
- **ROI**: origin `{gt['roi_origin']}`, shape `{gt['roi_shape']}` (supervised fraction `{gt['supervised_fraction']:.2%}`, ink fraction `{gt['ground_truth_ink_fraction']:.2%}`)
- **Inference Throughput / VRAM**: `{gt_inf['throughput_tile_voxels_per_sec']:,.1f}` tile voxels/s in `{gt_inf['gpu_seconds']} s` | Peak VRAM `{gt_inf['peak_vram_allocated_mib']} MiB` allocated / `{gt_inf['peak_vram_reserved_mib']} MiB` reserved

| Metric | Score (`th=0.50`) | Optimal F0.5 (`th={acc['best_f0_5_threshold']}`) |
| :--- | :---: | :---: |
| **ROC-AUC** | **`{acc['roc_auc']}`** | `{acc['roc_auc']}` |
| **F0.5 Score** | `{acc['f0_5']}` | **`{acc['best_f0_5']}`** |
| **Dice / F1** | **`{acc['dice_f1']}`** | `{acc['dice_f1_at_best_f0_5']}` |
| **IoU (Jaccard)** | **`{acc['iou']}`** | `{acc['iou_at_best_f0_5']}` |
"""

    gt_art_line = (
        f"\n- Ground-Truth Validation Comparison: `{arts['ground_truth_validation_png']}`"
        if "ground_truth_validation_png" in arts
        else ""
    )

    return f"""# Vesuvius Ink-Detection Benchmark Report

Generated: `{report['timestamp_utc']}`

## 1. Environment & Pretrained Checkpoint

| Component | Value |
| :--- | :--- |
| **Platform** | `{env['platform']}` |
| **Python / PyTorch** | `{env['python']}` / `{env['torch']}` |
| **CUDA** | `{env['cuda_version']}` (`cuda_available={env['cuda_available']}`) |
| **GPU** | `{env['gpu_name']}` |
| **Precision / AMP** | `{inf['amp_dtype']}` (`pin_staging_buffer={inf['pin_staging_buffer']}`, `channels_last_3d={inf['use_channels_last_3d']}`, `seed={inf['seed']}`) |
| **HF Registry Repo** | `{ckpt['hf_repo_id']}` |
| **Checkpoint File** | `{ckpt['checkpoint_path']}` (epoch `{ckpt['epoch']}`, step `{ckpt['global_step']}`, `weights_only=True`) |
| **Architecture** | `{ckpt['backbone']}` (`{ckpt['num_state_dict_tensors']}` tensors, `strict=True`) |

## 2. Streaming OME-Zarr & Host DDR5 RAM Cache

- **Source URI**: `{zarr_src['uri']}` (level `{zarr_src['level']}`)
- **Full Level Shape**: `{zarr_src['shape']}` | **Chunk Shape**: `{zarr_src['chunks']}` | **Dtype**: `{zarr_src['dtype']}` | **Input Scaling**: `{zarr_src['input_intensity_scaling']}`
- **Streamed Sub-Volume ROI**: origin `{zarr_src['selected_roi_origin']}`, shape `{zarr_src['selected_roi_shape']}` ({zarr_src['roi_bytes'] / (1024**2):.2f} MiB, non-zero voxel fraction `{zarr_src['roi_nonzero_fraction']:.2%}`)
- **DDR5 Cache Budget**: `{cache['budget_gib']} GiB`
- **Cold Stream Time (AWS S3 -> DDR5)**: `{cache['cold_stream_seconds']} s` (fetched `{cache['after_cold_stats']['bytes_fetched_remote'] / (1024**2):.2f} MiB`)
- **Warm Cache Read Time (DDR5 -> Host)**: `{cache['warm_ddr5_read_seconds']} s` (`{cache['warm_ddr5_bandwidth_gb_s']} GB/s`, hit rate `{cache['after_warm_stats']['hit_rate']:.2%}`)

## 3. Mixed-Precision Inference Throughput & VRAM (RTX 5080 16GB)

| Metric | Value |
| :--- | :--- |
| **Tile Shape / Stride** | `{inf['tile_shape']}` / `{inf['stride']}` |
| **Total 3D Tiles / Batch Size** | `{inf['num_tiles']}` / `{inf['batch_size']}` |
| **GPU Inference Time** | `{inf['gpu_seconds']} s` (wall `{inf['wall_seconds']} s`) |
| **Throughput (Tile Voxels/sec)** | **`{inf['throughput_tile_voxels_per_sec']:,.1f}` voxels/s** |
| **Throughput (ROI Voxels/sec)** | **`{inf['throughput_roi_voxels_per_sec']:,.1f}` voxels/s** |
| **Tile Rate** | `{inf['tiles_per_sec']} tiles/s` |
| **Peak VRAM Allocated** | `{inf['peak_vram_allocated_mib']} MiB` |
| **Peak VRAM Reserved** | `{inf['peak_vram_reserved_mib']} MiB` (budget `{inf['vram_limit_gib']} GiB`, within budget: `{inf['within_vram_budget']}`) |
| **2D Ink Surface Probability Range** | `min={prob['min']}`, `mean={prob['mean']}`, `p99={prob['p99']}`, `max={prob['max']}` (`std={prob['std']}`) |

## 4. 3D + Z-Hann Seam-Blending Quality

| Metric | Unblended Tiling | 3D + Z-Hann Blending |
| :--- | :---: | :---: |
| **Seam Boundary Gradient** | `{seam['unblended_seam_gradient']}` | `{seam['hann_seam_gradient']}` |
| **Interior Gradient** | `{seam['unblended_interior_gradient']}` | `{seam['hann_interior_gradient']}` |
| **Seam-to-Interior Gradient Ratio** | `{seam['unblended_seam_to_interior_ratio']}x` | **`{seam['hann_seam_to_interior_ratio']}x`** |
| **Seam Discontinuity Reduction** | — | **`{seam['seam_discontinuity_reduction_pct']}%`** |
{gt_section}
## 6. Exported Visualizations

- Predicted 2D Ink Surface: `{arts['ink_surface_prediction_png']}`
- Seam-Blending Comparison: `{arts['seam_blending_comparison_png']}`{gt_art_line}
"""


def main() -> None:
    defaults = ZarrSourceConfig()
    inf_defaults = InferenceConfig()
    parser = argparse.ArgumentParser(description="Run vesuvius-ink-bench verification & benchmark.")
    parser.add_argument("--uri", type=str, default=defaults.uri, help="OME-Zarr URI (s3://, https://, or local)")
    parser.add_argument("--level", type=str, default=defaults.level, help="OME-Zarr multiscale level")
    parser.add_argument("--roi-shape", type=int, nargs=3, default=[128, 384, 384], help="Sub-volume shape D H W")
    parser.add_argument("--batch-size", type=int, default=inf_defaults.batch_size, help="Inference batch size")
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--checkpoint", type=Path, default=inf_defaults.checkpoint_path, help="Pretrained checkpoint path")
    parser.add_argument("--skip-gt-validation", action="store_true", help="Skip PHercParis4 ground-truth validation")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    cfg = BenchmarkConfig(
        source=ZarrSourceConfig(
            uri=args.uri,
            level=args.level,
            roi_shape=tuple(args.roi_shape),  # type: ignore[arg-type]
        ),
        cache=CacheConfig(),
        inference=InferenceConfig(
            batch_size=args.batch_size,
            amp_dtype=args.amp_dtype,
            checkpoint_path=args.checkpoint,
        ),
        output_dir=args.output_dir,
    )
    report = run_benchmark(cfg, include_gt_validation=not args.skip_gt_validation)
    print(json.dumps(report["inference_benchmark"], indent=2))
    if report.get("ground_truth_validation"):
        print(json.dumps(report["ground_truth_validation"]["accuracy"], indent=2))


if __name__ == "__main__":
    main()
