"""Production CLI for low-VRAM 3D Hann-blended ink inference (vesuvius-infer).

The primary output product is the seamless 2D ink surface probability map
(``surface_prob_2d`` in the OME-Zarr store, ``ink_surface_16bit.tiff``, and
``ink_surface.png``), produced by Z-Hann-weighted depth aggregation over the
3D Hann-blended depth-slab predictions stored at OME-Zarr level ``0``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
import zarr.codecs
from PIL import Image

from vesuvius_ink_bench.config import (
    CacheConfig,
    InferenceConfig,
    ZarrSourceConfig,
)
from vesuvius_ink_bench.model import VillaInkResNet3D
from vesuvius_ink_bench.pipeline import MixedPrecisionInkPipeline
from vesuvius_ink_bench.zarr_loader import StreamingZarrSubvolumeLoader


def _ensure_checkpoint(checkpoint_path: Path, hf_repo_id: str) -> Path:
    if not checkpoint_path.exists():
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=hf_repo_id,
            local_dir=str(checkpoint_path.parent),
        )
    return checkpoint_path


def _export_ome_zarr(
    blended_3d: np.ndarray,
    blended_surface_2d: np.ndarray,
    zarr_out_path: Path,
) -> Path:
    """Export seamless 3D depth-slab predictions (``0``) and primary 2D surface map."""
    compressor = zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
    root = zarr.open_group(str(zarr_out_path), mode="w")
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "ink_probability_3d_depth_slabs",
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [{"path": "0"}],
        }
    ]
    d, h, w = blended_3d.shape
    chunks_3d = (min(64, d), min(128, h), min(128, w))
    arr3d = root.create_array(
        "0",
        shape=blended_3d.shape,
        chunks=chunks_3d,
        dtype="float32",
        compressors=compressor,
        overwrite=True,
    )
    arr3d[:] = blended_3d.astype(np.float32, copy=False)

    chunks_2d = (min(256, h), min(256, w))
    arr2d = root.create_array(
        "surface_prob_2d",
        shape=blended_surface_2d.shape,
        chunks=chunks_2d,
        dtype="float32",
        compressors=compressor,
        overwrite=True,
    )
    arr2d[:] = blended_surface_2d.astype(np.float32, copy=False)
    return zarr_out_path


def _export_16bit_tiff_and_png(
    blended_surface_2d: np.ndarray,
    tiff_path: Path,
    png_path: Path,
) -> Tuple[Path, Path]:
    """Export the primary 2D ink surface projection as a 16-bit TIFF and PNG."""
    prob_clipped = np.clip(blended_surface_2d, 0.0, 1.0)
    u16 = np.rint(prob_clipped * 65535.0).astype(np.uint16)
    Image.fromarray(u16).save(tiff_path, format="TIFF")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    im = ax.imshow(prob_clipped, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title("Predicted 2D Ink Surface (3D + Z-Hann)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return tiff_path, png_path


def run_inference(
    *,
    uri: str,
    level: str = "2",
    origin: Optional[Tuple[int, int, int]] = None,
    shape: Optional[Tuple[int, int, int]] = None,
    full_level: bool = False,
    output_dir: Path = Path("outputs"),
    batch_size: int = 8,
    amp_dtype: str = "bfloat16",
    checkpoint_path: Optional[Path] = None,
    hf_repo_id: Optional[str] = None,
    anon: bool = True,
    pin_staging_buffer: bool = True,
    use_channels_last_3d: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run low-VRAM 3D Hann-blended ink inference and export OME-Zarr, 16-bit TIFF, and PNG."""
    src_defaults = ZarrSourceConfig()
    inf_defaults = InferenceConfig()

    ckpt_path = _ensure_checkpoint(
        checkpoint_path or inf_defaults.checkpoint_path,
        hf_repo_id or inf_defaults.hf_repo_id,
    )

    source_cfg = ZarrSourceConfig(
        uri=uri,
        level=level,
        anon=anon,
        roi_origin=origin or src_defaults.roi_origin,
        roi_shape=shape or src_defaults.roi_shape,
    )
    cache_cfg = CacheConfig(pin_staging_buffer=pin_staging_buffer)
    loader = StreamingZarrSubvolumeLoader(source_cfg, cache_cfg)

    if full_level:
        roi_origin = (0, 0, 0)
        roi_shape = loader.shape
    else:
        roi_origin = origin or src_defaults.roi_origin
        roi_shape = shape or src_defaults.roi_shape

    volume = loader.read_subvolume(origin=roi_origin, roi_shape=roi_shape)

    model, ckpt_meta = VillaInkResNet3D.from_villa_checkpoint(ckpt_path, strict=True)
    inf_cfg = InferenceConfig(
        batch_size=batch_size,
        amp_dtype=amp_dtype,  # type: ignore[arg-type]
        checkpoint_path=ckpt_path,
        hf_repo_id=hf_repo_id or inf_defaults.hf_repo_id,
        use_channels_last_3d=use_channels_last_3d,
        seed=seed,
    )
    pipeline = MixedPrecisionInkPipeline(
        model, inf_cfg, pin_staging_buffer=cache_cfg.pin_staging_buffer
    )
    result = pipeline.run(volume, warmup_batches=1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zarr_out = _export_ome_zarr(
        result.blended_3d,
        result.blended_surface_2d,
        out_dir / "ink_predictions.zarr",
    )
    tiff_out, png_out = _export_16bit_tiff_and_png(
        result.blended_surface_2d,
        out_dir / "ink_surface_16bit.tiff",
        out_dir / "ink_surface.png",
    )

    summary: Dict[str, Any] = {
        "zarr_uri": uri,
        "level": level,
        "roi_origin": list(roi_origin),
        "roi_shape": list(roi_shape),
        "checkpoint": ckpt_meta,
        "metrics": result.metrics,
        "exports": {
            "primary_tiff_16bit": str(tiff_out),
            "primary_png_surface": str(png_out),
            "ome_zarr": str(zarr_out),
        },
    }
    (out_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    src_defaults = ZarrSourceConfig()
    inf_defaults = InferenceConfig()
    parser = argparse.ArgumentParser(
        prog="vesuvius-infer",
        description=(
            "Low-VRAM 3D Hann-blended ink detection on AWS Open Data or local OME-Zarr volumes. "
            "Exports primary 2D ink surface projections (16-bit TIFF, PNG, surface_prob_2d) "
            "and 3D depth-slab predictions (OME-Zarr level 0)."
        ),
    )
    parser.add_argument(
        "--uri",
        type=str,
        default=src_defaults.uri,
        help="OME-Zarr URI (s3://, https://, or local .zarr path)",
    )
    parser.add_argument("--level", type=str, default=src_defaults.level, help="OME-Zarr multiscale level")
    parser.add_argument(
        "--origin",
        type=int,
        nargs=3,
        default=list(src_defaults.roi_origin),
        metavar=("Z", "Y", "X"),
        help="Bounding-box origin (z y x)",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=list(src_defaults.roi_shape),
        metavar=("D", "H", "W"),
        help="Bounding-box shape (d h w)",
    )
    parser.add_argument(
        "--full-level",
        action="store_true",
        help="Run inference over the entire selected OME-Zarr level instead of a sub-volume ROI",
    )
    parser.add_argument("--batch-size", type=int, default=inf_defaults.batch_size, help="Inference batch size")
    parser.add_argument(
        "--amp-dtype",
        choices=["bfloat16", "float16", "float32"],
        default=inf_defaults.amp_dtype,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=inf_defaults.checkpoint_path,
        help="Pretrained checkpoint path (auto-downloads from Hugging Face if missing)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Export directory")
    args = parser.parse_args()

    summary = run_inference(
        uri=args.uri,
        level=args.level,
        origin=tuple(args.origin),  # type: ignore[arg-type]
        shape=tuple(args.shape),  # type: ignore[arg-type]
        full_level=args.full_level,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        amp_dtype=args.amp_dtype,
        checkpoint_path=args.checkpoint,
    )
    print(json.dumps(summary["exports"], indent=2))


if __name__ == "__main__":
    main()
