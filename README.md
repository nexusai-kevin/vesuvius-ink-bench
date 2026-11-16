# vesuvius-ink-bench

Volumetric ink-detection benchmarking workspace compatible with **`ScrollPrize/villa`** OME-Zarr conventions, the official **`scrollprize/ink_canonical_2um`** Hugging Face checkpoint, and public **AWS Open Data Herculaneum scroll Zarr volumes** (`s3://vesuvius-challenge-open-data/`).

## Key Features

- **Pretrained ScrollPrize Model (`model.py`)**: `VillaInkResNet3D` implements the canonical **ResNet3D-152 + 3D FPN decoder (`i3d`)** with depth-attention collapse and `BatchNorm3d` input normalization, loading `r152_3ddec_v2_l5_epoch13.ckpt` (`976` tensors, epoch `13`, global step `113,246`) with `strict=True`.
- **Streaming OME-Zarr Sub-Volume Loader (`zarr_loader.py`)**: Streams chunk-aligned 3D ROIs over `s3://` (anonymous), `https://`, or local `.zarr` stores without downloading multi-terabyte datasets.
- **Host DDR5 RAM LRU Chunk Cache (`DDR5ChunkCache`)**: Thread-safe, budgeted LRU cache in host DDR5 RAM with concurrent chunk prefetching and cache hit/miss/bandwidth telemetry.
- **Mixed-Precision PyTorch Pipeline (`pipeline.py`)**: Tuned for **NVIDIA RTX 5080 (16GB VRAM)** using `torch.amp.autocast("cuda", dtype=torch.bfloat16)`, pinned host memory transfers (`non_blocking=True`), CUDA event timing, and peak VRAM budget tracking.
- **3D Hann-Window Overlap Blending (`windowing.py`)**: Separable 3D raised-Hann windowing across `(Z, Y, X)` to suppress tile-seam artifacts, paired with quantitative Boundary Gradient Discontinuity (BGD) seam metrics.
- **Reproducible Reports & Visualizations (`benchmark.py`, `visualize.py`)**: Exports 2D ink surface PNGs and reproducible Markdown + JSON benchmark reports.

## Verified RTX 5080 Benchmark (`PHerc0332`, Level `2`, `bfloat16`)

| Metric | Value |
| :--- | :--- |
| **Pretrained Checkpoint** | `scrollprize/ink_canonical_2um` (`r152_3ddec_v2_l5_epoch13.ckpt`, `strict=True`) |
| **ROI / Tile Shape** | `[128, 384, 384]` / `[64, 128, 128]` (stride `[32, 64, 64]`, `75` tiles, `batch_size=8`) |
| **Tile Throughput** | `68,833,531.1` voxels/s (`65.64` tiles/s; `15,813,451.5` ROI voxels/s) |
| **Peak VRAM** | `2,442.13 MiB` allocated / `2,912.0 MiB` reserved (14.5 GiB budget) |
| **Seam Discontinuity Reduction** | `71.73%` (`13.1677x` unblended vs `4.1602x` 3D Hann-blended) |
| **2D Ink MIP Probability** | `min=0.036768`, `mean=0.348697`, `p99=0.708491`, `max=0.855716` |

## Quick Start: Production Low-VRAM Inference (`vesuvius-infer`)

`vesuvius-infer` streams any public AWS Open Data Herculaneum OME-Zarr URL (or local `.zarr` volume) using the host DDR5 RAM cache, runs `bfloat16` mixed-precision ResNet3D-152 inference in **under 3 GiB VRAM** with 3D Hann-window overlap blending, and exports:
- `ink_predictions.zarr` — Blosc/Zstd-compressed OME-Zarr (`0` 3D volume + `surface_mip_2d` projection)
- `ink_surface_16bit.tiff` — 16-bit grayscale TIFF surface projection (`uint16`)
- `ink_surface.png` — 2D ink probability heatmap
- `inference_summary.json` — ROI, VRAM, and throughput telemetry

On first run, `vesuvius-infer` (and `vesuvius-bench`) automatically downloads the pretrained `scrollprize/ink_canonical_2um` checkpoint (`r152_3ddec_v2_l5_epoch13.ckpt`) from Hugging Face into `checkpoints/ink_canonical_2um/` (git-ignored) if not already present.

```powershell
cd vesuvius-ink-bench

# 1. Custom (z, y, x) bounding-box inference on AWS Open Data PHerc0332
.venv\Scripts\vesuvius-infer.exe `
  --uri s3://vesuvius-challenge-open-data/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr `
  --level 2 `
  --origin 4096 1792 1792 `
  --shape 128 384 384 `
  --output-dir outputs

# 2. Or full-level inference on a local or remote OME-Zarr level
.venv\Scripts\vesuvius-infer.exe --uri path/to/scroll.zarr --level 2 --full-level --output-dir outputs
```

## Reproducible Validation Suite (`vesuvius-bench`)

Use `vesuvius-bench` to run the end-to-end verification pass (cold S3 stream vs. warm DDR5 cache read, RTX 5080 throughput & VRAM profiling, seam-blending discontinuity quantification, and Markdown/JSON report generation) and `pytest` to run the unit/integration suite:

```powershell
.venv\Scripts\python.exe -m pytest -v
.venv\Scripts\vesuvius-bench.exe --output-dir reports
```

## License & Dataset Attribution

- **Code License**: Released under the **MIT License** (see [LICENSE](LICENSE)).
- **Dataset Attribution**: Tomographic sub-volume benchmarks use the **EduceLab-Scrolls / Herculaneum CT scan dataset (`PHerc0332`)** hosted on AWS Open Data (`s3://vesuvius-challenge-open-data/`) in accordance with the **Vesuvius Challenge Data Agreement** (underlying tomographic data distributed under **CC BY-NC 4.0** by EduceLab / Vesuvius Challenge).
- **Pretrained Weights**: `scrollprize/ink_canonical_2um` (`r152_3ddec_v2_l5_epoch13.ckpt`), released under the **MIT License** by the Vesuvius Challenge.

