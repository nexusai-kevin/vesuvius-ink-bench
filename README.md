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

## Quick Start

On first run, `vesuvius-bench` automatically downloads the pretrained `scrollprize/ink_canonical_2um` checkpoint (`r152_3ddec_v2_l5_epoch13.ckpt`) from Hugging Face into `checkpoints/ink_canonical_2um/` (which is git-ignored) if not already present.

```powershell
cd vesuvius-ink-bench
.venv\Scripts\python.exe -m pytest -v
.venv\Scripts\python.exe -m vesuvius_ink_bench.benchmark --output-dir reports
```
