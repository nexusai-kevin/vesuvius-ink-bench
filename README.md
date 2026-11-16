# vesuvius-ink-bench

Volumetric ink-detection benchmarking and low-VRAM inference workspace compatible with **`ScrollPrize/villa`** OME-Zarr conventions, the official **`scrollprize/ink_canonical_2um`** Hugging Face checkpoint, and public **AWS Open Data Herculaneum scroll Zarr volumes** (`s3://vesuvius-challenge-open-data/`).

## Installation

```powershell
git clone https://github.com/nexusai-kevin/vesuvius-ink-bench.git
cd vesuvius-ink-bench
uv venv --python 3.12

# Standard install (CPU or default PyPI PyTorch wheel)
uv pip install -e .

# Optional: NVIDIA CUDA 12.8 PyTorch wheel index (recommended for RTX 50-series / Blackwell)
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
```

## Quick Start: Production Low-VRAM Inference (`vesuvius-infer`)

`vesuvius-infer` streams any public AWS Open Data Herculaneum OME-Zarr URL (or local `.zarr` v2/v3 volume) through a budgeted host DDR5 RAM LRU cache, runs `bfloat16` mixed-precision **ResNet3D-152 + 3D FPN decoder (`i3d`)** inference in **under 3 GiB VRAM** with strictly separable 3D + Z-Hann overlap blending, and exports:

- **Primary Output Product — 2D Ink Surface Probability Map**:
  - `ink_surface_16bit.tiff` — 16-bit grayscale TIFF surface projection (`uint16`)
  - `ink_surface.png` — 2D ink surface probability heatmap
  - `ink_predictions.zarr/surface_prob_2d` — Blosc/Zstd-compressed 2D Zarr array
- **Depth-Slab Predictions**:
  - `ink_predictions.zarr/0` — Blosc/Zstd-compressed 3D OME-Zarr array storing the 3D Hann-blended depth-slab predictions across `(Z, Y, X)`

On first run, `vesuvius-infer` (and `vesuvius-bench`) automatically downloads the pretrained `scrollprize/ink_canonical_2um` checkpoint (`r152_3ddec_v2_l5_epoch13.ckpt`) from Hugging Face into `checkpoints/ink_canonical_2um/` (git-ignored) if not already present, and verifies `weights_only=True` + `strict=True`.

```powershell
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

`vesuvius-bench` runs the full end-to-end verification pass:
1. Cold AWS Open Data S3 stream vs. warm host DDR5 RAM cache read (`PHerc0332`, level `2`)
2. RTX 5080 `bfloat16` throughput & peak VRAM profiling (`[0, 1]` input normalization)
3. Unblended vs. separable 3D + Z-Hann seam-blending discontinuity quantification
4. Ground-truth validation against labeled **`PHercParis4/segments/20230702185753`** (`2.4 µm` surface volume + `inklabels.zarr` + `supervision.zarr`)
5. Markdown (`reports/benchmark_report.md`), JSON (`reports/benchmark_report.json`), and PNG exports

```powershell
.venv\Scripts\python.exe -m pytest -v
.venv\Scripts\vesuvius-bench.exe --output-dir reports
```

## Key Architecture & Features

- **Exact `ScrollPrize/villa` Architecture (`model.py`)**: `VillaInkResNet3D` matches `ink-detection/optimized_inference/model_resnet3d_3d_decoder.py` (`ResNet3DBackbone`, `DecoderResBlock3D` with `GroupNorm(min(32, C), C)` + `ReLU`, `DepthAttentionCollapse`, and `BatchNorm3d(1)` normalization), loading all `976` checkpoint tensors via `torch.load(..., weights_only=True)` with `strict=True`.
- **Streaming OME-Zarr Loader & DDR5 Cache (`zarr_loader.py`)**: Supports both Zarr v2 and Zarr v3 (`FsspecStore`) over `s3://` (anonymous), `https://`, or local storage with a thread-safe `DDR5ChunkCache`.
- **Separable 3D + Z-Hann Blending (`windowing.py`)**: Per-axis raised-Hann weights preserve smooth `(Y, X)` tapering across every depth slice, and Z-Hann depth aggregation eliminates rectangular Z-slab footprints.
- **RTX 5080 Mixed-Precision Pipeline (`pipeline.py`)**: `bfloat16` autocast, pinned host staging buffers, optional `channels_last_3d` for 3D convolutions, deterministic seeding, CUDA event timing, and peak VRAM monitoring.

## Verified RTX 5080 Benchmark & Ground-Truth Validation

| Benchmark | Metric | Value |
| :--- | :--- | :--- |
| **`PHerc0332` (Level `2`, `bfloat16`)** | Tile Throughput | `83,196,727.9` voxels/s (`79.34` tiles/s; `19,385,945.7` ROI voxels/s) |
| **`PHerc0332` (Level `2`, `bfloat16`)** | Peak VRAM | `2,506.13 MiB` allocated / `2,912.0 MiB` reserved (`< 3 GiB`) |
| **`PHerc0332` (3D + Z-Hann Blending)** | Seam Discontinuity Reduction | `99.42%` (`122.4066x` unblended vs. `0.3466x` Hann-blended) |
| **`PHercParis4` (`2.4 µm`, `inklabels.zarr`)** | ROC-AUC | **`0.9546`** |
| **`PHercParis4` (`2.4 µm`, `inklabels.zarr`)** | F0.5 / Dice-F1 / IoU (`th=0.50`) | `0.7903` / `0.8271` / `0.7052` (best F0.5 **`0.8724`** at `th=0.70`) |

## License & Dataset Attribution

- **Code License**: Released under the **MIT License** (see [LICENSE](LICENSE)).
- **Dataset Attribution**: Tomographic sub-volume benchmarks and validation use the **EduceLab-Scrolls / Herculaneum CT scan datasets (`PHerc0332` and `PHercParis4`)** hosted on AWS Open Data (`s3://vesuvius-challenge-open-data/`) in accordance with the **Vesuvius Challenge Data Agreement** (underlying tomographic data distributed under **CC BY-NC 4.0** by EduceLab / Vesuvius Challenge).
- **Pretrained Weights**: `scrollprize/ink_canonical_2um` (`r152_3ddec_v2_l5_epoch13.ckpt`), released under the **MIT License** by the Vesuvius Challenge.
