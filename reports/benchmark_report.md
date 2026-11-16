# Vesuvius Ink-Detection Benchmark Report

Generated: `2026-11-16T20:56:00.936590+00:00`

## 1. Environment & Pretrained Checkpoint

| Component | Value |
| :--- | :--- |
| **Platform** | `Windows-11-10.0.26200-SP0` |
| **Python / PyTorch** | `3.12.13` / `2.11.0+cu128` |
| **CUDA** | `12.8` (`cuda_available=True`) |
| **GPU** | `NVIDIA GeForce RTX 5080` |
| **Precision / AMP** | `bfloat16` |
| **HF Registry Repo** | `scrollprize/ink_canonical_2um` |
| **Checkpoint File** | `checkpoints\ink_canonical_2um\r152_3ddec_v2_l5_epoch13.ckpt` (epoch `13`, step `113246`) |
| **Architecture** | `ResNet3D-152 + 3D FPN Decoder (i3d)` (`976` tensors, `strict=True`) |

## 2. Streaming OME-Zarr & Host DDR5 RAM Cache

- **Source URI**: `s3://vesuvius-challenge-open-data/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr` (level `2`)
- **Full Level Shape**: `(8398, 3941, 3941)` | **Chunk Shape**: `(128, 128, 128)` | **Dtype**: `uint8`
- **Streamed Sub-Volume ROI**: origin `[4096, 1792, 1792]`, shape `[128, 384, 384]` (18.00 MiB, non-zero voxel fraction `100.00%`)
- **DDR5 Cache Budget**: `16.0 GiB`
- **Cold Stream Time (AWS S3 -> DDR5)**: `1.753 s` (fetched `18.00 MiB`)
- **Warm Cache Read Time (DDR5 -> Host)**: `0.007922 s` (`2.38 GB/s`, hit rate `50.00%`)

## 3. Mixed-Precision Inference Throughput & VRAM (RTX 5080 16GB)

| Metric | Value |
| :--- | :--- |
| **Tile Shape / Stride** | `[64, 128, 128]` / `[32, 64, 64]` |
| **Total 3D Tiles / Batch Size** | `75` / `8` |
| **GPU Inference Time** | `1.1425 s` (wall `1.1936 s`) |
| **Throughput (Tile Voxels/sec)** | **`68,833,531.1` voxels/s** |
| **Throughput (ROI Voxels/sec)** | **`15,813,451.5` voxels/s** |
| **Tile Rate** | `65.64 tiles/s` |
| **Peak VRAM Allocated** | `2442.13 MiB` |
| **Peak VRAM Reserved** | `2912.0 MiB` (budget `14.5 GiB`, within budget: `True`) |
| **2D Ink MIP Probability Range** | `min=0.036768`, `mean=0.348697`, `p99=0.708491`, `max=0.855716` (`std=0.147967`) |

## 4. 3D Hann-Window Seam-Blending Quality

| Metric | Unblended Tiling | 3D Hann-Window Blending |
| :--- | :---: | :---: |
| **Seam Boundary Gradient** | `0.118811` | `0.033587` |
| **Interior Gradient** | `0.009023` | `0.008073` |
| **Seam-to-Interior Gradient Ratio** | `13.1677x` | **`4.1602x`** |
| **Seam Discontinuity Reduction** | — | **`71.73%`** |

## 5. Exported Visualizations

- Predicted 2D Ink Surface: `reports\ink_surface_prediction.png`
- Seam-Blending Comparison: `reports\seam_blending_comparison.png`
