# Vesuvius Ink-Detection Benchmark Report

Generated: `2026-11-16T22:36:14.358884+00:00`

## 1. Environment & Pretrained Checkpoint

| Component | Value |
| :--- | :--- |
| **Platform** | `Windows-11-10.0.26200-SP0` |
| **Python / PyTorch** | `3.12.13` / `2.11.0+cu128` |
| **CUDA** | `12.8` (`cuda_available=True`) |
| **GPU** | `NVIDIA GeForce RTX 5080` |
| **Precision / AMP** | `bfloat16` (`pin_staging_buffer=True`, `channels_last_3d=False`, `seed=42`) |
| **HF Registry Repo** | `scrollprize/ink_canonical_2um` |
| **Checkpoint File** | `checkpoints\ink_canonical_2um\r152_3ddec_v2_l5_epoch13.ckpt` (epoch `13`, step `113246`, `weights_only=True`) |
| **Architecture** | `ResNet3D-152 + 3D FPN Decoder (i3d)` (`976` tensors, `strict=True`) |

## 2. Streaming OME-Zarr & Host DDR5 RAM Cache

- **Source URI**: `s3://vesuvius-challenge-open-data/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr` (level `2`)
- **Full Level Shape**: `(8398, 3941, 3941)` | **Chunk Shape**: `(128, 128, 128)` | **Dtype**: `uint8` | **Input Scaling**: `[0, 1] (uint8 / 255.0)`
- **Streamed Sub-Volume ROI**: origin `[4096, 1792, 1792]`, shape `[128, 384, 384]` (18.00 MiB, non-zero voxel fraction `100.00%`)
- **DDR5 Cache Budget**: `16.0 GiB`
- **Cold Stream Time (AWS S3 -> DDR5)**: `1.7517 s` (fetched `18.00 MiB`)
- **Warm Cache Read Time (DDR5 -> Host)**: `0.00886 s` (`2.13 GB/s`, hit rate `50.00%`)

## 3. Mixed-Precision Inference Throughput & VRAM (RTX 5080 16GB)

| Metric | Value |
| :--- | :--- |
| **Tile Shape / Stride** | `[64, 128, 128]` / `[32, 64, 64]` |
| **Total 3D Tiles / Batch Size** | `75` / `8` |
| **GPU Inference Time** | `0.9453 s` (wall `0.9736 s`) |
| **Throughput (Tile Voxels/sec)** | **`83,196,727.9` voxels/s** |
| **Throughput (ROI Voxels/sec)** | **`19,385,945.7` voxels/s** |
| **Tile Rate** | `79.34 tiles/s` |
| **Peak VRAM Allocated** | `2506.13 MiB` |
| **Peak VRAM Reserved** | `2912.0 MiB` (budget `14.5 GiB`, within budget: `True`) |
| **2D Ink Surface Probability Range** | `min=0.140929`, `mean=0.36425`, `p99=0.615562`, `max=0.635512` (`std=0.140087`) |

## 4. 3D + Z-Hann Seam-Blending Quality

| Metric | Unblended Tiling | 3D + Z-Hann Blending |
| :--- | :---: | :---: |
| **Seam Boundary Gradient** | `0.114695` | `0.000661` |
| **Interior Gradient** | `0.000937` | `0.001906` |
| **Seam-to-Interior Gradient Ratio** | `122.4066x` | **`0.3466x`** |
| **Seam Discontinuity Reduction** | — | **`99.42%`** |

## 5. Ground-Truth Validation (`PHercParis4/segments/20230702185753 (2.4 um)`)

- **Surface Volume URI**: `s3://vesuvius-challenge-open-data/PHercParis4/segments/20230702185753/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr`
- **Ground-Truth Labels URI**: `s3://vesuvius-challenge-open-data/PHercParis4/segments/20230702185753/ink-labels/2.4um-volume-20260411134726/20260918/inklabels.zarr`
- **ROI**: origin `[0, 46080, 9728]`, shape `[109, 512, 512]` (supervised fraction `100.00%`, ink fraction `27.98%`)
- **Inference Throughput / VRAM**: `87,244,423.5` tile voxels/s in `1.7668 s` | Peak VRAM `3121.41 MiB` allocated / `3444.0 MiB` reserved

| Metric | Score (`th=0.50`) | Optimal F0.5 (`th=0.7`) |
| :--- | :---: | :---: |
| **ROC-AUC** | **`0.9546`** | `0.9546` |
| **F0.5 Score** | `0.7903` | **`0.8724`** |
| **Dice / F1** | **`0.8271`** | `0.8491` |
| **IoU (Jaccard)** | **`0.7052`** | `0.7377` |

## 6. Exported Visualizations

- Predicted 2D Ink Surface: `reports\ink_surface_prediction.png`
- Seam-Blending Comparison: `reports\seam_blending_comparison.png`
- Ground-Truth Validation Comparison: `reports\ground_truth_validation.png`
