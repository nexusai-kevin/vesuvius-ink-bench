"""Configuration dataclasses for vesuvius-ink-bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Tuple


DEFAULT_S3_ZARR_URI = (
    "s3://vesuvius-challenge-open-data/"
    "PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr"
)
DEFAULT_HF_REPO_ID = "scrollprize/ink_canonical_2um"
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/ink_canonical_2um/r152_3ddec_v2_l5_epoch13.ckpt")


@dataclass(slots=True)
class ZarrSourceConfig:
    """Configuration for a ScrollPrize/villa or AWS Open Data OME-Zarr volume."""

    uri: str = DEFAULT_S3_ZARR_URI
    level: str = "2"
    anon: bool = True
    # Sub-volume ROI in (z, y, x) order: (start, stop)
    # Level 2 of PHerc0332 has shape (8398, 3941, 3941) with 128^3 chunks.
    roi_origin: Tuple[int, int, int] = (4096, 1792, 1792)
    roi_shape: Tuple[int, int, int] = (128, 384, 384)


@dataclass(slots=True)
class CacheConfig:
    """Host DDR5 RAM LRU chunk cache configuration."""

    max_bytes: int = 16 * 1024**3  # 16 GiB host DDR5 budget
    prefetch_workers: int = 8
    pin_staging_buffer: bool = True


@dataclass(slots=True)
class InferenceConfig:
    """Mixed-precision PyTorch inference & 3D Hann-window tiling config (RTX 5080 16GB)."""

    tile_shape: Tuple[int, int, int] = (64, 128, 128)
    stride: Tuple[int, int, int] = (32, 64, 64)
    batch_size: int = 8
    amp_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    device: str = "cuda"
    use_channels_last_3d: bool = False
    compile_model: bool = False
    vram_limit_gib: float = 14.5
    seed: int = 42
    hf_repo_id: str = DEFAULT_HF_REPO_ID
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH


@dataclass(slots=True)
class BenchmarkConfig:
    """Top-level configuration for a reproducible verification & benchmark run."""

    source: ZarrSourceConfig = field(default_factory=ZarrSourceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output_dir: Path = Path("reports")
    warmup_batches: int = 2
