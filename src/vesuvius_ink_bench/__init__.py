"""Volumetric ink-detection benchmarking workspace for ScrollPrize/villa & AWS Open Data Zarr volumes."""

from vesuvius_ink_bench.config import (
    BenchmarkConfig,
    CacheConfig,
    InferenceConfig,
    ZarrSourceConfig,
)
from vesuvius_ink_bench.zarr_loader import DDR5ChunkCache, StreamingZarrSubvolumeLoader
from vesuvius_ink_bench.windowing import (
    compute_hann_window_3d,
    compute_seam_metrics,
)
from vesuvius_ink_bench.infer import run_inference
from vesuvius_ink_bench.model import VillaInkResNet3D
from vesuvius_ink_bench.pipeline import MixedPrecisionInkPipeline

__all__ = [
    "BenchmarkConfig",
    "CacheConfig",
    "DDR5ChunkCache",
    "InferenceConfig",
    "MixedPrecisionInkPipeline",
    "StreamingZarrSubvolumeLoader",
    "VillaInkResNet3D",
    "ZarrSourceConfig",
    "compute_hann_window_3d",
    "compute_seam_metrics",
    "run_inference",
]
