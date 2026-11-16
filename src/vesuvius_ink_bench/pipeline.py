"""Mixed-precision PyTorch 3D inference pipeline with 3D Hann-window blending."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from vesuvius_ink_bench.config import InferenceConfig
from vesuvius_ink_bench.windowing import (
    compute_hann_window_3d,
    compute_seam_metrics,
    compute_tile_grid,
    project_surface_z_hann,
)


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(slots=True)
class InferenceResult:
    """Outputs and telemetry from a 3D volumetric inference pass."""

    blended_3d: np.ndarray
    unblended_3d: np.ndarray
    blended_surface_2d: np.ndarray
    blended_mean_2d: np.ndarray
    unblended_surface_2d: np.ndarray
    metrics: Dict[str, object]


class MixedPrecisionInkPipeline:
    """RTX 5080-tuned mixed-precision 3D tiling & Hann-window inference pipeline."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[InferenceConfig] = None,
        *,
        pin_staging_buffer: bool = True,
    ) -> None:
        self.config = config or InferenceConfig()
        self.pin_staging_buffer = pin_staging_buffer
        torch.manual_seed(self.config.seed)

        self.device = torch.device(
            self.config.device if (self.config.device != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)

        self.amp_dtype = _DTYPE_MAP[self.config.amp_dtype]
        self.model = model.to(self.device).eval()
        if self.config.use_channels_last_3d:
            for mod in self.model.modules():
                if isinstance(mod, (nn.Conv3d, nn.BatchNorm3d)):
                    mod.to(memory_format=torch.channels_last_3d)

        if self.config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode="reduce-overhead")

        self.hann_window_3d = compute_hann_window_3d(
            self.config.tile_shape,
            device=self.device,
            dtype=torch.float32,
        )

    def _run_batch(self, batch_cpu: torch.Tensor) -> torch.Tensor:
        if self.device.type == "cuda":
            if self.config.use_channels_last_3d:
                batch_dev = batch_cpu.to(
                    self.device,
                    memory_format=torch.channels_last_3d,
                    non_blocking=True,
                )
            else:
                batch_dev = batch_cpu.to(self.device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                logits = self.model(batch_dev)
        else:
            batch_in = (
                batch_cpu.to(memory_format=torch.channels_last_3d)
                if self.config.use_channels_last_3d
                else batch_cpu
            )
            logits = self.model(batch_in)
        return torch.sigmoid(logits.float())

    @torch.inference_mode()
    def run(
        self,
        volume: np.ndarray,
        *,
        warmup_batches: int = 1,
    ) -> InferenceResult:
        """Run tiled 3D inference on a ``(D, H, W)`` uint8/float32 sub-volume."""
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume (D, H, W), got shape {volume.shape}")

        vd, vh, vw = volume.shape
        td, th, tw = self.config.tile_shape
        coords = compute_tile_grid((vd, vh, vw), (td, th, tw), self.config.stride)
        num_tiles = len(coords)

        # Normalize uint8 CT intensities to [0, 1] matching ScrollPrize/villa BatchNorm3d stats
        vol_host = torch.from_numpy(np.ascontiguousarray(volume)).to(torch.float32) / 255.0
        if self.pin_staging_buffer and self.device.type == "cuda":
            vol_host = vol_host.pin_memory()

        blended_accum = torch.zeros((vd, vh, vw), dtype=torch.float32, device=self.device)
        weight_accum = torch.zeros((vd, vh, vw), dtype=torch.float32, device=self.device)
        unblended_vol = torch.zeros((vd, vh, vw), dtype=torch.float32, device=self.device)

        if warmup_batches > 0 and self.device.type == "cuda":
            dummy = torch.zeros(
                (min(self.config.batch_size, num_tiles), 1, td, th, tw), dtype=torch.float32
            )
            for _ in range(warmup_batches):
                _ = self._run_batch(dummy)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(self.device)

        t_wall_0 = time.perf_counter()
        if self.device.type == "cuda":
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()

        bs = self.config.batch_size
        total_tile_voxels = 0
        for i in range(0, num_tiles, bs):
            batch_coords = coords[i : i + bs]
            tiles: List[torch.Tensor] = [
                vol_host[z : z + td, y : y + th, x : x + tw].unsqueeze(0)
                for (z, y, x) in batch_coords
            ]
            batch_tensor = torch.stack(tiles, dim=0)
            if self.pin_staging_buffer and self.device.type == "cuda":
                batch_tensor = batch_tensor.pin_memory()

            probs = self._run_batch(batch_tensor).squeeze(1)  # (B, th, tw) or (B, td, th, tw)
            total_tile_voxels += len(batch_coords) * td * th * tw

            for b_idx, (z, y, x) in enumerate(batch_coords):
                p = probs[b_idx]
                if p.ndim == 2:
                    p = p.unsqueeze(0).expand(td, th, tw)
                blended_accum[z : z + td, y : y + th, x : x + tw] += p * self.hann_window_3d
                weight_accum[z : z + td, y : y + th, x : x + tw] += self.hann_window_3d
                unblended_vol[z : z + td, y : y + th, x : x + tw] = p

        if self.device.type == "cuda":
            ev_end.record()
            torch.cuda.synchronize()
            gpu_ms = float(ev_start.elapsed_time(ev_end))
            peak_alloc_bytes = int(torch.cuda.max_memory_allocated(self.device))
            peak_reserved_bytes = int(torch.cuda.max_memory_reserved(self.device))
        else:
            gpu_ms = (time.perf_counter() - t_wall_0) * 1000.0
            peak_alloc_bytes = 0
            peak_reserved_bytes = 0

        wall_s = time.perf_counter() - t_wall_0
        blended_vol = blended_accum / weight_accum.clamp_min(1e-24)

        blended_np = blended_vol.cpu().numpy()
        unblended_np = unblended_vol.cpu().numpy()

        # Smooth Z-Hann depth aggregation eliminates Z-slab rectangular boundaries
        blended_surface_2d = project_surface_z_hann(blended_np)
        blended_mean_2d = blended_np.mean(axis=0)
        unblended_surface_2d = project_surface_z_hann(unblended_np)

        seam_metrics = compute_seam_metrics(
            blended_surface_2d,
            unblended_surface_2d,
            tile_shape=(th, tw),
            stride=(self.config.stride[1], self.config.stride[2]),
        )

        roi_voxels = int(vd * vh * vw)
        gpu_s = max(gpu_ms / 1000.0, 1e-9)
        metrics: Dict[str, object] = {
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
            "amp_dtype": self.config.amp_dtype,
            "pin_staging_buffer": self.pin_staging_buffer,
            "use_channels_last_3d": self.config.use_channels_last_3d,
            "seed": self.config.seed,
            "roi_shape": [vd, vh, vw],
            "roi_voxels": roi_voxels,
            "tile_shape": [td, th, tw],
            "stride": list(self.config.stride),
            "num_tiles": num_tiles,
            "batch_size": bs,
            "total_processed_tile_voxels": total_tile_voxels,
            "wall_seconds": round(wall_s, 4),
            "gpu_seconds": round(gpu_s, 4),
            "throughput_roi_voxels_per_sec": round(roi_voxels / wall_s, 1),
            "throughput_tile_voxels_per_sec": round(total_tile_voxels / gpu_s, 1),
            "tiles_per_sec": round(num_tiles / gpu_s, 2),
            "peak_vram_allocated_mib": round(peak_alloc_bytes / (1024**2), 2),
            "peak_vram_reserved_mib": round(peak_reserved_bytes / (1024**2), 2),
            "vram_limit_gib": self.config.vram_limit_gib,
            "within_vram_budget": (peak_reserved_bytes / (1024**3)) <= self.config.vram_limit_gib,
            "seam_quality": seam_metrics,
        }

        return InferenceResult(
            blended_3d=blended_np,
            unblended_3d=unblended_np,
            blended_surface_2d=blended_surface_2d,
            blended_mean_2d=blended_mean_2d,
            unblended_surface_2d=unblended_surface_2d,
            metrics=metrics,
        )
