"""Unit and integration tests for vesuvius-ink-bench."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import zarr

from vesuvius_ink_bench.config import CacheConfig, InferenceConfig, ZarrSourceConfig
from vesuvius_ink_bench.infer import run_inference
from vesuvius_ink_bench.model import VillaInkResNet3D
from vesuvius_ink_bench.pipeline import MixedPrecisionInkPipeline
from vesuvius_ink_bench.windowing import (
    compute_hann_window_3d,
    compute_ink_accuracy_metrics,
    compute_seam_metrics,
    project_surface_z_hann,
)
from vesuvius_ink_bench.zarr_loader import DDR5ChunkCache, StreamingZarrSubvolumeLoader


def test_ddr5_chunk_cache_lru_eviction() -> None:
    cache = DDR5ChunkCache(max_bytes=1024)
    a = np.ones((8, 8, 8), dtype=np.uint8)  # 512 B
    b = np.full((8, 8, 8), 2, dtype=np.uint8)  # 512 B
    c = np.full((8, 8, 8), 3, dtype=np.uint8)  # 512 B

    cache.put(("u", "0", 0, 0, 0), a)
    cache.put(("u", "0", 0, 0, 1), b)
    assert cache.get(("u", "0", 0, 0, 0)) is not None  # refresh a
    cache.put(("u", "0", 0, 0, 2), c)  # should evict b

    assert cache.get(("u", "0", 0, 0, 1)) is None
    assert cache.get(("u", "0", 0, 0, 0)) is not None
    assert cache.get(("u", "0", 0, 0, 2)) is not None
    assert cache.stats.evictions == 1


def test_streaming_zarr_loader_local_ome_zarr(tmp_path: Path) -> None:
    store_path = tmp_path / "scroll_sample.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.attrs["multiscales"] = [{"version": "0.4"}]
    arr = root.create_array("0", shape=(64, 128, 128), chunks=(32, 64, 64), dtype="uint8")
    rng = np.random.default_rng(0)
    data = rng.integers(10, 240, size=(64, 128, 128), dtype=np.uint8)
    arr[:] = data

    loader = StreamingZarrSubvolumeLoader(
        ZarrSourceConfig(uri=str(store_path), level="0", roi_origin=(16, 32, 32), roi_shape=(32, 64, 64)),
        CacheConfig(max_bytes=1024 * 1024, prefetch_workers=2),
    )
    sub1 = loader.read_subvolume()
    np.testing.assert_array_equal(sub1, data[16:48, 32:96, 32:96])
    assert loader.cache.stats.misses > 0

    hits_before = loader.cache.stats.hits
    sub2 = loader.read_subvolume()
    np.testing.assert_array_equal(sub2, sub1)
    assert loader.cache.stats.hits > hits_before


def test_hann_window_3d_separable_and_seam_suppression() -> None:
    win = compute_hann_window_3d((32, 64, 64), min_weight=1e-3)
    assert win.shape == (32, 64, 64)
    assert torch.isclose(win[16, 32, 32], torch.tensor(1.0), atol=1e-2)
    # Separable raised window preserves Y-X taper even at z=0
    assert float(win[0, 32, 32]) > float(win[0, 0, 0])
    assert float(win.min()) > 0.0

    # Smooth Z-Hann projection removes boundary slice dominance
    vol3d = np.ones((16, 32, 32), dtype=np.float32) * 0.2
    vol3d[0, :, :] = 1.0  # noisy outer slice
    proj = project_surface_z_hann(vol3d)
    assert float(proj.mean()) < 0.25

    # Synthetic seam test: step discontinuities at stride=32 vs smooth field
    y, x = np.mgrid[0:128, 0:128]
    smooth = 0.5 + 0.2 * np.sin(y / 10.0) * np.cos(x / 10.0)
    seamed = smooth.copy()
    seamed[32:64, :] += 0.25
    seamed[:, 64:96] -= 0.25

    metrics = compute_seam_metrics(smooth, seamed, tile_shape=(64, 64), stride=(32, 32))
    assert metrics["hann_seam_to_interior_ratio"] < metrics["unblended_seam_to_interior_ratio"]
    assert metrics["seam_discontinuity_reduction_pct"] > 50.0


def test_ink_accuracy_metrics() -> None:
    gt = np.array([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.uint8)
    prob = np.array([[0.9, 0.8, 0.2, 0.1], [0.7, 0.3, 0.1, 0.2]], dtype=np.float32)
    m = compute_ink_accuracy_metrics(gt, prob, threshold=0.5)
    assert m["roc_auc"] == 1.0
    assert m["dice_f1"] == 1.0
    assert m["f0_5"] == 1.0
    assert m["iou"] == 1.0


def test_pretrained_checkpoint_strict_load() -> None:
    ckpt_path = Path("checkpoints/ink_canonical_2um/r152_3ddec_v2_l5_epoch13.ckpt")
    if not ckpt_path.exists():
        return
    model, meta = VillaInkResNet3D.from_villa_checkpoint(ckpt_path, strict=True)
    assert meta["epoch"] == 13
    assert meta["global_step"] == 113246
    assert meta["num_state_dict_tensors"] == 976
    x = torch.zeros((1, 1, 16, 64, 64), dtype=torch.float32)
    with torch.inference_mode():
        y = model.eval()(x)
    assert y.shape == (1, 1, 64, 64)


def test_pipeline_end_to_end_smoke() -> None:
    vol = np.random.default_rng(7).integers(0, 255, size=(64, 128, 128), dtype=np.uint8)
    model = VillaInkResNet3D(with_norm=True)
    cfg = InferenceConfig(
        tile_shape=(32, 64, 64),
        stride=(16, 32, 32),
        batch_size=4,
        amp_dtype="bfloat16" if torch.cuda.is_available() else "float32",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_channels_last_3d=True,
        seed=123,
    )
    pipe = MixedPrecisionInkPipeline(model, cfg, pin_staging_buffer=False)
    res = pipe.run(vol, warmup_batches=0)
    assert res.blended_surface_2d.shape == (128, 128)
    assert res.metrics["num_tiles"] == 27
    assert res.metrics["within_vram_budget"] is True
    assert res.metrics["pin_staging_buffer"] is False
    assert res.metrics["use_channels_last_3d"] is True
    assert res.metrics["seed"] == 123


def test_vesuvius_infer_exports_ome_zarr_tiff_and_png(tmp_path: Path) -> None:
    ckpt_path = Path("checkpoints/ink_canonical_2um/r152_3ddec_v2_l5_epoch13.ckpt")
    if not ckpt_path.exists():
        return

    from PIL import Image

    src_zarr = tmp_path / "input_scroll.zarr"
    root = zarr.open_group(str(src_zarr), mode="w")
    arr = root.create_array("2", shape=(64, 128, 128), chunks=(64, 128, 128), dtype="uint8")
    arr[:] = np.random.default_rng(11).integers(20, 220, size=(64, 128, 128), dtype=np.uint8)

    out_dir = tmp_path / "infer_out"
    summary = run_inference(
        uri=str(src_zarr),
        level="2",
        full_level=True,
        output_dir=out_dir,
        batch_size=4,
        amp_dtype="bfloat16" if torch.cuda.is_available() else "float32",
        checkpoint_path=ckpt_path,
    )

    zarr_out = Path(summary["exports"]["ome_zarr"])
    tiff_out = Path(summary["exports"]["primary_tiff_16bit"])
    png_out = Path(summary["exports"]["primary_png_surface"])

    assert zarr_out.exists()
    assert tiff_out.exists()
    assert png_out.exists()
    assert (out_dir / "inference_summary.json").exists()

    out_group = zarr.open_group(str(zarr_out), mode="r")
    assert "multiscales" in out_group.attrs
    assert out_group["0"].shape == (64, 128, 128)
    assert out_group["surface_prob_2d"].shape == (128, 128)

    with Image.open(tiff_out) as im:
        u16 = np.asarray(im)
        assert u16.dtype == np.uint16
        assert u16.shape == (128, 128)
