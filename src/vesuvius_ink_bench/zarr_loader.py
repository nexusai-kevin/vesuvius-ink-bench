"""Streaming OME-Zarr sub-volume loader with host DDR5 RAM LRU chunk caching.

Compatible with ScrollPrize/villa OME-Zarr conventions and public AWS Open Data
Herculaneum scroll buckets (e.g. ``s3://vesuvius-challenge-open-data/``).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fsspec
import numpy as np
import zarr

from vesuvius_ink_bench.config import CacheConfig, ZarrSourceConfig


ChunkKey = Tuple[str, str, int, int, int]


@dataclass(slots=True)
class CacheStats:
    """Telemetry for host DDR5 RAM chunk caching."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_cached: int = 0
    bytes_fetched_remote: int = 0
    bytes_served_from_cache: int = 0
    remote_fetch_seconds: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
            "bytes_cached": self.bytes_cached,
            "bytes_fetched_remote": self.bytes_fetched_remote,
            "bytes_served_from_cache": self.bytes_served_from_cache,
            "remote_fetch_seconds": round(self.remote_fetch_seconds, 4),
        }


class DDR5ChunkCache:
    """Thread-safe LRU cache residing in host DDR5 RAM for 3D Zarr chunks."""

    def __init__(self, max_bytes: int = 16 * 1024**3) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._entries: OrderedDict[ChunkKey, np.ndarray] = OrderedDict()
        self.stats = CacheStats()

    def get(self, key: ChunkKey) -> Optional[np.ndarray]:
        with self._lock:
            chunk = self._entries.get(key)
            if chunk is None:
                self.stats.misses += 1
                return None
            self._entries.move_to_end(key)
            self.stats.hits += 1
            self.stats.bytes_served_from_cache += chunk.nbytes
            return chunk

    def put(self, key: ChunkKey, chunk: np.ndarray, *, fetch_seconds: float = 0.0) -> None:
        chunk_contig = np.ascontiguousarray(chunk)
        nbytes = chunk_contig.nbytes
        if nbytes > self.max_bytes:
            return

        with self._lock:
            if key in self._entries:
                old = self._entries.pop(key)
                self.stats.bytes_cached -= old.nbytes
            while self.stats.bytes_cached + nbytes > self.max_bytes and self._entries:
                _, evicted = self._entries.popitem(last=False)
                self.stats.bytes_cached -= evicted.nbytes
                self.stats.evictions += 1

            self._entries[key] = chunk_contig
            self.stats.bytes_cached += nbytes
            self.stats.bytes_fetched_remote += nbytes
            self.stats.remote_fetch_seconds += fetch_seconds

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.stats.bytes_cached = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def _open_zarr_group(uri: str, anon: bool = True) -> zarr.Group:
    """Open a local, HTTPS, or S3 OME-Zarr group without downloading the dataset."""
    if uri.startswith("s3://"):
        fs = fsspec.filesystem("s3", anon=anon)
        store = fs.get_mapper(uri.rstrip("/") + "/")
        return zarr.open_group(store, mode="r")
    if uri.startswith(("http://", "https://")):
        store = fsspec.get_mapper(uri.rstrip("/") + "/")
        return zarr.open_group(store, mode="r")
    return zarr.open_group(uri, mode="r")


class StreamingZarrSubvolumeLoader:
    """Streams chunk-aligned 3D ROIs from OME-Zarr volumes into host DDR5 RAM."""

    def __init__(
        self,
        source: ZarrSourceConfig,
        cache_config: Optional[CacheConfig] = None,
        *,
        cache: Optional[DDR5ChunkCache] = None,
        zarr_group: Optional[zarr.Group] = None,
    ) -> None:
        self.source = source
        self.cache_config = cache_config or CacheConfig()
        self.cache = cache or DDR5ChunkCache(max_bytes=self.cache_config.max_bytes)
        self.group = zarr_group if zarr_group is not None else _open_zarr_group(source.uri, anon=source.anon)

        if source.level not in self.group:
            available = list(self.group.array_keys())
            raise KeyError(
                f"Resolution level '{source.level}' not found in {source.uri}. Available: {available}"
            )
        self.array: zarr.Array = self.group[source.level]
        self.shape: Tuple[int, int, int] = tuple(int(x) for x in self.array.shape)
        self.chunks: Tuple[int, int, int] = tuple(int(x) for x in self.array.chunks)
        self.dtype = np.dtype(self.array.dtype)

    def metadata_summary(self) -> Dict[str, Any]:
        """Return OME-Zarr volume metadata without triggering bulk data reads."""
        attrs = dict(self.group.attrs)
        return {
            "uri": self.source.uri,
            "level": self.source.level,
            "shape": self.shape,
            "chunks": self.chunks,
            "dtype": str(self.dtype),
            "multiscales_present": "multiscales" in attrs,
        }

    def _fetch_chunk(self, cz: int, cy: int, cx: int) -> np.ndarray:
        key: ChunkKey = (self.source.uri, self.source.level, cz, cy, cx)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        cd, ch, cw = self.chunks
        z0, y0, x0 = cz * cd, cy * ch, cx * cw
        z1 = min(z0 + cd, self.shape[0])
        y1 = min(y0 + ch, self.shape[1])
        x1 = min(x0 + cw, self.shape[2])

        t0 = time.perf_counter()
        data = np.asarray(self.array[z0:z1, y0:y1, x0:x1], dtype=self.dtype)
        dt = time.perf_counter() - t0
        self.cache.put(key, data, fetch_seconds=dt)
        return data

    def _intersecting_chunks(
        self,
        origin: Tuple[int, int, int],
        roi_shape: Tuple[int, int, int],
    ) -> List[Tuple[int, int, int]]:
        oz, oy, ox = origin
        dz, dy, dx = roi_shape
        cd, ch, cw = self.chunks

        cz0, cz1 = oz // cd, (oz + dz - 1) // cd
        cy0, cy1 = oy // ch, (oy + dy - 1) // ch
        cx0, cx1 = ox // cw, (ox + dx - 1) // cw

        coords: List[Tuple[int, int, int]] = []
        for cz in range(cz0, cz1 + 1):
            for cy in range(cy0, cy1 + 1):
                for cx in range(cx0, cx1 + 1):
                    coords.append((cz, cy, cx))
        return coords

    def read_subvolume(
        self,
        origin: Optional[Tuple[int, int, int]] = None,
        roi_shape: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Stream a 3D sub-volume ``(D, H, W)`` using concurrent chunk fetch + DDR5 cache."""
        oz, oy, ox = origin or self.source.roi_origin
        dz, dy, dx = roi_shape or self.source.roi_shape

        if oz < 0 or oy < 0 or ox < 0:
            raise ValueError(f"ROI origin must be non-negative, got {(oz, oy, ox)}")
        if oz + dz > self.shape[0] or oy + dy > self.shape[1] or ox + dx > self.shape[2]:
            raise ValueError(
                f"ROI {(oz, oy, ox)} + {(dz, dy, dx)} exceeds volume shape {self.shape}"
            )

        coords = self._intersecting_chunks((oz, oy, ox), (dz, dy, dx))
        workers = min(self.cache_config.prefetch_workers, max(1, len(coords)))

        fetched: Dict[Tuple[int, int, int], np.ndarray] = {}
        if workers == 1:
            for c in coords:
                fetched[c] = self._fetch_chunk(*c)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._fetch_chunk, *c): c for c in coords}
                for fut, c in futures.items():
                    fetched[c] = fut.result()

        out = np.empty((dz, dy, dx), dtype=self.dtype)
        cd, ch, cw = self.chunks
        for (cz, cy, cx), chunk in fetched.items():
            gz0, gy0, gx0 = cz * cd, cy * ch, cx * cw
            gz1, gy1, gx1 = gz0 + chunk.shape[0], gy0 + chunk.shape[1], gx0 + chunk.shape[2]

            iz0, iz1 = max(oz, gz0), min(oz + dz, gz1)
            iy0, iy1 = max(oy, gy0), min(oy + dy, gy1)
            ix0, ix1 = max(ox, gx0), min(ox + dx, gx1)

            out[iz0 - oz : iz1 - oz, iy0 - oy : iy1 - oy, ix0 - ox : ix1 - ox] = chunk[
                iz0 - gz0 : iz1 - gz0,
                iy0 - gy0 : iy1 - gy0,
                ix0 - gx0 : ix1 - gx0,
            ]
        return out

    def find_nonempty_roi(
        self,
        candidate_origins: Iterable[Tuple[int, int, int]],
        roi_shape: Tuple[int, int, int],
        min_nonzero_fraction: float = 0.05,
    ) -> Tuple[Tuple[int, int, int], np.ndarray]:
        """Probe candidate origins until a papyrus-bearing sub-volume is found."""
        last_origin = None
        last_vol = None
        for origin in candidate_origins:
            vol = self.read_subvolume(origin=origin, roi_shape=roi_shape)
            frac = float(np.count_nonzero(vol)) / float(vol.size)
            last_origin, last_vol = origin, vol
            if frac >= min_nonzero_fraction:
                return origin, vol
        assert last_origin is not None and last_vol is not None
        return last_origin, last_vol
