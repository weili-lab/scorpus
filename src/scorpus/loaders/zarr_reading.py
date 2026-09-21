"""Optional Zarr read-engine helpers for CSR readers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "ZARR_READ_ENGINES",
    "CSRArrayHandles",
    "normalize_zarr_read_engine",
    "open_csr_arrays",
]

ZARR_READ_ENGINES: tuple[str, ...] = (
    "zarr-python",
    "zarrs",
    "tensorstore",
)


@dataclass(frozen=True)
class CSRArrayHandles:
    row_offsets: Any
    indices: Any
    counts: Any


def normalize_zarr_read_engine(raw: str | None) -> str:
    engine = "zarr-python" if raw is None else str(raw).strip()
    if engine not in ZARR_READ_ENGINES:
        raise ValueError(
            f"Unsupported zarr read engine '{engine}'. "
            f"Supported: {list(ZARR_READ_ENGINES)}"
        )
    return engine


def _normalize_scalar_index(index: Any, size: int) -> int:
    resolved = int(index)
    if resolved < 0:
        resolved += size
    if resolved < 0 or resolved >= size:
        raise IndexError(f"index {index} out of range for axis of size {size}")
    return resolved


def _normalize_slice(index: slice, size: int) -> slice:
    start, stop, step = index.indices(size)
    return slice(start, stop, step)


class _TensorStoreArray:
    def __init__(self, store: Any):
        self._store = store
        self.shape = tuple(int(v) for v in store.shape)

    def __getitem__(self, index: Any):
        if len(self.shape) != 1:
            raise ValueError("TensorStore CSR helper expects 1D arrays")
        if isinstance(index, slice):
            resolved = _normalize_slice(index, self.shape[0])
            return np.asarray(self._store[resolved].read().result())
        resolved = _normalize_scalar_index(index, self.shape[0])
        return np.asarray(self._store[resolved].read().result()).item()


def _open_zarr_python_array(path: str | Path, array_name: str, *, use_zarrs: bool):
    import zarr

    config_set = None
    if use_zarrs:
        __import__("zarrs")

        config_set = zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
    try:
        group = zarr.open_group(str(path), mode="r")
        return group[array_name]
    finally:
        if config_set is not None:
            config_set.__exit__(None, None, None)


def _open_tensorstore_array(path: str | Path, array_name: str):
    ts = __import__("tensorstore")

    path = Path(path)
    if not (path / "zarr.json").exists():
        raise ValueError(
            "TensorStore route currently expects a Zarr v3 store containing zarr.json; "
            f"got {path}"
        )
    store = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(path)},
            "path": array_name,
        },
        read=True,
    ).result()
    return _TensorStoreArray(store)


def open_csr_arrays(
    offsets_path: str | Path,
    indices_path: str | Path,
    counts_path: str | Path,
    *,
    read_engine: str = "zarr-python",
) -> CSRArrayHandles:
    engine = normalize_zarr_read_engine(read_engine)
    if engine == "zarr-python":
        return CSRArrayHandles(
            row_offsets=_open_zarr_python_array(offsets_path, "row_offsets", use_zarrs=False),
            indices=_open_zarr_python_array(indices_path, "indices", use_zarrs=False),
            counts=_open_zarr_python_array(counts_path, "counts", use_zarrs=False),
        )
    if engine == "zarrs":
        return CSRArrayHandles(
            row_offsets=_open_zarr_python_array(offsets_path, "row_offsets", use_zarrs=True),
            indices=_open_zarr_python_array(indices_path, "indices", use_zarrs=True),
            counts=_open_zarr_python_array(counts_path, "counts", use_zarrs=True),
        )
    return CSRArrayHandles(
        row_offsets=_open_tensorstore_array(offsets_path, "row_offsets"),
        indices=_open_tensorstore_array(indices_path, "indices"),
        counts=_open_tensorstore_array(counts_path, "counts"),
    )
