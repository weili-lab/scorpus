"""Zarr backend writers for federated and aggregate materialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from ..chunk_translation import ChunkBundle


_ROW_OFFSET_CHUNK = 4_096
_ROW_OFFSET_SHARD = 1_048_576
_NNZ_CHUNK = 131_072
_NNZ_SHARD = 16_777_216


def _create_sharded_array(group: Any, name: str, *, shape: tuple[int, ...], dtype: str):
    from zarr.codecs import BloscCodec

    chunks = (_ROW_OFFSET_CHUNK,) if name == "row_offsets" else (_NNZ_CHUNK,)
    shards = (_ROW_OFFSET_SHARD,) if name == "row_offsets" else (_NNZ_SHARD,)
    return group.create_array(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        shards=shards,
        compressors=[BloscCodec(cname="lz4")],
    )


def _open_zarr_state(
    csr_path: Path,
    bundle: ChunkBundle,
    *,
    append_existing: bool,
) -> dict[str, Any]:
    import zarr

    if append_existing and csr_path.exists():
        csr_zarr = zarr.open_group(str(csr_path), mode="a")
        indices = cast(Any, csr_zarr["indices"])
        counts = cast(Any, csr_zarr["counts"])
        row_offsets = cast(Any, csr_zarr["row_offsets"])
        current_nnz = int(indices.shape[0])
        current_rows = int(row_offsets.shape[0]) - 1
        if counts.shape[0] != current_nnz or current_rows < 0:
            raise ValueError("aggregate Zarr artifacts have inconsistent shapes")
        if int(row_offsets[-1]) != current_nnz:
            raise ValueError("aggregate Zarr row_offsets[-1] does not match stored nnz")
    else:
        initial_nnz = max(len(bundle.indices), 1)
        csr_zarr = zarr.open_group(str(csr_path), mode="w", zarr_format=3)
        _create_sharded_array(csr_zarr, "indices", shape=(initial_nnz,), dtype="i4")
        _create_sharded_array(csr_zarr, "counts", shape=(initial_nnz,), dtype="i4")
        _create_sharded_array(
            csr_zarr,
            "row_offsets",
            shape=(bundle.row_count + 1,),
            dtype="i8",
        )
        current_nnz = 0
        current_rows = 0

    return {
        "csr_zarr": csr_zarr,
        "global_nnz": current_nnz,
        "row_count": current_rows,
    }


def _write_zarr(
    *,
    bundle: ChunkBundle,
    paths: dict[str, Path],
    _writer_state: dict[str, Any] | None,
    _is_last_chunk: bool,
    append_existing: bool,
) -> tuple[dict[str, Path], dict[str, Any] | None]:
    csr_path = paths["csr"]
    csr_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        append_existing
        and _writer_state is None
        and not csr_path.exists()
        and bundle.row_count
        and int(bundle.global_row_index[0]) != 0
    ):
        raise FileNotFoundError(
            f"aggregate Zarr CSR artifact not found for append: {csr_path}"
        )
    if _writer_state is None:
        _writer_state = _open_zarr_state(
            csr_path,
            bundle,
            append_existing=append_existing,
        )

    assert bundle.indptr[0] == 0, f"chunk indptr[0] == {bundle.indptr[0]}, expected 0"
    current_nnz = _writer_state["global_nnz"]
    current_rows = _writer_state["row_count"]
    chunk_nnz = len(bundle.indices)
    chunk_rows = bundle.row_count
    if append_existing and chunk_rows and int(bundle.global_row_index[0]) != current_rows:
        raise ValueError(
            "aggregate Zarr append expected next global row "
            f"{current_rows}, got {int(bundle.global_row_index[0])}"
        )

    row_offsets = _writer_state["csr_zarr"]["row_offsets"]
    ro_end = current_rows + chunk_rows + 1
    if ro_end > row_offsets.shape[0]:
        row_offsets.resize((ro_end,))
    row_offsets[current_rows:ro_end] = bundle.indptr + current_nnz

    needed_nnz = current_nnz + chunk_nnz
    indices = _writer_state["csr_zarr"]["indices"]
    counts = _writer_state["csr_zarr"]["counts"]
    if needed_nnz > indices.shape[0]:
        indices.resize((needed_nnz,))
        counts.resize((needed_nnz,))
    indices[current_nnz:needed_nnz] = bundle.indices
    counts[current_nnz:needed_nnz] = bundle.counts

    _writer_state["global_nnz"] = needed_nnz
    _writer_state["row_count"] = current_rows + chunk_rows

    if _is_last_chunk:
        indices.resize((_writer_state["global_nnz"],))
        counts.resize((_writer_state["global_nnz"],))
        row_offsets.resize((_writer_state["row_count"] + 1,))
        return paths, None
    return paths, _writer_state


def write_zarr_federated(
    bundle: ChunkBundle,
    dataset_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
) -> tuple[dict[str, Path], dict[str, Any] | None]:
    paths = {
        "csr": matrix_root / f"{dataset_id}-csr.zarr",
    }
    return _write_zarr(
        bundle=bundle,
        paths=paths,
        _writer_state=_writer_state,
        _is_last_chunk=_is_last_chunk,
        append_existing=False,
    )


def write_zarr_aggregate(
    bundle: ChunkBundle,
    dataset_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
) -> tuple[dict[str, Path], dict[str, Any] | None]:
    paths = {
        "csr": matrix_root / "aggregated-csr.zarr",
    }
    return _write_zarr(
        bundle=bundle,
        paths=paths,
        _writer_state=_writer_state,
        _is_last_chunk=_is_last_chunk,
        append_existing=True,
    )
