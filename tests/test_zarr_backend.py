from __future__ import annotations

import json
from typing import Any, cast

import numpy as np

from scorpus.materializers.backends.zarr import write_zarr_aggregate
from scorpus.materializers.chunk_translation import ChunkBundle


def _bundle(global_start: int, indptr: list[int], indices: list[int]) -> ChunkBundle:
    return ChunkBundle(
        global_row_index=np.arange(
            global_start,
            global_start + len(indptr) - 1,
            dtype=np.int64,
        ),
        row_sums=np.ones(len(indptr) - 1, dtype=np.float64),
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int32),
        counts=np.arange(1, len(indices) + 1, dtype=np.int32),
        row_count=len(indptr) - 1,
    )


def test_zarr_writer_uses_single_v3_sharded_csr_group(tmp_path) -> None:
    import zarr

    matrix_root = tmp_path / "matrix"
    first = _bundle(0, [0, 2, 3], [0, 3, 1])
    second = _bundle(2, [0, 1, 3], [2, 0, 4])

    paths, state = write_zarr_aggregate(
        first,
        "mock",
        matrix_root,
        _writer_state=None,
        _is_last_chunk=False,
    )
    paths, state = write_zarr_aggregate(
        second,
        "mock",
        matrix_root,
        _writer_state=state,
        _is_last_chunk=True,
    )

    assert state is None
    assert paths == {"csr": matrix_root / "aggregated-csr.zarr"}
    assert (matrix_root / "aggregated-csr.zarr").is_dir()
    assert not (matrix_root / "aggregated-indices.zarr").exists()

    group = zarr.open_group(str(matrix_root / "aggregated-csr.zarr"), mode="r")
    row_offsets = cast(Any, group["row_offsets"])
    indices = cast(Any, group["indices"])
    counts = cast(Any, group["counts"])
    assert group.metadata.zarr_format == 3
    assert indices.shards is not None
    np.testing.assert_array_equal(row_offsets[:], [0, 2, 3, 4, 6])
    np.testing.assert_array_equal(indices[:], [0, 3, 1, 2, 0, 4])
    np.testing.assert_array_equal(counts[:], [1, 2, 3, 1, 2, 3])

    row_offsets_meta = json.loads((matrix_root / "aggregated-csr.zarr" / "row_offsets" / "zarr.json").read_text())
    indices_meta = json.loads((matrix_root / "aggregated-csr.zarr" / "indices" / "zarr.json").read_text())
    assert row_offsets_meta["codecs"][0]["configuration"]["chunk_shape"] == [4_096]
    assert indices_meta["codecs"][0]["configuration"]["chunk_shape"] == [131_072]
    assert row_offsets_meta["codecs"][0]["configuration"]["codecs"][1]["name"] == "blosc"
    assert indices_meta["codecs"][0]["configuration"]["codecs"][1]["name"] == "blosc"
    assert row_offsets_meta["codecs"][0]["configuration"]["codecs"][1]["configuration"]["cname"] == "lz4"
    assert indices_meta["codecs"][0]["configuration"]["codecs"][1]["configuration"]["cname"] == "lz4"
