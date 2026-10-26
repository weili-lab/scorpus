"""Lance backend writers for federated and aggregate materialization."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any

import pyarrow as pa

from ..chunk_translation import ChunkBundle


HEAVY_CELL_SCHEMA = pa.schema(
    [
        pa.field("global_row_index", pa.int64()),
        pa.field("expressed_gene_indices", pa.list_(pa.int32())),
        pa.field("expression_counts", pa.list_(pa.int32())),
    ]
)


def _import_lance():
    try:
        import lance
    except ImportError as exc:
        raise ImportError(
            "lance backend requires the Lance library; install lance in the selected runtime"
        ) from exc
    return lance


def _bundle_to_table(bundle: ChunkBundle) -> pa.Table:
    offsets = pa.array(bundle.indptr.astype("int32", copy=False), type=pa.int32())
    value_type = pa.from_numpy_dtype(bundle.counts.dtype)
    schema = HEAVY_CELL_SCHEMA.set(2, pa.field("expression_counts", pa.list_(value_type)))
    return pa.table(
        {
            "global_row_index": pa.array(bundle.global_row_index, type=pa.int64()),
            "expressed_gene_indices": pa.ListArray.from_arrays(
                offsets,
                pa.array(bundle.indices, type=pa.int32()),
            ),
            "expression_counts": pa.ListArray.from_arrays(
                offsets,
                pa.array(bundle.counts, type=value_type),
            ),
        },
        schema=schema,
    )


def _write_lance(
    *,
    bundle: ChunkBundle,
    lance_path: Path,
    _writer_state: dict[str, Any] | None,
    _is_last_chunk: bool,
    append_existing: bool,
) -> tuple[dict[str, Path], dict[str, Any] | None]:
    """Stream all chunks of one table into a single ``lance.write_dataset`` call.

    The first chunk starts the write in a background thread that reads chunks from
    a small queue; the last chunk closes the stream and waits for the commit. The
    table is therefore committed once, as one version with Lance's default ~1M-row
    fragments. Appending chunk by chunk would make one fragment, version and
    transaction file per chunk, and many small fragments slow random ``take``.
    If the caller stops before its last chunk, nothing is committed.
    """
    lance = _import_lance()
    table = _bundle_to_table(bundle)

    if _writer_state is None:
        lance_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "append" if append_existing and lance_path.exists() else "create"
        chunks: queue.Queue = queue.Queue(maxsize=2)
        _writer_state = {"lance_path": str(lance_path), "queue": chunks, "errors": []}

        def batches():
            while (item := chunks.get()) is not _END_OF_STREAM:
                yield from item.to_batches()

        def write():
            try:
                reader = pa.RecordBatchReader.from_batches(table.schema, batches())
                lance.write_dataset(reader, str(lance_path), mode=mode)
            except BaseException as exc:  # handed to the caller's thread by _put / the final join
                _writer_state["errors"].append(exc)

        # Daemon, so a write abandoned by a failed caller cannot keep the process alive.
        _writer_state["thread"] = threading.Thread(target=write, daemon=True)
        _writer_state["thread"].start()

    _put(_writer_state, table)
    if not _is_last_chunk:
        return {"cells": lance_path}, _writer_state
    _put(_writer_state, _END_OF_STREAM)
    _writer_state["thread"].join()
    if _writer_state["errors"]:
        raise _writer_state["errors"][0]
    return {"cells": lance_path}, None


_END_OF_STREAM = object()


def _put(state: dict[str, Any], item: Any) -> None:
    """Queue one item for the writer thread; raise its error if it has died."""
    while True:
        try:
            state["queue"].put(item, timeout=1)
            return
        except queue.Full:
            if not state["thread"].is_alive():
                raise state["errors"][0]


def write_lance_federated(
    bundle: ChunkBundle,
    dataset_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
) -> tuple[dict[str, Path], dict[str, Any] | None]:
    return _write_lance(
        bundle=bundle,
        lance_path=matrix_root / f"{dataset_id}.lance",
        _writer_state=_writer_state,
        _is_last_chunk=_is_last_chunk,
        append_existing=False,
    )


def write_lance_aggregate(
    bundle: ChunkBundle,
    dataset_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
) -> tuple[dict[str, Path], dict[str, Any] | None]:
    return _write_lance(
        bundle=bundle,
        lance_path=matrix_root / "aggregated-cells.lance",
        _writer_state=_writer_state,
        _is_last_chunk=_is_last_chunk,
        append_existing=True,
    )
