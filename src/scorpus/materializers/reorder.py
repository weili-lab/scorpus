"""Rewrite a standalone corpus with its cells in a new order.

Every per-cell table moves together: the expression table, each ``extra/*.lance``
table, ``obs.parquet`` and ``statistics.parquet``. ``var.parquet`` and the extra
feature tables are copied unchanged. With ``pairing.write_order`` as the order,
each group becomes one contiguous block, shuffled within, so ``Sampler(...,
run_length=r)`` reads runs of neighbouring rows. An order that leaves out rows
(``write_order(..., keep=...)``) writes a corpus of only those cells.

Two ways to gather the rows (``method``):

- ``"random"``: each worker reads its part of the new order with random reads. Fast on
  node-local NVMe or RAM; slow on network storage, where every row is a request.
- ``"bins"``: two sequential passes through a temporary folder. Pass 1 reads the source in
  consecutive row ranges and appends every row to the bin (an Arrow IPC file) of its new
  position. Pass 2 writes each chunk of the new order from its bins, one bin at a time, each
  put in new order. The bins are uncompressed by default (about 8 bytes per nonzero value, 2-3x
  the Lance table) on temporary disk. ``bin_compression`` ("lz4" or "zstd") compresses the IPC
  buffers; the bin files stay plain Arrow IPC. A worker holds about one bin of ``BIN_ROWS`` rows in memory.
"""

from __future__ import annotations

import itertools
import math
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import yaml

from ..loaders.expression import FederatedLanceReader, LanceDatasetEntry

# Rows gathered per lean-reader read (one sorted take); also one record batch of the new table.
BLOCK_ROWS = 65_536
# Most rows of the new order in one chunk; each chunk is written by one worker as one Lance fragment.
FRAGMENT_ROWS = 1024 * 1024
# New rows per temporary bin (method "bins"); a worker holds about one bin in memory.
BIN_ROWS = 65_536


def reorder_corpus(source_root, dest_root, order, row_order_note: dict, layout: str = "two_column",
                   workers: int = 1, method: str = "random", temp_dir=None,
                   bin_compression: str | None = None) -> Path:
    """Write the corpus at ``source_root`` to ``dest_root`` with rows ``order``.

    ``order`` gives the source row for each new position (``new = old[order]``),
    as returned by ``pairing.write_order``. It may leave out source rows (a filtered
    rewrite); the new corpus then has ``len(order)`` cells. ``row_order_note`` (for
    example the grouping columns, seed and original source path) is stored as
    ``row_order`` in the new ``corpus.yaml``.

    Lance tables get ``global_row_index`` = new position and ``source_row`` =
    original row. ``workers`` processes read and write chunks of the new order in
    parallel, each holding one lean reader (about 5-15 GB for a large corpus).

    ``method`` is ``"random"`` or ``"bins"`` (module docstring); ``"bins"`` needs ``temp_dir``.
    ``bin_compression`` is ``None``, ``"lz4"`` or ``"zstd"`` (method ``"bins"`` only).

    ``layout="packed"`` writes the expression table as one int64 list column
    ``packed`` with values ``(count_bits << 16) | gene`` (see
    ``loaders.expression._unpack_genes_and_counts``); extra tables, obs, var and
    statistics are written the same way in both layouts.
    """
    source_root, dest_root = Path(source_root), Path(dest_root)
    order = np.asarray(order, dtype=np.int64)
    manifest = yaml.safe_load((source_root / "corpus.yaml").read_text())
    n_cells, n_genes = manifest["shape"]
    if layout == "packed" and n_genes >= 2**16:
        raise ValueError(f"Cannot pack {n_genes} genes: gene indices must fit in 16 bits")
    dest_root.mkdir(parents=True)

    # 1. Expression and extra tables, gathered in the new order through the lean reader.
    source_expression = LanceDatasetEntry(
        "expression", 0, n_cells, source_root / "expression.lance",
        manifest.get("expression_layout", "two_column"), manifest["dtype"],
    )
    _write_reordered_table(source_expression, dest_root / "expression.lance", order,
                           packed=layout == "packed", workers=workers, method=method, temp_dir=temp_dir,
                           bin_compression=bin_compression)
    extra = manifest.get("extra", {})
    if extra:
        (dest_root / "extra").mkdir()
    for name in extra:
        source_extra = LanceDatasetEntry(name, 0, n_cells, source_root / "extra" / f"{name}.lance")
        _write_reordered_table(source_extra, dest_root / "extra" / f"{name}.lance", order,
                               packed=False, workers=workers, method=method, temp_dir=temp_dir,
                               bin_compression=bin_compression)
        shutil.copyfile(source_root / "extra" / f"{name}_features.parquet",
                        dest_root / "extra" / f"{name}_features.parquet")

    # 2. Per-cell metadata in the new order; gene metadata unchanged. Size factors keep
    #    their source values, also in a filtered rewrite.
    pd.read_parquet(source_root / "obs.parquet").iloc[order].to_parquet(dest_root / "obs.parquet")
    if manifest["expression_kind"] != "non_count":
        statistics = pd.read_parquet(source_root / "statistics.parquet")
        statistics.iloc[order].to_parquet(dest_root / "statistics.parquet", index=False)
    shutil.copyfile(source_root / "var.parquet", dest_root / "var.parquet")

    # 3. The manifest last, so an interrupted rewrite is not a loadable corpus.
    manifest["shape"] = [len(order), n_genes]
    manifest["row_order"] = row_order_note
    if layout == "packed":
        manifest["expression_layout"] = "packed"  # counts dtype stays in `dtype`
    else:
        manifest.pop("expression_layout", None)
    (dest_root / "corpus.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return dest_root


def _write_reordered_table(source: LanceDatasetEntry, dest_path: Path, order: np.ndarray, packed: bool,
                           workers: int, method: str, temp_dir, bin_compression: str | None) -> None:
    """Write the rows ``order`` of one table as fragments, then commit them once, in new-row order.

    The new order is cut into chunks of consecutive new rows, at least one per worker and at most
    ``FRAGMENT_ROWS`` rows each; every chunk becomes one fragment.
    """
    chunk_rows = min(FRAGMENT_ROWS, math.ceil(len(order) / workers))
    if method == "random":
        written = _write_through_random_reads(source, dest_path, order, packed, workers, chunk_rows)
    elif method == "bins":
        written = _write_through_bins(source, dest_path, order, packed, workers, chunk_rows,
                                      Path(temp_dir) / f"{dest_path.stem}_bins", bin_compression)
    else:
        raise ValueError(f"Unknown method {method!r}: use 'random' or 'bins'")
    schema, _ = written[0]  # the same for every chunk of this table
    fragments = [fragment for _, worker_fragments in written for fragment in worker_fragments]
    _commit_in_order(dest_path, schema, fragments, len(order))


def _in_processes(function, tasks: list[tuple], workers: int) -> list:
    """``[function(*task) for task in tasks]`` in ``workers`` processes; ``workers=1`` runs here.

    Processes are spawned, not forked, because Lance is not fork-safe.
    """
    if workers == 1:
        return [function(*task) for task in tasks]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(function, *zip(*tasks)))


def _write_through_random_reads(source: LanceDatasetEntry, dest_path: Path, order: np.ndarray, packed: bool,
                                workers: int, chunk_rows: int) -> list[tuple[pa.Schema, list]]:
    """Write the rows ``order`` in chunks of ``chunk_rows`` new rows; returns each worker's fragments.

    Each worker writes a contiguous range of chunks with one lean reader, so the workers'
    fragments, concatenated, are in row order.
    """
    starts = list(range(0, len(order), chunk_rows))
    chunks = [order[start:start + chunk_rows] for start in starts]
    worker_chunks = np.array_split(np.arange(len(chunks)), min(workers, len(chunks)))
    tasks = [(source, dest_path, [starts[i] for i in ids], [chunks[i] for i in ids], packed)
             for ids in worker_chunks]
    return _in_processes(_write_chunks, tasks, len(tasks))


def _write_through_bins(source: LanceDatasetEntry, dest_path: Path, order: np.ndarray, packed: bool,
                        workers: int, chunk_rows: int, bins_dir: Path,
                        bin_compression: str | None) -> list[tuple[pa.Schema, list]]:
    """Write the rows ``order`` through temporary bins; returns each chunk's fragment.

    Chunk ``c`` holds new rows ``c * chunk_rows`` up to ``(c + 1) * chunk_rows``, cut into bins of
    ``BIN_ROWS`` new rows, so no bin crosses a chunk boundary. Each bin is one file per pass-1
    worker, so no two processes write the same file.
    """
    bins_dir.mkdir()
    # New position of every source row; -1 for rows the order leaves out.
    destinations = np.full(source.global_end, -1, dtype=np.int64)
    destinations[order] = np.arange(len(order))

    # Pass 1: one consecutive range of source rows per worker, read in order.
    bounds = np.linspace(0, len(destinations), workers + 1).astype(np.int64)
    _in_processes(_fill_bins, [(source, first, destinations[first:stop], packed, bins_dir, worker, chunk_rows, BIN_ROWS,
                                bin_compression)
                               for worker, (first, stop) in enumerate(zip(bounds[:-1], bounds[1:]))], workers)

    # Pass 2: one task per chunk; it streams the chunk's bins, in order, into one fragment.
    chunk_lengths = [min(chunk_rows, len(order) - start) for start in range(0, len(order), chunk_rows)]
    tasks = [(bins_dir, dest_path, chunk, math.ceil(length / BIN_ROWS)) for chunk, length in enumerate(chunk_lengths)]
    written = _in_processes(_write_chunk_from_bins, tasks, workers)
    shutil.rmtree(bins_dir)
    return written


def _fill_bins(source: LanceDatasetEntry, first_row: int, destinations: np.ndarray, packed: bool,
               bins_dir: Path, worker: int, chunk_rows: int, bin_rows: int,
               bin_compression: str | None) -> None:
    """Read source rows ``first_row, first_row + 1, ...`` in blocks and append each to the bin of its new position.

    ``destinations[i]`` is the new position of source row ``first_row + i``, or -1 to leave it out.
    ``chunk_rows``, ``bin_rows`` and ``bin_compression`` come from the main process (a spawned worker re-imports this module).
    """
    reader = FederatedLanceReader([source])
    bin_options = pa.ipc.IpcWriteOptions(compression=bin_compression)  # the reader detects compression itself
    bin_files = {}
    for block_start in range(0, len(destinations), BLOCK_ROWS):
        block_destinations = destinations[block_start:block_start + BLOCK_ROWS]
        source_rows = np.arange(first_row + block_start, first_row + block_start + len(block_destinations))
        block = _record_batch(reader.read_expression_flat(source_rows), block_destinations, source_rows, packed)
        kept = block_destinations >= 0
        chunks = block_destinations // chunk_rows
        bins_in_chunk = block_destinations % chunk_rows // bin_rows
        for chunk, b in set(zip(chunks[kept].tolist(), bins_in_chunk[kept].tolist())):
            if (chunk, b) not in bin_files:
                path = bins_dir / f"chunk_{chunk:05d}_bin_{b:03d}_worker_{worker:03d}.arrow"
                bin_files[chunk, b] = pa.ipc.new_file(str(path), block.schema, options=bin_options)
            rows_in_bin = np.flatnonzero(kept & (chunks == chunk) & (bins_in_chunk == b))
            bin_files[chunk, b].write_batch(block.take(rows_in_bin))
    for bin_file in bin_files.values():
        bin_file.close()


def _write_chunk_from_bins(bins_dir: Path, dest_path: Path, chunk: int, n_bins: int) -> tuple[pa.Schema, list]:
    """Write chunk ``chunk`` as one fragment from its ``n_bins`` bins."""
    return _write_fragments(dest_path, _sorted_bin_batches(bins_dir, chunk, n_bins))


def _sorted_bin_batches(bins_dir: Path, chunk: int, n_bins: int):
    """Record batches of the chunk's bins in turn, each bin loaded from all pass-1 workers and sorted by new position.

    Only one bin (and its sorted copy) is in memory at a time.
    """
    for b in range(n_bins):
        paths = bins_dir.glob(f"chunk_{chunk:05d}_bin_{b:03d}_worker_*.arrow")
        rows = pa.concat_tables([pa.ipc.open_file(str(path)).read_all() for path in paths])
        rows = rows.take(np.argsort(rows["global_row_index"].to_numpy()))
        yield from rows.to_batches()


def _write_chunks(source: LanceDatasetEntry, dest_path: Path, starts: list[int], chunks: list[np.ndarray],
                  packed: bool) -> tuple[pa.Schema, list]:
    """Write each chunk (new rows from ``start``, source rows ``chunk``) as fragments, reading with one reader.

    Returns the table schema and the fragments in chunk order; nothing is committed here.
    """
    reader = FederatedLanceReader([source])
    fragments = []
    for start, chunk in zip(starts, chunks):
        schema, chunk_fragments = _write_fragments(dest_path, _reordered_batches(reader, start, chunk, packed))
        fragments += chunk_fragments
    return schema, fragments


def _reordered_batches(reader: FederatedLanceReader, start: int, source_rows: np.ndarray, packed: bool):
    """Record batches of the rows ``source_rows``, which take the new positions ``start, start + 1, ...``."""
    for block_start in range(0, len(source_rows), BLOCK_ROWS):
        block_rows = source_rows[block_start:block_start + BLOCK_ROWS]
        # Rows come back in the requested order (the reader restores it after its sorted take).
        first_new_row = start + block_start
        new_rows = np.arange(first_new_row, first_new_row + len(block_rows), dtype=np.int64)
        yield _record_batch(reader.read_expression_flat(block_rows), new_rows, block_rows, packed)


def _record_batch(batch, new_rows: np.ndarray, source_rows: np.ndarray, packed: bool) -> pa.RecordBatch:
    """One record batch of the new table from an ``ExpressionBatch`` read at ``source_rows``."""
    offsets = pa.array(batch.row_offsets, type=pa.int32())
    columns = {
        "global_row_index": pa.array(new_rows),
        "source_row": pa.array(source_rows),
    }
    if packed:
        values = _pack(batch.expressed_gene_indices, batch.expression_counts)
        columns["packed"] = pa.ListArray.from_arrays(offsets, pa.array(values))
    else:
        columns["expressed_gene_indices"] = pa.ListArray.from_arrays(offsets, pa.array(batch.expressed_gene_indices))
        columns["expression_counts"] = pa.ListArray.from_arrays(offsets, pa.array(batch.expression_counts))
    return pa.RecordBatch.from_pydict(columns)


def _write_fragments(dest_path: Path, batches) -> tuple[pa.Schema, list]:
    """Encode ``batches`` into new fragment files under ``dest_path``, not yet part of any table version."""
    import lance

    first = next(batches)  # its schema carries the counts type of this table
    stream = pa.RecordBatchReader.from_batches(first.schema, itertools.chain([first], batches))
    fragments = lance.fragment.write_fragments(stream, str(dest_path), mode="create", data_storage_version="2.2")
    return first.schema, fragments


def _commit_in_order(dest_path: Path, schema: pa.Schema, fragments: list, n_rows: int) -> None:
    """Create the table from ``fragments``, in list order, and check its rows are new rows ``0 .. n_rows - 1``.

    ``Overwrite`` gives the fragments ids in list order. Never ``compact_files`` afterwards
    (in Lance 13 it committed fragments out of row order).
    """
    import lance

    table = lance.LanceDataset.commit(str(dest_path), lance.LanceOperation.Overwrite(schema, fragments))
    # A misordered commit is still a complete, readable table, so only the row order shows it.
    new_rows = table.to_table(columns=["global_row_index"])["global_row_index"].to_numpy()
    if not np.array_equal(new_rows, np.arange(n_rows)):
        raise RuntimeError(f"{dest_path}: fragments were committed out of row order")


def _pack(genes: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Packed values ``(count_bits << 16) | gene``, as int64.

    ``count_bits`` is the raw 32 bits of a float32 count, or the count itself.
    The reader decodes non-float32 count bits as an unsigned 32-bit integer, so
    those counts must be whole numbers in [0, 2**32); anything else would read
    back silently wrong, so it is refused.
    """
    if counts.dtype == np.float32:
        count_bits = counts.view(np.uint32).astype(np.int64)
    else:
        count_bits = counts.astype(np.int64)
        fits = np.array_equal(count_bits, counts) and count_bits.min(initial=0) >= 0 and count_bits.max(initial=0) < 2**32
        if not fits:
            raise ValueError("Cannot pack counts that are not float32 or whole numbers in [0, 2**32)")
    return (count_bits << 16) | genes.astype(np.int64)
