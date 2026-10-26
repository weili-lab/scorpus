"""Rewrite a standalone corpus with its cells in a new order.

Every per-cell table moves together: the expression table, each ``extra/*.lance``
table, ``obs.parquet`` and ``statistics.parquet``. ``var.parquet`` and the extra
feature tables are copied unchanged. With ``pairing.write_order`` as the order,
each group becomes one contiguous block, shuffled within, so ``Sampler(...,
run_length=r)`` reads runs of neighbouring rows. An order that leaves out rows
(``write_order(..., keep=...)``) writes a corpus of only those cells.
"""

from __future__ import annotations

import itertools
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
# Rows of the new order written by one worker task: about one Lance fragment.
FRAGMENT_ROWS = 1024 * 1024


def reorder_corpus(source_root, dest_root, order, row_order_note: dict, layout: str = "two_column",
                   workers: int = 1) -> Path:
    """Write the corpus at ``source_root`` to ``dest_root`` with rows ``order``.

    ``order`` gives the source row for each new position (``new = old[order]``),
    as returned by ``pairing.write_order``. It may leave out source rows (a filtered
    rewrite); the new corpus then has ``len(order)`` cells. ``row_order_note`` (for
    example the grouping columns, seed and original source path) is stored as
    ``row_order`` in the new ``corpus.yaml``.

    Lance tables get ``global_row_index`` = new position and ``source_row`` =
    original row. ``workers`` processes read and write chunks of the new order in
    parallel, each holding one lean reader (about 5-15 GB for a large corpus).

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
                           packed=layout == "packed", workers=workers)
    extra = manifest.get("extra", {})
    if extra:
        (dest_root / "extra").mkdir()
    for name in extra:
        source_extra = LanceDatasetEntry(name, 0, n_cells, source_root / "extra" / f"{name}.lance")
        _write_reordered_table(source_extra, dest_root / "extra" / f"{name}.lance", order,
                               packed=False, workers=workers)
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
                           workers: int) -> None:
    """Write the rows ``order`` of one table as fragments of ``FRAGMENT_ROWS`` rows, committed once in order.

    Each worker writes a contiguous range of chunks with one lean reader, so the workers'
    fragments, concatenated, are in row order. Workers are spawned, not forked, because
    Lance is not fork-safe; ``workers=1`` writes in this process.
    """
    starts = list(range(0, len(order), FRAGMENT_ROWS))
    chunks = [order[start:start + FRAGMENT_ROWS] for start in starts]
    if workers == 1:
        written = [_write_chunks(source, dest_path, starts, chunks, packed)]
    else:
        worker_chunks = np.array_split(np.arange(len(chunks)), min(workers, len(chunks)))
        with ProcessPoolExecutor(len(worker_chunks), mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_write_chunks, source, dest_path,
                                   [starts[i] for i in ids], [chunks[i] for i in ids], packed)
                       for ids in worker_chunks]
            written = [future.result() for future in futures]
    schema, _ = written[0]  # the same for every chunk of this table
    fragments = [fragment for _, worker_fragments in written for fragment in worker_fragments]
    _commit_in_order(dest_path, schema, fragments, len(order))


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
        batch = reader.read_expression_flat(block_rows)
        offsets = pa.array(batch.row_offsets, type=pa.int32())
        first_new_row = start + block_start
        columns = {
            "global_row_index": pa.array(np.arange(first_new_row, first_new_row + len(block_rows), dtype=np.int64)),
            "source_row": pa.array(block_rows),
        }
        if packed:
            values = _pack(batch.expressed_gene_indices, batch.expression_counts)
            columns["packed"] = pa.ListArray.from_arrays(offsets, pa.array(values))
        else:
            columns["expressed_gene_indices"] = pa.ListArray.from_arrays(offsets, pa.array(batch.expressed_gene_indices))
            columns["expression_counts"] = pa.ListArray.from_arrays(offsets, pa.array(batch.expression_counts))
        yield pa.RecordBatch.from_pydict(columns)


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
