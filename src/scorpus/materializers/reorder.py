"""Rewrite a standalone corpus with its cells in a new order.

Every per-cell table moves together: the expression table, each ``extra/*.lance``
table, ``obs.parquet`` and ``statistics.parquet``. ``var.parquet`` and the extra
feature tables are copied unchanged. With ``pairing.write_order`` as the order,
each group becomes one contiguous block, shuffled within, so ``Sampler(...,
run_length=r)`` reads runs of neighbouring rows.
"""

from __future__ import annotations

import itertools
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import yaml

from ..loaders.expression import FederatedLanceReader, LanceDatasetEntry

# Rows gathered per lean-reader read (one sorted take); also one record batch of the new table.
BLOCK_ROWS = 65_536


def reorder_corpus(source_root, dest_root, order, row_order_note: dict, layout: str = "two_column") -> Path:
    """Write the corpus at ``source_root`` to ``dest_root`` with rows ``order``.

    ``order`` gives the source row for each new position (``new = old[order]``),
    as returned by ``pairing.write_order``. ``row_order_note`` (for example the
    grouping columns, seed and original source path) is stored as ``row_order``
    in the new ``corpus.yaml``.

    Lance tables get ``global_row_index`` = new position and ``source_row`` =
    original row. ``layout="packed"`` writes the expression table as one int64
    list column ``packed`` with values ``(count_bits << 16) | gene`` (see
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
    _write_reordered_table(source_expression, dest_root / "expression.lance", order, packed=layout == "packed")
    if manifest.get("extra"):
        (dest_root / "extra").mkdir()
    for name in manifest.get("extra", {}):
        source_extra = LanceDatasetEntry(name, 0, n_cells, source_root / "extra" / f"{name}.lance")
        _write_reordered_table(source_extra, dest_root / "extra" / f"{name}.lance", order, packed=False)
        shutil.copyfile(source_root / "extra" / f"{name}_features.parquet",
                        dest_root / "extra" / f"{name}_features.parquet")

    # 2. Per-cell metadata in the new order; gene metadata unchanged.
    pd.read_parquet(source_root / "obs.parquet").iloc[order].to_parquet(dest_root / "obs.parquet")
    if manifest["expression_kind"] != "non_count":
        statistics = pd.read_parquet(source_root / "statistics.parquet")
        statistics.iloc[order].to_parquet(dest_root / "statistics.parquet", index=False)
    shutil.copyfile(source_root / "var.parquet", dest_root / "var.parquet")

    # 3. The manifest last, so an interrupted rewrite is not a loadable corpus.
    manifest["row_order"] = row_order_note
    if layout == "packed":
        manifest["expression_layout"] = "packed"  # counts dtype stays in `dtype`
    else:
        manifest.pop("expression_layout", None)
    (dest_root / "corpus.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return dest_root


def _write_reordered_table(source: LanceDatasetEntry, dest_path: Path, order: np.ndarray, packed: bool) -> None:
    """Stream the rows ``order`` of one table, block by block, into one ``write_dataset``.

    One write keeps the rows in order in ~1M-row fragments; never ``compact_files``
    afterwards (in Lance 13 it committed fragments out of row order).
    """
    import lance

    reader = FederatedLanceReader([source])

    def record_batches():
        for start in range(0, len(order), BLOCK_ROWS):
            source_rows = order[start:start + BLOCK_ROWS]
            # Rows come back in the requested order (the reader restores it after its sorted take).
            batch = reader.read_expression_flat(source_rows)
            offsets = pa.array(batch.row_offsets, type=pa.int32())
            columns = {
                "global_row_index": pa.array(np.arange(start, start + len(source_rows), dtype=np.int64)),
                "source_row": pa.array(source_rows),
            }
            if packed:
                values = _pack(batch.expressed_gene_indices, batch.expression_counts)
                columns["packed"] = pa.ListArray.from_arrays(offsets, pa.array(values))
            else:
                columns["expressed_gene_indices"] = pa.ListArray.from_arrays(offsets, pa.array(batch.expressed_gene_indices))
                columns["expression_counts"] = pa.ListArray.from_arrays(offsets, pa.array(batch.expression_counts))
            yield pa.RecordBatch.from_pydict(columns)

    batches = record_batches()
    first = next(batches)  # its schema carries the counts type of this table
    stream = pa.RecordBatchReader.from_batches(first.schema, itertools.chain([first], batches))
    lance.write_dataset(stream, str(dest_path), mode="create", data_storage_version="2.2")


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
