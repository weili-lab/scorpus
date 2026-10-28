"""Backend-agnostic expression readers.

Defines the ``ExpressionReader`` protocol, a ``BaseExpressionReader`` base
class holding the dataset routing entries, and the Lance readers.

Lance readers (and views over them) use ``SortedTakeReader``: one sorted
``take`` per table, all tables read at once, and at most one copy into the batch
(none for sorted, distinct rows of one two-column table).

Supported backend: Lance.
Supported topologies: aggregate and federated.

Readers return **only expression data** via
``read_expression_flat(global_indices) -> ExpressionBatch``. Identity and
metadata fields (dataset_id, dataset_index, local_row_index, etc.) belong in
``MetadataIndex``.
"""

from __future__ import annotations

import os
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Protocol, Sequence

import numpy as np

__all__ = [
    "ExpressionBatch",
    "ExpressionReader",
    "BaseExpressionReader",
    "SortedTakeReader",
    "AggregateLanceReader",
    "FederatedLanceReader",
    "LanceDatasetEntry",
    "DatasetEntry",
    "build_expression_reader",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpressionBatch:
    """Flat sparse expression arrays for one batch."""

    batch_size: int
    global_row_index: np.ndarray
    row_offsets: np.ndarray
    expressed_gene_indices: np.ndarray
    expression_counts: np.ndarray

    def row_slice(self, row_position: int) -> slice:
        start = int(self.row_offsets[row_position])
        stop = int(self.row_offsets[row_position + 1])
        return slice(start, stop)

    def row_gene_indices(self, row_position: int) -> np.ndarray:
        return self.expressed_gene_indices[self.row_slice(row_position)]

    def row_counts(self, row_position: int) -> np.ndarray:
        return self.expression_counts[self.row_slice(row_position)]


@dataclass(frozen=True)
class DatasetEntry:
    """Base routing entry describing a dataset's global index range.

    Subclasses add backend-specific path attributes.
    """

    dataset_id: str
    global_start: int  # inclusive, first global index belonging to this dataset
    global_end: int  # exclusive, first global index after this dataset


@dataclass(frozen=True)
class LanceDatasetEntry(DatasetEntry):
    """Lance-specific dataset entry with a path to the Lance file.

    ``expression_layout`` is ``"two_column"`` (list columns
    ``expressed_gene_indices`` and ``expression_counts``) or ``"packed"`` (one
    int64 list column ``packed``, see ``_unpack_genes_and_counts``). A packed
    table also needs ``counts_dtype``, the dtype its counts were stored in.
    """

    lance_path: str | Path
    expression_layout: str = "two_column"
    counts_dtype: str | None = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class ExpressionReader(Protocol):
    """Protocol for topology-aware expression readers.

    Implementations must return expression data **only**. Metadata
    enrichment is handled separately by ``MetadataIndex`` and higher-level
    loader code.
    """

    def read_expression_flat(
        self, global_indices: Sequence[int]
    ) -> ExpressionBatch:
        """Read expression data as an order-preserving ``ExpressionBatch``."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_expression_batch(global_indices: Sequence[int] | None = None) -> ExpressionBatch:
    """Return an empty ``ExpressionBatch`` preserving index dtype conventions."""
    if global_indices is None:
        indices = np.array([], dtype=np.int64)
    else:
        indices = np.array([int(idx) for idx in global_indices], dtype=np.int64)
    return ExpressionBatch(
        batch_size=0,
        global_row_index=indices,
        row_offsets=np.array([0], dtype=np.int64),
        expressed_gene_indices=np.array([], dtype=np.int32),
        expression_counts=np.array([], dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# BaseExpressionReader — holds the dataset routing entries
# ---------------------------------------------------------------------------


class BaseExpressionReader(ABC):
    """Abstract base for expression readers.

    Parameters
    ----------
    entries : list of DatasetEntry
        One entry per dataset, sorted by ``global_start``.  For aggregate
        topology this is a single entry covering the full range.
    """

    def __init__(self, entries: list[DatasetEntry]):
        self._entries: list[DatasetEntry] = sorted(
            entries, key=lambda e: e.global_start
        )


# ===================================================================
# Lance readers
# ===================================================================


class _RowBlock(NamedTuple):
    """Consecutive rows read from one table, as views into the read buffers.

    Row ``i`` has genes ``genes[offsets[i]:offsets[i + 1]]`` and counts at the
    same positions of ``counts``. ``counts_dtype`` is the dtype these counts
    take in the batch.
    """

    offsets: np.ndarray
    genes: np.ndarray
    counts: np.ndarray
    counts_dtype: np.dtype


def _unpack_genes_and_counts(packed: np.ndarray, counts_dtype: str) -> tuple[np.ndarray, np.ndarray]:
    """Views of the genes and counts held in little-endian packed int64 values.

    Each value is ``(count_bits << 16) | gene``: the gene is in bytes 0-1 and the
    count bits start at byte 2. Count bits are the integer count, or the raw 32
    bits of a float32 count. Counts are read as 32 bits from byte 2 (exact up to
    ~4.3 billion); a 16-bit count view would wrap above 65,535.
    """
    four_uint16_per_value = packed.view("<u2").reshape(-1, 4)
    genes = four_uint16_per_value[:, 0]
    count_bits_dtype = "<f4" if np.dtype(counts_dtype) == np.float32 else "<u4"
    counts = four_uint16_per_value[:, 1:3].view(count_bits_dtype)[:, 0]
    return genes, counts


# Lance take returns int32-offset lists, so each piece must stay under 2^31 values; 2**17 rows is safe for any corpus with fewer than 16,384 genes.
_TAKE_PIECE_ROWS = 2**17


def _take_lance_rows(
    dataset, rows: np.ndarray, expression_layout: str, counts_dtype: str | None
) -> list[_RowBlock]:
    """Read sorted, distinct ``rows`` of one Lance table, ``take`` by ``take`` of at most ``_TAKE_PIECE_ROWS`` rows.

    Returns one block of views per Arrow record batch of the result; nothing is
    copied here.
    """
    blocks = []
    for begin in range(0, len(rows), _TAKE_PIECE_ROWS):
        piece = rows[begin:begin + _TAKE_PIECE_ROWS]
        if expression_layout == "packed":
            for record_batch in dataset.take(piece, columns=["packed"]).to_batches():
                offsets, packed = _list_views(record_batch.column(0))
                genes, counts = _unpack_genes_and_counts(packed, counts_dtype)
                blocks.append(_RowBlock(offsets, genes, counts, np.dtype(counts_dtype)))
        else:
            table = dataset.take(piece, columns=["expressed_gene_indices", "expression_counts"])
            for record_batch in table.to_batches():
                offsets, genes = _list_views(record_batch.column(0))
                counts = record_batch.column(1).flatten().to_numpy()
                blocks.append(_RowBlock(offsets, genes, counts, counts.dtype))
    return blocks


def _list_views(list_array) -> tuple[np.ndarray, np.ndarray]:
    """Offsets (starting at 0) and values of an Arrow list array, without copying the values."""
    # flatten() starts at the first row's values, so shift offsets to start at 0.
    offsets = np.asarray(list_array.offsets)
    return offsets - offsets[0], list_array.flatten().to_numpy()


def _fill_small_gaps(rows: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Sorted, distinct ``rows`` plus all rows in gaps of at most ``max_gap`` between them.

    Returns the rows to read and the position of each of ``rows`` among them.
    """
    gaps = np.diff(rows) - 1
    filled_after = np.append(np.where(gaps <= max_gap, gaps, 0), 0)
    rows_read_per_row = 1 + filled_after  # the row itself, then its filled gap
    positions = np.cumsum(rows_read_per_row) - rows_read_per_row
    # Position k in the segment that starts at positions[i] holds row rows[i] + (k - positions[i]).
    rows_to_read = np.repeat(rows - positions, rows_read_per_row) + np.arange(rows_read_per_row.sum())
    return rows_to_read, positions


class SortedTakeReader(BaseExpressionReader):
    """Reader that fetches each table's rows with one sorted read.

    Subclasses define how to read sorted, distinct rows of one table
    (``_take_rows``) and, if their tables differ from the routing entries,
    which tables there are (``_tables``). Set ``reader.max_gap`` to also read
    gaps of up to that many rows between requested rows (dropped in memory).
    """

    max_gap = 0

    def read_expression_flat(
        self, global_indices: Sequence[int]
    ) -> ExpressionBatch:
        """Read the rows in the given order (duplicates allowed) as one batch.

        ``max_gap`` (default 0): gaps of at most this many rows between
        requested rows of one table are read too, so each stretch is one
        contiguous range; the gap rows are dropped and the batch is unchanged.

        When the rows are sorted, distinct, from one two-column table and no
        gap was filled, the gene and count arrays are read-only views of
        Lance's read buffers: copy before modifying them (``.copy()``, or
        ``torch.as_tensor`` / ``torch.tensor`` rather than ``torch.from_numpy``).
        """
        indices = np.array(global_indices, dtype=np.int64)
        if len(indices) == 0:
            return _empty_expression_batch()

        # 1. Distinct rows in sorted order; inverse[i] is the position of
        #    batch row i among them (duplicates share one position).
        rows, inverse = np.unique(indices, return_inverse=True)
        tables = self._tables()
        stops = np.array([table.global_end for table in tables], dtype=np.int64)
        if rows[0] < tables[0].global_start or rows[-1] >= stops[-1]:
            raise IndexError(
                f"global indices must lie in [{tables[0].global_start}, {stops[-1]}); "
                f"got min {rows[0]}, max {rows[-1]}"
            )

        # 2. Tables cover consecutive ranges, so each table's rows are one
        #    consecutive, already sorted slice of `rows`. Small gaps between
        #    them are read too; read_position[i] is where distinct row i lies
        #    among all rows read.
        table_ends = np.searchsorted(rows, stops)
        jobs, read_positions = [], []
        n_read = 0
        begin = 0
        for table, end in zip(tables, table_ends):
            if end > begin:
                table_rows, positions = _fill_small_gaps(rows[begin:end] - table.global_start, self.max_gap)
                jobs.append((table, table_rows))
                read_positions.append(n_read + positions)
                n_read += len(table_rows)
            begin = end
        read_position = np.concatenate(read_positions)

        # 3-4. One read per table, all tables at the same time.
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            blocks_per_table = list(pool.map(lambda job: self._take_rows(*job), jobs))
        blocks = [block for table_blocks in blocks_per_table for block in table_blocks]

        # 5. For each row read (sorted order): its block, and where its
        #    values start and stop in that block.
        block_of_row = np.repeat(np.arange(len(blocks)), [len(block.offsets) - 1 for block in blocks])
        row_starts = np.concatenate([block.offsets[:-1] for block in blocks])
        row_stops = np.concatenate([block.offsets[1:] for block in blocks])

        # 6. Put the requested rows in batch order (filled gap rows drop out
        #    here); genes become int32 and counts the tables' common counts dtype.
        batch_position = read_position[inverse]
        block_of_row = block_of_row[batch_position]
        row_starts = row_starts[batch_position]
        row_stops = row_stops[batch_position]
        row_offsets = np.zeros(len(indices) + 1, dtype=np.int64)
        np.cumsum(row_stops - row_starts, out=row_offsets[1:])
        counts_dtype = np.result_type(*[block.counts_dtype for block in blocks])
        if len(blocks) == 1 and n_read == len(rows) and np.array_equal(inverse, np.arange(len(indices))):
            # One block already in batch order and no gap filled, so no copy is needed
            # (packed values still need one, to unpack them).
            genes = np.ascontiguousarray(blocks[0].genes, dtype=np.int32)
            counts = np.ascontiguousarray(blocks[0].counts, dtype=counts_dtype)
        else:
            # One copy, one slice per range of rows that follow on in a block.
            continues_previous = (block_of_row[1:] == block_of_row[:-1]) & (row_starts[1:] == row_stops[:-1])
            first_row = np.flatnonzero(np.concatenate([[True], ~continues_previous]))
            last_row = np.append(first_row[1:], len(indices)) - 1
            ranges = list(zip(block_of_row[first_row].tolist(), row_starts[first_row].tolist(), row_stops[last_row].tolist()))
            genes = np.concatenate([blocks[b].genes[start:stop] for b, start, stop in ranges], dtype=np.int32)
            counts = np.concatenate([blocks[b].counts[start:stop] for b, start, stop in ranges], dtype=counts_dtype)
        return ExpressionBatch(
            batch_size=len(indices),
            global_row_index=indices,
            row_offsets=row_offsets,
            expressed_gene_indices=genes,
            expression_counts=counts,
        )

    def _tables(self) -> list[DatasetEntry]:
        """Tables in global order; each is read with one ``_take_rows``."""
        return self._entries

    def _take_rows(self, table: DatasetEntry, rows: np.ndarray) -> list[_RowBlock]:
        """Read sorted, distinct table-local ``rows`` of one table."""
        raise NotImplementedError


class AggregateLanceReader(SortedTakeReader):
    """Expression reader for aggregate Lance topology.

    Opens a single ``lance.Dataset`` handle and reads all requested rows
    with one ``take()``.

    Parameters
    ----------
    lance_path : str or Path
        Path to the aggregated ``.lance`` directory.
    entries : list of DatasetEntry
        Index ranges for each dataset in the corpus.  For aggregate
        topology this is typically a single entry covering the full
        range (0 to total_rows).
    """

    def __init__(self, lance_path: str | Path, entries: list[DatasetEntry]):
        super().__init__(entries)
        self._lance_path = str(Path(lance_path))
        self._dataset = None
        self._dataset_pid: int | None = None
        self._total_rows: int | None = None

    @property
    def lance_path(self) -> str:
        return self._lance_path

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_dataset"] = None
        state["_dataset_pid"] = None
        return state

    def _open_dataset(self):
        import lance

        current_pid = os.getpid()
        if self._dataset is None or self._dataset_pid != current_pid:
            self._dataset = lance.dataset(self._lance_path)
            self._dataset_pid = current_pid
        return self._dataset

    def _count_rows(self) -> int:
        if self._total_rows is None:
            self._total_rows = int(self._open_dataset().count_rows())
        return self._total_rows

    def _tables(self) -> list[DatasetEntry]:
        # Global indices are rows of the one aggregate file, whatever the entries.
        return [DatasetEntry("aggregate", 0, self._count_rows())]

    def _take_rows(self, table: DatasetEntry, rows: np.ndarray) -> list[_RowBlock]:
        return _take_lance_rows(self._open_dataset(), rows, "two_column", None)


class FederatedLanceReader(SortedTakeReader):
    """Expression reader for federated Lance topology.

    Opens per-dataset ``lance.Dataset`` handles (lazily cached) and
    reads each dataset's rows with one ``take()``, all datasets at once.

    Parameters
    ----------
    entries : list of LanceDatasetEntry
        One entry per dataset, each with its own Lance file path and
        global index range.
    """

    def __init__(self, entries: list[LanceDatasetEntry]):
        super().__init__(entries)  # type: ignore[arg-type]
        self._datasets: dict[str, Any] = {}
        self._datasets_pid: int | None = None

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_datasets"] = {}
        state["_datasets_pid"] = None
        return state

    def _open_dataset(self, entry: LanceDatasetEntry):
        import lance

        current_pid = os.getpid()
        if self._datasets_pid != current_pid:
            self._datasets = {}
            self._datasets_pid = current_pid
        # Called from several threads at once: return the local handle rather
        # than looking it up again in a dict another thread may have replaced.
        dataset = self._datasets.get(entry.dataset_id)
        if dataset is None:
            dataset = lance.dataset(str(entry.lance_path))
            self._datasets[entry.dataset_id] = dataset
        return dataset

    def _take_rows(self, table: LanceDatasetEntry, rows: np.ndarray) -> list[_RowBlock]:
        return _take_lance_rows(self._open_dataset(table), rows, table.expression_layout, table.counts_dtype)


# ===================================================================
# Factory
# ===================================================================


def build_expression_reader(
    backend: str,
    topology: str,
    entries: list[DatasetEntry],
    **kwargs,
) -> ExpressionReader:
    """Build an expression reader for the given backend and topology.

    Parameters
    ----------
    backend : str
        Only ``"lance"``.
    topology : str
        Either ``"aggregate"`` or ``"federated"``.
    entries : list of DatasetEntry
        Dataset routing entries.  The concrete entry type must match the
        backend (e.g., ``LanceDatasetEntry`` for Lance).
    **kwargs
        Backend-specific arguments (e.g., ``lance_path`` for aggregate Lance).

    Returns
    -------
    ExpressionReader
        A concrete reader instance.

    Raises
    ------
    ValueError
        If the backend/topology combination is unsupported.
    """
    if topology not in ("aggregate", "federated"):
        raise ValueError(
            f"Unknown topology '{topology}'. Expected 'aggregate' or 'federated'."
        )

    if backend == "lance":
        if topology == "aggregate":
            lance_path = kwargs.pop("lance_path")
            return AggregateLanceReader(lance_path, entries)
        else:
            return FederatedLanceReader(entries)  # type: ignore[arg-type]

    raise ValueError(f"Unknown backend '{backend}'. Supported: lance.")
