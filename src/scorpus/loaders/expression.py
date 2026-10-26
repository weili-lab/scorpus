"""Backend-agnostic expression readers.

Defines the ``ExpressionReader`` protocol, a ``BaseExpressionReader`` abstract
class that handles global→local routing and order-preserving reassembly, and
backend-specific implementations for the remaining slim-main readers.

Lance readers (and views over them) use ``SortedTakeReader``: one sorted
``take`` per table, all tables read at once, and at most one copy into the batch
(none for sorted, distinct rows of one two-column table).

Supported backends: Lance and Zarr.
Supported topologies: aggregate and federated.

Readers return **only expression data** via
``read_expression_flat(global_indices) -> ExpressionBatch``. Identity and
metadata fields (dataset_id, dataset_index, local_row_index, size_factor,
etc.) belong in ``MetadataIndex``.
"""

from __future__ import annotations

import os
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Protocol, Sequence

import numpy as np

from .zarr_reading import open_csr_arrays, normalize_zarr_read_engine

__all__ = [
    "ExpressionBatch",
    "ExpressionReader",
    "BaseExpressionReader",
    "SortedTakeReader",
    "AggregateLanceReader",
    "FederatedLanceReader",
    "AggregateZarrReader",
    "FederatedZarrReader",
    "LanceDatasetEntry",
    "ZarrDatasetEntry",
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


@dataclass(frozen=True)
class ZarrDatasetEntry(DatasetEntry):
    """Zarr-specific dataset entry with paths to the CSR-format arrays.

    For the v3 sharded CSR layout, all three paths point to the same Zarr
    group. Legacy corpora keep one Zarr group per array.
    """

    offsets_path: str | Path
    indices_path: str | Path
    counts_path: str | Path


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


def _cell_arrays_to_expression_batch(
    global_indices: Sequence[int],
    cells: Sequence[tuple[np.ndarray, np.ndarray]],
) -> ExpressionBatch:
    """Convert ordered per-cell arrays into an ``ExpressionBatch``."""
    indices = np.array([int(idx) for idx in global_indices], dtype=np.int64)
    if len(cells) == 0:
        return _empty_expression_batch(indices.tolist())

    row_offsets = np.zeros(len(cells) + 1, dtype=np.int64)
    egi_parts: list[np.ndarray] = []
    ec_parts: list[np.ndarray] = []
    for pos, (gene_indices, counts) in enumerate(cells):
        gene_indices = np.asarray(gene_indices, dtype=np.int32)
        counts = np.asarray(counts)
        egi_parts.append(gene_indices)
        ec_parts.append(counts)
        row_offsets[pos + 1] = row_offsets[pos] + len(gene_indices)

    return ExpressionBatch(
        batch_size=len(cells),
        global_row_index=indices,
        row_offsets=row_offsets,
        expressed_gene_indices=np.concatenate(egi_parts),
        expression_counts=np.concatenate(ec_parts),
    )


# ---------------------------------------------------------------------------
# BaseExpressionReader — handles routing, grouping, and reassembly
# ---------------------------------------------------------------------------


class BaseExpressionReader(ABC):
    """Abstract base for expression readers.

    Handles global→local routing, per-dataset grouping, and order-preserving
    reassembly.  Subclasses implement only backend-specific local-row reading.

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
        # Precompute stop array for np.searchsorted-based routing
        self._stops: np.ndarray = np.array(
            [e.global_end for e in self._entries], dtype=np.int64
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_expression_flat(
        self, global_indices: Sequence[int]
    ) -> ExpressionBatch:
        """Read expression data with shared routing and order reassembly.

        Backends with a more efficient flat representation can override this
        method. Other readers inherit this shared order-preserving
        implementation backed by per-dataset local reads.
        """
        if not global_indices:
            return _empty_expression_batch()

        indices = [int(idx) for idx in global_indices]
        self._validate_all(indices)

        grouped: dict[str, list[tuple[int, int]]] = {}
        for output_pos, global_idx in enumerate(indices):
            entry, local_idx = self._resolve_entry(global_idx)
            grouped.setdefault(entry.dataset_id, []).append((output_pos, local_idx))

        cells_by_pos: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for ds_id, selections in grouped.items():
            entry = self._find_entry_by_id(ds_id)
            local_indices = [local for _, local in selections]
            cells = self._read_local_cells(entry, local_indices)
            if len(cells) != len(selections):
                raise ValueError(
                    f"Reader for dataset_id '{ds_id}' returned {len(cells)} cells "
                    f"for {len(selections)} requested indices"
                )
            for (output_pos, _), cell in zip(selections, cells, strict=True):
                cells_by_pos[output_pos] = cell

        ordered_cells = [cells_by_pos[pos] for pos in range(len(indices))]
        return _cell_arrays_to_expression_batch(indices, ordered_cells)

    # ------------------------------------------------------------------
    # Routing helpers (shared across all backends)
    # ------------------------------------------------------------------

    def _resolve_entry(self, global_index: int) -> tuple[DatasetEntry, int]:
        """Resolve a global index to its owning dataset entry and local index.

        Uses ``np.searchsorted`` on ``global_end`` stops for O(log N) lookup.
        """
        entry_idx = _searchsorted_entry(self._stops, global_index)
        entry = self._entries[entry_idx]
        if not (entry.global_start <= global_index < entry.global_end):
            raise IndexError(
                f"global_index {global_index} does not fall within any "
                f"registered dataset range"
            )
        local_idx = global_index - entry.global_start
        return entry, local_idx

    def _find_entry_by_id(self, dataset_id: str) -> DatasetEntry:
        """Return the entry for *dataset_id*."""
        for entry in self._entries:
            if entry.dataset_id == dataset_id:
                return entry
        raise KeyError(
            f"dataset_id '{dataset_id}' not found; "
            f"available: {[e.dataset_id for e in self._entries]}"
        )

    def _validate_all(self, indices: list[int]) -> None:
        """Validate that all indices are within the total corpus range.

        Subclasses may override to add backend-specific checks (e.g.,
        Lance row count check).  The base implementation checks only
        that indices are non-negative and within the overall start/end
        bounds implied by the entry list.
        """
        total_start = self._entries[0].global_start
        total_end = self._entries[-1].global_end
        for idx in indices:
            if idx < total_start or idx >= total_end:
                raise IndexError(
                    f"global_index {idx} out of range "
                    f"[{total_start}, {total_end})"
                )

    # ------------------------------------------------------------------
    # Backend-specific hook
    # ------------------------------------------------------------------

    def _read_local_cells(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Read per-cell expression arrays within one dataset.

        Returns one ``(expressed_gene_indices, expression_counts)`` tuple per
        requested local index, in the same order as ``local_indices``.
        Needed by readers that use the per-cell ``read_expression_flat`` above
        (Zarr); ``SortedTakeReader`` subclasses read whole tables instead.
        """
        raise NotImplementedError


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
# Zarr readers
# ===================================================================


def _zarr_read_cells(
    offsets: np.ndarray,
    indices_flat: np.ndarray,
    counts_flat: np.ndarray,
    local_indices: list[int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read per-cell expression arrays from CSR-format Zarr arrays."""
    cells: list[tuple[np.ndarray, np.ndarray]] = []
    for local_idx in local_indices:
        s = slice(offsets[local_idx], offsets[local_idx + 1])
        cells.append((indices_flat[s].copy(), counts_flat[s].copy()))
    return cells


class AggregateZarrReader(BaseExpressionReader):
    """Expression reader for aggregate Zarr topology (CSR format).

    Parameters
    ----------
    offsets_path : str or Path
        Path to the row-offsets Zarr array (int64, shape [n_rows+1]).
    indices_path : str or Path
        Path to the flat gene-indices Zarr array (int32).
    counts_path : str or Path
        Path to the flat expression-counts Zarr array (int32).
    entries : list of DatasetEntry
        Dataset range entries (single entry for aggregate).
    """

    def __init__(
        self,
        offsets_path: str | Path,
        indices_path: str | Path,
        counts_path: str | Path,
        entries: list[DatasetEntry],
        *,
        read_engine: str = "zarr-python",
    ):
        super().__init__(entries)
        self._read_engine = normalize_zarr_read_engine(read_engine)
        arrays = open_csr_arrays(
            offsets_path,
            indices_path,
            counts_path,
            read_engine=self._read_engine,
        )
        self._offsets = arrays.row_offsets
        self._indices = arrays.indices
        self._counts = arrays.counts

    def _read_local_cells(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        global_positions = [entry.global_start + li for li in local_indices]
        return _zarr_read_cells(
            self._offsets,
            self._indices,
            self._counts,
            global_positions,
        )


class FederatedZarrReader(BaseExpressionReader):
    """Expression reader for federated Zarr topology (CSR format).

    Parameters
    ----------
    entries : list of ZarrDatasetEntry
        One entry per dataset, each with its own Zarr array paths.
    """

    def __init__(self, entries: list[ZarrDatasetEntry], *, read_engine: str = "zarr-python"):
        super().__init__(entries)  # type: ignore[arg-type]
        self._read_engine = normalize_zarr_read_engine(read_engine)
        self._arrays_cache: dict[str, Any] = {}

    def _open_arrays(self, entry: ZarrDatasetEntry):
        ds_id = entry.dataset_id
        if ds_id not in self._arrays_cache:
            self._arrays_cache[ds_id] = open_csr_arrays(
                entry.offsets_path,
                entry.indices_path,
                entry.counts_path,
                read_engine=self._read_engine,
            )
        arrays = self._arrays_cache[ds_id]
        return arrays.row_offsets, arrays.indices, arrays.counts

    def _read_local_cells(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        zarr_entry = self._find_entry_by_id(entry.dataset_id)
        assert isinstance(zarr_entry, ZarrDatasetEntry)
        offsets, indices_arr, counts_arr = self._open_arrays(zarr_entry)
        return _zarr_read_cells(
            offsets, indices_arr, counts_arr, local_indices
        )


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
        One of ``"lance"`` or ``"zarr"``.
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

    elif backend == "zarr":
        read_engine = kwargs.pop("read_engine", "zarr-python")
        if topology == "aggregate":
            offsets_path = kwargs.pop("offsets_path")
            indices_path = kwargs.pop("indices_path")
            counts_path = kwargs.pop("counts_path")
            return AggregateZarrReader(
                offsets_path,
                indices_path,
                counts_path,
                entries,
                read_engine=read_engine,
            )
        else:
            return FederatedZarrReader(entries, read_engine=read_engine)  # type: ignore[arg-type]

    raise ValueError(f"Unknown backend '{backend}'. Supported: lance, zarr.")


# ---------------------------------------------------------------------------
# Internal helper: np.searchsorted-based entry lookup
# ---------------------------------------------------------------------------


def _searchsorted_entry(stops: np.ndarray, global_index: int) -> int:
    """Return the index into *stops* array for *global_index*.

    Uses ``np.searchsorted(stops, global_index, side='right')`` which
    returns the first stop > *global_index*, i.e., the entry whose range
    contains *global_index* (if any).
    """
    return int(np.searchsorted(stops, global_index, side="right"))
