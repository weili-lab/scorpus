"""Corpus factory for materialized perturbation corpora.

``load_corpus()`` reconstructs a training-ready ``Corpus`` from a corpus
directory using canonical metadata as the source of truth. Slim main supports
only Lance and Zarr corpora in aggregate or federated topology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence, cast
import warnings

import numpy as np
import polars as pl
import yaml

from .expression import (
    DatasetEntry,
    ExpressionBatch,
    ExpressionReader,
    LanceDatasetEntry,
    ZarrDatasetEntry,
    build_expression_reader,
)
from .feature_registry import FeatureRegistry
from .index import MetadataIndex
from .zarr_reading import normalize_zarr_read_engine

from ..materializers.paths import resolve_corpus_paths

__all__ = [
    "Corpus",
    "load_corpus",
]
# ---------------------------------------------------------------------------
# Corpus dataclass
# ---------------------------------------------------------------------------


@dataclass
class Corpus:
    """Loaded corpus components.

    Attributes
    ----------
    expression_reader : ExpressionReader
        Backend-aware flat expression reader.
    feature_registry : FeatureRegistry
        Per-dataset local→global gene ID mapping.
    metadata_index : MetadataIndex
        Polars-backed flat-schema metadata table for all cells.
    dataset_entries : list[DatasetEntry]
        Backend-aware dataset routing entries.
    topology : str
        Corpus topology: ``"aggregate"`` or ``"federated"``.
    backend : str
        Storage backend key (normalized to ``build_expression_reader`` form).
    corpus_root : Path
        Absolute path to the corpus root directory.
    """

    expression_reader: ExpressionReader
    feature_registry: FeatureRegistry
    metadata_index: MetadataIndex
    dataset_entries: list[DatasetEntry] = field(default_factory=list)
    dataset_index_by_id: dict[str, int] = field(default_factory=dict)
    topology: str = ""
    backend: str = ""
    corpus_root: Path = Path()
    canonical_obs_paths: dict[str, Path] = field(default_factory=dict)
    canonical_var_paths: dict[str, Path] = field(default_factory=dict)
    obs_frames: dict[str, Any] = field(default_factory=dict)
    var_frames: dict[str, Any] = field(default_factory=dict)
    members: dict[str, Any] = field(default_factory=dict)
    expression_kinds: dict[str, str] = field(default_factory=dict)
    obs_view: Any = None
    var_view: Any = None
    features_aligned: bool = True

    @property
    def obs(self):
        """Original single-dataset obs, or the explicitly selected collective obs."""
        if self.obs_view is not None:
            return self.obs_view
        if len(self.obs_frames) == 1:
            return next(iter(self.obs_frames.values()))
        return self.metadata_index.df.to_pandas()

    @property
    def var(self):
        """Original single-dataset var, or the shared feature axis of a view."""
        import pandas as pd

        if self.var_view is not None:
            return self.var_view
        if len(self.var_frames) == 1 and not self.members:
            return next(iter(self.var_frames.values()))
        return pd.DataFrame(index=pd.Index(self.feature_registry.global_feature_ids, name="feature_id"))

    @property
    def feature_presence(self):
        """Dataset-by-feature availability; absent features are not measured zeros."""
        import pandas as pd

        return pd.DataFrame(
            self.feature_registry.dataset_has_gene,
            index=self.dataset_ids, columns=self.feature_registry.global_feature_ids,
        )

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        """Dataset IDs in corpus routing order."""
        return tuple(entry.dataset_id for entry in self.dataset_entries)

    def take_metadata(
        self,
        indices: np.ndarray | Sequence[int],
        *,
        columns: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray | tuple]:
        """Return columnar metadata for selected corpus-global row indices.

        Use this to recover provenance fields such as ``local_row_index`` or to
        inspect rich annotations without constructing a DataLoader.
        """
        normalized_indices = _normalize_batch_indices(indices)
        resolved_columns = _normalize_take_columns(self.metadata_index, columns)
        return self.metadata_index.take(normalized_indices, resolved_columns)

    def to_anndata(
        self,
        *,
        dataset_id: str | Sequence[str] | None = None,
        global_row_indices: Sequence[int] | np.ndarray | None = None,
        obs_columns: Sequence[str] | None = None,
        var_join: Literal["inner", "exact", "outer"] = "exact",
    ):
        """Export selected corpus rows as an in-memory AnnData object.

        Pass ``global_row_indices`` to export rows in that exact order. Selected
        rows must all belong to one dataset, whose var metadata and HVG rankings
        are attached to the output. Without row indices, ``dataset_id`` retains
        the whole-dataset export behavior. For multi-dataset exports,
        ``var_join="exact"`` requires a shared ordered ``canonical_gene_id``
        axis; ``var_join="inner"`` exports the feature intersection.
        """
        import anndata as ad
        from scipy import sparse

        if self.obs_frames:
            from .composition import view_to_anndata

            return view_to_anndata(
                self, dataset_id=dataset_id, global_row_indices=global_row_indices,
                obs_columns=obs_columns, var_join=var_join, lazy=False,
            )
        if global_row_indices is not None:
            indices, selected = _resolve_single_dataset_global_indices(
                self,
                global_row_indices,
                dataset_id=dataset_id,
            )
            var_df, n_vars, _mapping = _resolve_var_axis(self, selected, "exact")
            var_df = _add_dataset_hvg_columns(self, var_df, selected[0])
            obs = _build_obs_dataframe_for_indices(
                self,
                indices,
                obs_columns=obs_columns,
            )
            batch = self.expression_reader.read_expression_flat(indices.tolist())
            x = _expression_batch_to_csr(batch, n_vars=n_vars)
            adata = ad.AnnData(X=x, obs=obs, var=_var_dataframe_to_pandas(var_df))
            if not adata.obs_names.is_unique:
                adata.obs_names_make_unique()
            return adata

        if dataset_id is None:
            raise ValueError("dataset_id is required when global_row_indices is not provided")
        selected = _normalize_dataset_selection(dataset_id)
        var_df, n_vars, mapping = _resolve_var_axis(self, selected, var_join)
        if len(selected) == 1:
            var_df = _add_dataset_hvg_columns(self, var_df, selected[0])
        obs = _build_obs_dataframe(self, selected, obs_columns=obs_columns)

        if mapping is None and len(selected) == 1:
            global_indices = _selected_dataset_global_indices(self, selected)
            batch = self.expression_reader.read_expression_flat(global_indices.tolist())
            x = _expression_batch_to_csr(batch, n_vars=n_vars)
        else:
            entries = _entry_map(self)
            blocks = []
            for ds_id in selected:
                entry = entries[ds_id]
                indices = np.arange(entry.global_start, entry.global_end, dtype=np.int64)
                batch = self.expression_reader.read_expression_flat(indices.tolist())
                lm = mapping[ds_id] if mapping is not None else None
                blocks.append(_expression_batch_to_csr(batch, n_vars=n_vars, local_to_output=lm))
            x = sparse.vstack(blocks)

        return ad.AnnData(X=x, obs=obs, var=_var_dataframe_to_pandas(var_df))

    def to_anndata_lazy(
        self,
        *,
        dataset_id: str | Sequence[str] | None = None,
        obs_columns: Sequence[str] | None = None,
        chunk_rows: int = 4096,
        device: str = "cpu",
        var_join: Literal["inner", "exact", "outer"] = "exact",
    ):
        """Export whole selected dataset(s) as AnnData with Dask-backed ``X``.

        Only ``X`` is lazy. Observation and feature metadata are loaded in
        memory because they are small compared with the count matrix.
        """
        import anndata as ad

        if self.obs_frames:
            from .composition import view_to_anndata

            if device != "cpu":
                raise ValueError("Metadata-view AnnData currently supports device='cpu'")
            return view_to_anndata(
                self, dataset_id=dataset_id, obs_columns=obs_columns,
                var_join=var_join, lazy=True, chunk_rows=chunk_rows,
            )
        if int(chunk_rows) <= 0:
            raise ValueError("chunk_rows must be positive")
        lazy_device = _normalize_lazy_device(device)
        if dataset_id is None:
            dataset_id = self.dataset_ids
        selected = _normalize_dataset_selection(dataset_id)
        var_df, n_vars, mapping = _resolve_var_axis(self, selected, var_join)
        obs = _build_obs_dataframe(self, selected, obs_columns=obs_columns)
        x = _build_lazy_expression_matrix(
            self,
            selected,
            n_vars=n_vars,
            chunk_rows=int(chunk_rows),
            device=lazy_device,
            per_dataset_local_to_output=mapping,
        )
        return ad.AnnData(X=x, obs=obs, var=_var_dataframe_to_pandas(var_df))

    def write_h5ad(
        self,
        path: str | Path,
        *,
        dataset_id: str | Sequence[str] | None = None,
        var_join: Literal["inner", "exact", "outer"] = "exact",
        chunk_rows: int = 4096,
    ) -> None:
        """Write a new h5ad through AnnData's chunked Dask writer.

        Reopen with anndata.read_h5ad(path, backed='r') for a native backed
        AnnData. This copies expression to a new file; to_anndata_lazy does not.
        """
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        self.to_anndata_lazy(
            dataset_id=dataset_id, var_join=var_join, chunk_rows=chunk_rows,
        ).write_h5ad(path)

    def add_obs_meta(
        self,
        frame: Any,
        *,
        on: Sequence[str],
    ) -> None:
        """Join additional observation metadata into this loaded corpus.

        This is runtime-only. The incoming frame must cover every corpus row
        exactly once using explicit join keys.
        """
        previous_columns = set(self.metadata_index.df.columns)
        self.metadata_index.df = _join_obs_metadata(
            self.metadata_index.df,
            frame,
            on=on,
        ).sort("global_row_index")
        if self.obs_frames:
            self.obs_view = self.obs.copy()
            for column in self.metadata_index.df.columns:
                if column not in previous_columns:
                    self.obs_view[column] = self.metadata_index.df[column].to_numpy()

# ---------------------------------------------------------------------------
# Backend name normalisation
# ---------------------------------------------------------------------------

def _normalize_backend(raw: str) -> str:
    """Map corpus-index backend strings to ``build_expression_reader`` keys."""
    if raw not in {"lance", "zarr"}:
        raise ValueError(
            f"Unsupported corpus backend '{raw}'. "
            "Supported: ['lance', 'zarr']"
        )
    return raw


def _build_range_entries(
    global_ranges: Sequence[tuple[str, int, int, int]],
) -> list[DatasetEntry]:
    return [
        DatasetEntry(
            dataset_id=ds_id,
            global_start=g_start,
            global_end=g_end,
        )
        for ds_id, _dsi, g_start, g_end in global_ranges
    ]


def _resolve_zarr_csr_paths(matrix_root: Path, stem: str) -> tuple[Path, Path, Path]:
    csr_path = matrix_root / f"{stem}-csr.zarr"
    if csr_path.is_dir():
        return csr_path, csr_path, csr_path

    row_offsets_path = matrix_root / f"{stem}-row-offsets.zarr"
    indices_path = matrix_root / f"{stem}-indices.zarr"
    counts_path = matrix_root / f"{stem}-counts.zarr"
    if not row_offsets_path.is_dir():
        raise FileNotFoundError(
            f"Zarr row-offsets artifact not found for '{stem}': {row_offsets_path}"
        )
    if not indices_path.is_dir():
        raise FileNotFoundError(
            f"Zarr indices artifact not found for '{stem}': {indices_path}"
        )
    if not counts_path.is_dir():
        raise FileNotFoundError(
            f"Zarr counts artifact not found for '{stem}': {counts_path}"
        )
    return row_offsets_path, indices_path, counts_path


def _build_aggregate_expression_components(
    root: Path,
    backend: str,
    global_ranges: Sequence[tuple[str, int, int, int]],
    *,
    zarr_read_engine: str,
) -> tuple[list[DatasetEntry], ExpressionReader]:
    entries = _build_range_entries(global_ranges)
    if backend == "lance":
        lance_path = root / "matrix" / "aggregated-cells.lance"
        if not lance_path.exists():
            raise FileNotFoundError(
                f"Aggregate Lance file not found: {lance_path}"
            )
        return entries, build_expression_reader(
            backend,
            "aggregate",
            entries,
            lance_path=str(lance_path),
        )

    row_offsets_path, indices_path, counts_path = _resolve_zarr_csr_paths(
        root / "matrix", "aggregated"
    )
    return entries, build_expression_reader(
        backend,
        "aggregate",
        entries,
        offsets_path=str(row_offsets_path),
        indices_path=str(indices_path),
        counts_path=str(counts_path),
        read_engine=zarr_read_engine,
    )


def _build_federated_dataset_entry(
    root: Path,
    backend: str,
    dataset_id: str,
    global_start: int,
    global_end: int,
) -> LanceDatasetEntry | ZarrDatasetEntry:
    matrix_root = resolve_corpus_paths("federated", root, dataset_id).matrix_root
    if backend == "lance":
        lance_path = matrix_root / f"{dataset_id}.lance"
        if not lance_path.exists():
            raise FileNotFoundError(
                f"Lance file not found for dataset '{dataset_id}': "
                f"{lance_path}"
            )
        return LanceDatasetEntry(
            dataset_id=dataset_id,
            global_start=global_start,
            global_end=global_end,
            lance_path=str(lance_path),
        )

    row_offsets_path, indices_path, counts_path = _resolve_zarr_csr_paths(
        matrix_root, dataset_id
    )
    return ZarrDatasetEntry(
        dataset_id=dataset_id,
        global_start=global_start,
        global_end=global_end,
        offsets_path=str(row_offsets_path),
        indices_path=str(indices_path),
        counts_path=str(counts_path),
    )


def _build_federated_expression_components(
    root: Path,
    backend: str,
    global_ranges: Sequence[tuple[str, int, int, int]],
    *,
    zarr_read_engine: str,
) -> tuple[list[DatasetEntry], ExpressionReader]:
    entries = [
        _build_federated_dataset_entry(root, backend, ds_id, g_start, g_end)
        for ds_id, _dsi, g_start, g_end in global_ranges
    ]
    return cast(list[DatasetEntry], entries), build_expression_reader(
        backend,
        "federated",
        cast(list[DatasetEntry], entries),
        read_engine=zarr_read_engine,
    )


def _build_expression_components(
    root: Path,
    topology: str,
    backend: str,
    global_ranges: Sequence[tuple[str, int, int, int]],
    *,
    zarr_read_engine: str,
) -> tuple[list[DatasetEntry], ExpressionReader]:
    if topology == "aggregate":
        return _build_aggregate_expression_components(
            root,
            backend,
            global_ranges,
            zarr_read_engine=zarr_read_engine,
        )
    if topology == "federated":
        return _build_federated_expression_components(
            root,
            backend,
            global_ranges,
            zarr_read_engine=zarr_read_engine,
        )
    raise ValueError(
        f"Unknown topology '{topology}'. "
        f"Expected 'aggregate' or 'federated'."
    )


def _normalize_take_columns(
    metadata_index: MetadataIndex,
    columns: Sequence[str] | None,
) -> tuple[str, ...]:
    """Validate columns for ``Corpus.take_metadata(...)``."""
    if columns is None:
        return tuple(metadata_index.df.columns)
    if isinstance(columns, (str, bytes)):
        raise TypeError("columns must be a sequence of column names")
    resolved: list[str] = []
    for name in columns:
        if not isinstance(name, str):
            raise TypeError("columns must contain strings")
        if name not in metadata_index.df.columns:
            raise ValueError(f"metadata column {name!r} not found")
        if name not in resolved:
            resolved.append(name)
    return tuple(resolved)


def _normalize_batch_indices(
    indices: np.ndarray | Sequence[int],
) -> np.ndarray:
    """Convert batch row indices to int64."""
    return np.asarray(indices, dtype=np.int64)


# ---------------------------------------------------------------------------
# AnnData and runtime metadata helpers
# ---------------------------------------------------------------------------


def _normalize_dataset_selection(dataset_id: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(dataset_id, str):
        return (dataset_id,)
    return tuple(dataset_id)


def _entry_map(corpus: Corpus) -> dict[str, DatasetEntry]:
    return {entry.dataset_id: entry for entry in corpus.dataset_entries}


def _selected_dataset_global_indices(
    corpus: Corpus,
    dataset_ids: Sequence[str],
) -> np.ndarray:
    entries = _entry_map(corpus)
    parts = [
        np.arange(entries[ds_id].global_start, entries[ds_id].global_end, dtype=np.int64)
        for ds_id in dataset_ids
    ]
    return np.concatenate(parts)


def _resolve_single_dataset_global_indices(
    corpus: Corpus,
    global_row_indices: Sequence[int] | np.ndarray,
    *,
    dataset_id: str | Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    indices = np.asarray(global_row_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("global_row_indices must be a 1-D sequence")
    if indices.size == 0:
        raise ValueError("global_row_indices must not be empty")
    if np.any(indices < 0) or np.any(indices >= len(corpus.metadata_index)):
        raise IndexError("global_row_indices contain out-of-range rows")

    row_dataset_ids = corpus.take_metadata(indices, columns=("dataset_id",))["dataset_id"]
    unique_dataset_ids = tuple(dict.fromkeys(str(value) for value in row_dataset_ids))
    if len(unique_dataset_ids) != 1:
        raise ValueError("global_row_indices must all belong to the same dataset")

    inferred_dataset_id = unique_dataset_ids[0]
    if dataset_id is not None:
        requested = _normalize_dataset_selection(dataset_id)
        if len(requested) != 1:
            raise ValueError("dataset_id must identify one dataset when global_row_indices are provided")
        if requested[0] != inferred_dataset_id:
            raise ValueError(
                f"global_row_indices belong to dataset {inferred_dataset_id!r}, "
                f"not requested dataset {requested[0]!r}"
            )
    return indices, (inferred_dataset_id,)


def _load_sorted_var_frame(corpus: Corpus, dataset_id: str) -> pl.DataFrame:
    var_df = pl.read_parquet(str(corpus.canonical_var_paths[dataset_id]))
    return var_df.with_columns(
        pl.col("origin_index").cast(pl.Int64, strict=True).alias("origin_index"),
        pl.col("canonical_gene_id").cast(pl.Utf8, strict=True).alias("canonical_gene_id"),
    ).sort("origin_index")


def _add_dataset_hvg_columns(
    corpus: Corpus,
    var_df: pl.DataFrame,
    dataset_id: str,
) -> pl.DataFrame:
    dataset_index = int(corpus.dataset_index_by_id[dataset_id])
    local_to_global = corpus.feature_registry.local_to_global_map[
        dataset_index,
        : len(var_df),
    ]
    hvg_rank = corpus.feature_registry.hvg_rank_matrix[
        dataset_index,
        local_to_global,
    ]
    highly_variable = corpus.feature_registry.hvg_mask[
        dataset_index,
        local_to_global,
    ]
    return var_df.with_columns(
        pl.Series("hvg_rank", hvg_rank.astype(np.int32, copy=False)),
        pl.Series("highly_variable", highly_variable.astype(bool, copy=False)),
    )


def _resolve_var_axis(
    corpus: Corpus,
    dataset_ids: Sequence[str],
    var_join: Literal["inner", "exact"],
) -> tuple[pl.DataFrame, int, dict[str, np.ndarray] | None]:
    """Resolve shared var axis across selected datasets.

    Returns ``(var_df, n_vars, per_dataset_local_to_output)`` where
    ``per_dataset_local_to_output`` is ``None`` when axes match exactly
    (local gene index == output column index for all datasets).
    """
    if var_join not in {"inner", "exact"}:
        raise ValueError("Legacy corpus export supports var_join='inner' or 'exact'; use a metadata view for outer joins")
    named_vars = {ds_id: _load_sorted_var_frame(corpus, ds_id) for ds_id in dataset_ids}
    first_id = dataset_ids[0]
    first_genes = named_vars[first_id]["canonical_gene_id"].to_list()

    axes_match = True
    for ds_id in dataset_ids[1:]:
        if named_vars[ds_id]["canonical_gene_id"].to_list() != first_genes:
            axes_match = False
            break

    if axes_match or var_join == "exact":
        if not axes_match:
            # axes_match implies we entered the loop above, so ds_id from
            # the first mismatching iteration is bound
            raise ValueError(
                "selected datasets do not share the same ordered canonical_gene_id axis"
            )
        return named_vars[first_id], len(first_genes), None

    intersection = [g for g in first_genes if all(
        g in named_vars[ds]["canonical_gene_id"] for ds in dataset_ids[1:]
    )]
    if not intersection:
        raise ValueError(
            "no common genes across selected datasets"
        )

    warnings.warn(
        f"selected datasets do not share identical ordered canonical_gene_id axes; "
        f"exporting {len(intersection)} intersection genes",
        UserWarning,
    )

    gene_to_out = {g: i for i, g in enumerate(intersection)}
    var_df = named_vars[first_id].filter(
        pl.col("canonical_gene_id").is_in(intersection)
    )

    mapping: dict[str, np.ndarray] = {}
    for ds_id in dataset_ids:
        local_genes = named_vars[ds_id]["canonical_gene_id"].to_list()
        arr = np.full(len(local_genes), -1, dtype=np.int32)
        for li, g in enumerate(local_genes):
            if g in gene_to_out:
                arr[li] = gene_to_out[g]
        mapping[ds_id] = arr

    return var_df, len(intersection), mapping


def _resolve_obs_columns(
    corpus: Corpus,
    obs_columns: Sequence[str] | None,
) -> tuple[str, ...]:
    if obs_columns is None:
        return tuple(corpus.metadata_index.df.columns)
    required = ("global_row_index", "dataset_id", "dataset_index", "local_row_index", "cell_id")
    return tuple(dict.fromkeys((*required, *obs_columns)))


def _build_obs_dataframe(
    corpus: Corpus,
    dataset_ids: Sequence[str],
    *,
    obs_columns: Sequence[str] | None,
):
    import pandas as pd

    columns = _resolve_obs_columns(corpus, obs_columns)
    entries = _entry_map(corpus)
    frames = []
    for ds_id in dataset_ids:
        entry = entries[ds_id]
        indices = np.arange(entry.global_start, entry.global_end, dtype=np.int64)
        frames.append(pd.DataFrame(corpus.take_metadata(indices, columns=columns)))
    obs = pd.concat(frames, ignore_index=True)
    obs.index = obs["global_row_index"].astype(str)
    obs.index.name = None
    return obs


def _build_obs_dataframe_for_indices(
    corpus: Corpus,
    global_row_indices: np.ndarray,
    *,
    obs_columns: Sequence[str] | None,
):
    import pandas as pd

    columns = _resolve_obs_columns(corpus, obs_columns)
    obs = pd.DataFrame(corpus.take_metadata(global_row_indices, columns=columns))
    obs.index = obs["global_row_index"].astype(str)
    if not obs.index.is_unique:
        obs.index = [f"{name}-{position}" for position, name in enumerate(obs.index)]
    obs.index.name = None
    return obs


def _var_dataframe_to_pandas(var_df: pl.DataFrame):
    var = var_df.to_pandas()
    var.index = var["canonical_gene_id"].astype(str)
    var.index.name = None
    return var


def _expression_batch_to_csr(batch: ExpressionBatch, *, n_vars: int, local_to_output: np.ndarray | None = None):
    from scipy import sparse

    counts = batch.expression_counts
    col_indices = batch.expressed_gene_indices
    row_offsets = batch.row_offsets
    if local_to_output is not None:
        col_indices = local_to_output[col_indices]
        keep = col_indices >= 0
        col_indices = col_indices[keep]
        counts = counts[keep]
        row_offsets = _recompute_row_offsets(batch.row_offsets, keep)
    return sparse.csr_matrix(
        (counts, col_indices, row_offsets),
        shape=(batch.batch_size, int(n_vars)),
        dtype=np.float32,
    )


def _recompute_row_offsets(orig_offsets: np.ndarray, keep: np.ndarray) -> np.ndarray:
    keep_int = keep.astype(np.int32)
    row_nnz = np.add.reduceat(keep_int, orig_offsets[:-1])
    new_offsets = np.zeros(len(orig_offsets), dtype=np.int64)
    new_offsets[1:] = np.cumsum(row_nnz)
    return new_offsets


def _expression_batch_to_cupy_csr(
    batch: ExpressionBatch,
    *,
    n_vars: int,
    device: str,
    local_to_output: np.ndarray | None = None,
):
    try:
        import cupy as cp
        from cupyx.scipy import sparse as cupyx_sparse
    except ImportError as exc:
        raise ImportError("device='cuda' requires cupy and cupyx") from exc

    counts = batch.expression_counts
    col_indices = batch.expressed_gene_indices
    row_offsets = batch.row_offsets
    if local_to_output is not None:
        col_indices = local_to_output[col_indices]
        keep = col_indices >= 0
        col_indices = col_indices[keep]
        counts = counts[keep]
        row_offsets = _recompute_row_offsets(batch.row_offsets, keep)

    try:
        with cp.cuda.Device(_cupy_device_id(device)):
            return cupyx_sparse.csr_matrix(
                (
                    cp.asarray(counts),
                    cp.asarray(col_indices),
                    cp.asarray(row_offsets),
                ),
                shape=(batch.batch_size, int(n_vars)),
                dtype=cp.float32,
            )
    except Exception as exc:
        raise RuntimeError("device='cuda' requires a working CUDA/CuPy runtime") from exc


def _normalize_lazy_device(device: str) -> str:
    normalized = str(device).lower()
    if normalized == "cpu" or normalized == "cuda":
        return normalized
    if normalized.startswith("cuda:"):
        _cupy_device_id(normalized)
        return normalized
    raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'")


def _cupy_device_id(device: str) -> int:
    if device == "cuda":
        return 0
    return int(device.split(":", 1)[1])


def _lazy_sparse_meta(device: str):
    if device == "cpu":
        from scipy import sparse

        return sparse.csr_matrix((0, 0), dtype=np.float32)

    try:
        import cupy as cp
        from cupyx.scipy import sparse as cupyx_sparse
    except ImportError as exc:
        raise ImportError("device='cuda' requires cupy and cupyx") from exc

    try:
        with cp.cuda.Device(_cupy_device_id(device)):
            return cupyx_sparse.csr_matrix((0, 0), dtype=cp.float32)
    except Exception as exc:
        raise RuntimeError("device='cuda' requires a working CUDA/CuPy runtime") from exc


def _read_expression_sparse_block(
    reader: ExpressionReader,
    start: int,
    stop: int,
    n_vars: int,
    device: str,
    local_to_output: np.ndarray | None = None,
):
    indices = list(range(int(start), int(stop)))
    batch = reader.read_expression_flat(indices)
    if device == "cpu":
        return _expression_batch_to_csr(batch, n_vars=n_vars, local_to_output=local_to_output)
    return _expression_batch_to_cupy_csr(batch, n_vars=n_vars, device=device, local_to_output=local_to_output)


def _build_lazy_expression_matrix(
    corpus: Corpus,
    dataset_ids: Sequence[str],
    *,
    n_vars: int,
    chunk_rows: int,
    device: str,
    per_dataset_local_to_output: dict[str, np.ndarray] | None = None,
):
    import dask.array as da
    from dask import delayed

    entries = _entry_map(corpus)
    blocks = []
    meta = _lazy_sparse_meta(device)
    for ds_id in dataset_ids:
        entry = entries[ds_id]
        mapping = per_dataset_local_to_output.get(ds_id) if per_dataset_local_to_output is not None else None
        for start in range(entry.global_start, entry.global_end, int(chunk_rows)):
            stop = min(start + int(chunk_rows), entry.global_end)
            task = delayed(_read_expression_sparse_block)(
                corpus.expression_reader,
                start,
                stop,
                int(n_vars),
                device,
                mapping,
            )
            blocks.append(
                da.from_delayed(
                    task,
                    shape=(stop - start, int(n_vars)),
                    dtype=np.float32,
                    meta=meta,
                )
            )
    return da.concatenate(blocks, axis=0)


def _frame_to_polars(frame: Any) -> pl.DataFrame:
    if isinstance(frame, pl.DataFrame):
        return frame
    try:
        import pandas as pd

        if isinstance(frame, pd.DataFrame):
            return pl.from_pandas(frame)
    except ImportError:
        pass
    return pl.DataFrame(frame)


def _normalize_join_keys(on: Sequence[str]) -> tuple[str, ...]:
    if isinstance(on, (str, bytes)):
        raise TypeError("on must be a sequence of join-key column names")
    keys = tuple(str(column) for column in on)
    if not keys:
        raise ValueError("on must contain at least one join key")
    if len(set(keys)) != len(keys):
        raise ValueError("on join keys must be unique")
    if any(not key for key in keys):
        raise ValueError("on join keys must be non-empty strings")
    return keys


def _has_duplicate_keys(df: pl.DataFrame, keys: Sequence[str]) -> bool:
    return bool(df.select(list(keys)).is_duplicated().any())


def _check_join_key_columns(df: pl.DataFrame, keys: Sequence[str], *, context: str) -> None:
    missing = [key for key in keys if key not in df.columns]
    if missing:
        raise ValueError(f"{context} missing join key column(s): {missing}")
    null_keys = [key for key in keys if int(df[key].null_count()) > 0]
    if null_keys:
        raise ValueError(f"{context} has null values in join key column(s): {null_keys}")


def _join_obs_metadata(
    corpus_df: pl.DataFrame,
    frame: Any,
    *,
    on: Sequence[str],
) -> pl.DataFrame:
    keys = _normalize_join_keys(on)
    incoming = _frame_to_polars(frame)
    _check_join_key_columns(corpus_df, keys, context="corpus metadata")
    _check_join_key_columns(incoming, keys, context="incoming metadata")

    value_columns = [column for column in incoming.columns if column not in keys]
    if not value_columns:
        raise ValueError("incoming metadata must contain at least one non-key column")
    collisions = [column for column in value_columns if column in corpus_df.columns]
    if collisions:
        raise ValueError(f"incoming metadata column(s) already exist in corpus: {collisions}")

    if _has_duplicate_keys(corpus_df, keys):
        raise ValueError("corpus metadata join keys are not unique")
    if _has_duplicate_keys(incoming, keys):
        raise ValueError("incoming metadata join keys are not unique")

    corpus_keys = corpus_df.select(list(keys))
    incoming_keys = incoming.select(list(keys))
    missing_rows = corpus_keys.join(incoming_keys, on=list(keys), how="anti")
    if missing_rows.height:
        raise ValueError("incoming metadata does not cover every corpus row")
    extra_rows = incoming_keys.join(corpus_keys, on=list(keys), how="anti")
    if extra_rows.height:
        raise ValueError("incoming metadata contains rows not present in corpus")

    return corpus_df.join(
        incoming.select(list(keys) + value_columns),
        on=list(keys),
        how="left",
    )


# ---------------------------------------------------------------------------
# load_corpus factory
# ---------------------------------------------------------------------------


def load_corpus(
    corpus_root: str | Path,
    *,
    extra_metadata_columns: Sequence[str] | None = None,
    zarr_read_engine: str = "zarr-python",
) -> Corpus:
    """Load a training-ready ``Corpus`` from a corpus directory.

    Reads ``corpus-index.yaml``, locates canonical obs/var parquets via
    ``resolve_corpus_paths()``, and constructs expression, metadata, and
    feature-registry components.

    Parameters
    ----------
    corpus_root : str or Path
        Path to a corpus directory containing ``corpus-index.yaml``.
    extra_metadata_columns : sequence of str, optional
        Additional canonical-obs parquet columns to load into
        ``metadata_index`` beyond the default canonical/core projection.

    Returns
    -------
    Corpus
        Fully-constructed corpus components ready for ``build_loader(...)`` or
        direct expression-reader / ``take_metadata(...)`` inspection.

    Raises
    ------
    FileNotFoundError
        If ``corpus-index.yaml``, any canonical parquet file, or any required
        backend matrix artifact is missing.
    ValueError
        If the corpus topology or backend is unsupported.
    """
    root = Path(corpus_root).resolve()
    if (root / "corpus.yaml").exists():
        from .composition import load_standalone

        return load_standalone(root)
    index_path = root / "corpus-index.yaml"
    if not index_path.exists():
        raise FileNotFoundError(
            f"corpus-index.yaml not found at {index_path}"
        )

    # ------------------------------------------------------------------
    # 1. Parse corpus-index.yaml
    # ------------------------------------------------------------------
    with open(index_path, encoding="utf-8") as handle:
        index_doc = yaml.safe_load(handle) or {}
    metadata = index_doc.get("global_metadata", {})
    topology = str(metadata.get("topology", ""))
    raw_backend = str(metadata.get("backend", ""))
    backend = _normalize_backend(raw_backend)
    resolved_zarr_read_engine = normalize_zarr_read_engine(zarr_read_engine)
    if backend != "zarr" and resolved_zarr_read_engine != "zarr-python":
        raise ValueError(
            "zarr_read_engine is only supported when loading a Zarr corpus"
        )
    datasets = index_doc.get("datasets", [])
    if not datasets:
        raise ValueError(
            f"No datasets list in corpus-index.yaml: {index_path}"
        )

    # ------------------------------------------------------------------
    # 2. Resolve canonical obs/var paths and matrix paths
    # ------------------------------------------------------------------
    canonical_obs_paths: dict[str, Path] = {}
    canonical_var_paths: dict[str, Path] = {}
    global_ranges: list[tuple[str, int, int, int]] = []
    # (dataset_id, dataset_index, global_start, global_end)

    for ds_entry in datasets:
        ds_id = str(ds_entry["dataset_id"])
        ds_index = int(ds_entry.get("dataset_index", 0))
        g_start = int(ds_entry.get("global_start", 0))
        g_end = int(ds_entry.get("global_end", 0))

        paths = resolve_corpus_paths(topology, root, ds_id)
        obs_path = paths.canonical_meta_root / "canonical-obs.parquet"
        var_path = paths.canonical_meta_root / "canonical-var.parquet"

        if not obs_path.exists():
            raise FileNotFoundError(
                f"canonical-obs.parquet not found for dataset '{ds_id}' "
                f"at {obs_path}"
            )
        if not var_path.exists():
            raise FileNotFoundError(
                f"canonical-var.parquet not found for dataset '{ds_id}' "
                f"at {var_path}"
            )

        canonical_obs_paths[ds_id] = obs_path
        canonical_var_paths[ds_id] = var_path
        global_ranges.append((ds_id, ds_index, g_start, g_end))

    # ------------------------------------------------------------------
    # 3. Build MetadataIndex from canonical obs parquets
    # ------------------------------------------------------------------
    metadata_index = MetadataIndex.from_canonical_obs_parquets(
        datasets_info=global_ranges,
        obs_paths=canonical_obs_paths,
        extra_metadata_columns=extra_metadata_columns,
    )

    # ------------------------------------------------------------------
    # 4. Build dataset entries and ExpressionReader
    # ------------------------------------------------------------------
    entries, expression_reader = _build_expression_components(
        root,
        topology,
        backend,
        global_ranges,
        zarr_read_engine=resolved_zarr_read_engine,
    )

    # ------------------------------------------------------------------
    # 5. Build FeatureRegistry from canonical var parquets
    # ------------------------------------------------------------------
    var_path_map: dict[str, str] = {
        ds_id: str(p) for ds_id, p in canonical_var_paths.items()
    }
    dataset_order = [ds_id for ds_id, *_ in global_ranges]
    feature_registry = FeatureRegistry.from_canonical_var_parquets(
        var_path_map,
        dataset_order=dataset_order,
    )

    # ------------------------------------------------------------------
    # 6. Return Corpus
    # ------------------------------------------------------------------
    return Corpus(
        expression_reader=expression_reader,
        feature_registry=feature_registry,
        metadata_index=metadata_index,
        dataset_entries=list(entries),
        dataset_index_by_id={ds_id: ds_index for ds_id, ds_index, *_ in global_ranges},
        topology=topology,
        backend=backend,
        corpus_root=root,
        canonical_obs_paths=canonical_obs_paths,
        canonical_var_paths=canonical_var_paths,
    )
