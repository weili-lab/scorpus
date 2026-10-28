"""In-memory metadata views over the existing sparse corpus readers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import polars as pl
import yaml
from scipy import sparse

from .corpus_loader import Corpus, _cupy_device_id, _lazy_sparse_meta, ex_path
from .expression import DatasetEntry, FederatedLanceReader, LanceDatasetEntry, SortedTakeReader
from .feature_registry import FeatureRegistry
from .index import MetadataIndex


def _hvg_ranks(var: pd.DataFrame) -> pl.DataFrame:
    """Gene-index HVG ranks from scanpy's var columns ``highly_variable`` and ``highly_variable_rank``.

    Only the ``highly_variable`` genes are ranked, 1..n in ``highly_variable_rank``
    order (scanpy's rank is 0-based and, with ``batch_key``, a median across
    batches, so ties keep gene order); all other genes get 0. The default HVG
    count is n, so the gene index's HVG mask is exactly ``highly_variable``.
    """
    selected = np.flatnonzero(var["highly_variable"].to_numpy(dtype=bool))
    rank = var["highly_variable_rank"].to_numpy(dtype=np.float64)
    order = selected[np.argsort(rank[selected], kind="stable")]
    return pl.DataFrame({
        "origin_index": order,
        "hvg_rank": np.arange(1, len(order) + 1),
        "selected_at_default_n_hvg": np.ones(len(order), dtype=bool),
    })


def _registry(named_ids: Mapping[str, list[str]], var_frames: Mapping[str, pd.DataFrame]) -> FeatureRegistry:
    # Canonical names here are internal coordinates, not required user columns.
    return FeatureRegistry(
        {
            name: pl.DataFrame({"origin_index": np.arange(len(ids)), "canonical_gene_id": ids})
            for name, ids in named_ids.items()
        },
        named_hvg_rank_dfs={name: _hvg_ranks(var) for name, var in var_frames.items() if "highly_variable" in var},
    )


# Routing fields that live only in MetadataIndex, never taken from user obs.
ROUTING_COLUMNS = ("global_row_index", "dataset_index", "dataset_id", "local_row_index", "cell_id")


def _metadata(obs: pd.DataFrame, name: str, index: int, start: int) -> pl.DataFrame:
    structural = {
        "global_row_index": np.arange(start, start + len(obs), dtype=np.int64),
        "dataset_index": np.full(len(obs), index, dtype=np.int32),
        "dataset_id": [name] * len(obs),
        "local_row_index": np.arange(len(obs), dtype=np.int64),
        "cell_id": obs.index.astype(str).to_numpy(),
    }
    # User columns are kept untouched in Corpus.obs. Routing columns belong to
    # MetadataIndex, and take precedence only in that internal table.
    user = pl.from_pandas(obs.drop(columns=list(ROUTING_COLUMNS), errors="ignore"), include_index=False)
    return pl.DataFrame(structural).hstack(user) if user.width else pl.DataFrame(structural)


# Corpora written before the scorpus rename carry the old format identifier.
STANDALONE_FORMATS = ("scorpus", "perturb-data-lab")


def load_standalone(root: Path, ex_obs: bool | str | None = None, ex_var: bool | str | None = None) -> Corpus:
    manifest = yaml.safe_load((root / "corpus.yaml").read_text())
    if (
        manifest["format"] not in STANDALONE_FORMATS
        or (manifest["version"], manifest["backend"]) != (1, "lance")
    ):
        raise ValueError("Unsupported standalone corpus format/version/backend")
    # An ex file holds the full obs/var (original plus saved columns) and replaces it.
    obs = pd.read_parquet(ex_path(root, "obs", ex_obs) if ex_obs else root / "obs.parquet")
    var = pd.read_parquet(ex_path(root, "var", ex_var) if ex_var else root / "var.parquet")
    if [len(obs), len(var)] != manifest["shape"]:
        raise ValueError("Stored metadata disagrees with manifest shape")
    name = root.name
    kind = manifest["expression_kind"]
    ids = var.index.astype(str).tolist()
    if not var.index.is_unique:
        ids = [f"local:{i}" for i in range(len(var))]
    # A table without an `expression_layout` entry is two-column; a packed
    # table's counts dtype is the corpus `dtype`.
    layout = manifest.get("expression_layout", "two_column")
    counts_dtype = manifest["dtype"] if layout == "packed" else None
    entry = LanceDatasetEntry(name, 0, len(obs), root / "expression.lance", layout, counts_dtype)
    return Corpus(
        expression_reader=FederatedLanceReader([entry]), feature_registry=_registry({name: ids}, {name: var}),
        metadata_index=MetadataIndex(_metadata(obs, name, 0, 0)),
        dataset_entries=[entry], dataset_index_by_id={name: 0}, topology="federated",
        backend="lance", corpus_root=root, obs_frames={name: obs}, var_frames={name: var},
        expression_kinds={name: kind}, features_aligned=var.index.is_unique,
    )


class _ViewExpressionReader(SortedTakeReader):
    """Route view-local rows to member readers; retain dataset-local gene indices.

    Members must be Lance readers (``SortedTakeReader`` subclasses).
    """

    def __init__(self, entries, members):
        super().__init__(entries)
        # Only the members' readers, not the member corpora (obs, var): the
        # reader then pickles as table paths and row ranges, e.g. into DataLoader workers.
        self.readers = {name: corpus.expression_reader for name, corpus in members.items()}

    def _take_rows(self, entry, rows):
        # Each member holds one table, so view-local rows are the member's
        # global rows. Its blocks join this batch directly, copied only once.
        reader = self.readers[entry.dataset_id]
        return reader._take_rows(reader._entries[0], rows)


def concat(
    corpora: Mapping[str, Corpus],
    *,
    obs_columns: Mapping[str, Mapping[str, str]] | None = None,
    obs: Mapping[str, pd.DataFrame] | None = None,
    var: Mapping[str, pd.DataFrame] | None = None,
    feature_key: Mapping[str, str] | None = None,
    var_columns: Mapping[str, Mapping[str, str]] | None = None,
) -> Corpus:
    """Reference independently converted datasets without rewriting expression.

    ``obs_columns`` maps collective names to {member: original column}.
    Optional ``obs``/``var`` tables let callers use ordinary pandas transforms;
    their indices and ordering must exactly match the corresponding member.
    ``feature_key`` chooses each member's shared feature identity column; use
    ``'index'`` for var_names. Without it, features remain member-local and a
    multi-member AnnData export requires explicit alignment first.

    Each input must contain one dataset. Original member tables remain available
    through ``result.members``; only requested columns enter collective obs.
    """
    if not corpora or any(not isinstance(k, str) or not k for k in corpora):
        raise ValueError("corpora must map nonempty member names to corpora")
    for tables in (obs, var, feature_key):
        if tables is not None and set(tables) != set(corpora):
            raise ValueError("Metadata/feature mappings must cover exactly the corpus members")
    entries, frames, metadata_frames, named_ids = [], [], [], {}
    obs_frames, var_frames, kinds = {}, {}, {}
    start = 0
    if set(ROUTING_COLUMNS).intersection(obs_columns or {}):
        raise ValueError("Collective column names conflict with internal routing fields")
    for index, (name, corpus) in enumerate(corpora.items()):
        if len(corpus.dataset_ids) != 1:
            raise ValueError("concat inputs must each contain one dataset")
        source_obs = corpus.obs if obs is None else obs[name]
        source_var = corpus.var if var is None else var[name]
        if not source_obs.index.equals(corpus.obs.index) or not source_var.index.equals(corpus.var.index):
            raise ValueError(f"{name}: supplied metadata must retain exact original row identity/order")
        obs_frames[name], var_frames[name] = source_obs.copy(), source_var.copy()
        selected = pd.DataFrame(index=source_obs.index)
        for column, mapping in (obs_columns or {}).items():
            if set(mapping) != set(corpora):
                raise ValueError(f"{column}: mapping must cover exactly the corpus members")
            selected[column] = source_obs[mapping[name]]
        metadata_frames.append(_metadata(selected, name, index, start))
        selected.index = pd.MultiIndex.from_arrays(
            [[name] * len(selected), np.arange(len(selected))], names=["member", "row"],
        )
        frames.append(selected)
        if feature_key is None:
            # JSON tuple encoding prevents namespace collisions with user IDs.
            import json
            ids = [json.dumps([name, i]) for i in range(len(source_var))]
        else:
            key = feature_key[name]
            values = source_var.index if key == "index" else source_var[key]
            if pd.isna(values).any() or any(not isinstance(v, str) or not v for v in values):
                raise ValueError(f"{name}: feature identities must be nonempty strings without nulls")
            ids = list(values)
            if len(set(ids)) != len(ids):
                raise ValueError(f"{name}: duplicate mapped features; resolve them before alignment")
        named_ids[name] = ids
        entries.append(DatasetEntry(name, start, start + len(source_obs)))
        start += len(source_obs)
        kinds[name] = corpus.expression_kinds.get(corpus.dataset_ids[0], "counts")
    # Strict vertical concatenation catches incompatible collective column types.
    metadata = MetadataIndex(pl.concat(metadata_frames, how="vertical"))
    registry = _registry(named_ids, var_frames)
    collective_var = pd.DataFrame(index=pd.Index(registry.global_feature_ids, name="feature_id"))
    for column, mapping in (var_columns or {}).items():
        if set(mapping) != set(corpora):
            raise ValueError(f"{column}: mapping must cover exactly the corpus members")
        values = pd.concat([
            pd.Series(var_frames[name][mapping[name]].to_numpy(), index=named_ids[name])
            for name in corpora
        ])
        distinct = values.groupby(level=0, sort=False).nunique(dropna=True)
        if (distinct > 1).any():
            raise ValueError(f"{column}: conflicting feature annotations across members")
        collective_var[column] = values.groupby(level=0, sort=False).first().reindex(collective_var.index)
    return Corpus(
        expression_reader=_ViewExpressionReader(entries, corpora), feature_registry=registry,
        metadata_index=metadata, dataset_entries=entries,
        dataset_index_by_id={name: i for i, name in enumerate(corpora)},
        topology="federated", backend="view", members=dict(corpora),
        obs_frames=obs_frames, var_frames=var_frames, obs_view=pd.concat(frames),
        var_view=collective_var,
        expression_kinds=kinds, features_aligned=feature_key is not None,
    )


def _read_aligned(reader, indices, local_to_output, width, dtype=None, device="cpu"):
    batch = reader.read_expression_flat(list(indices))
    columns = local_to_output[batch.expressed_gene_indices]
    offsets = batch.row_offsets
    if columns.min(initial=0) < 0:
        # Drop genes outside the output axis (inner join). Prefix sums work
        # for consecutive empty rows and a trailing empty row.
        keep = columns >= 0
        prefix = np.concatenate(([0], np.cumsum(keep, dtype=np.int64)))
        offsets = prefix[offsets]
        # Boolean indexing already copies, so these chunks are writable.
        counts, columns = batch.expression_counts[keep], columns[keep]
    else:
        # The reader may return read-only views of Lance's buffers; copy so
        # each chunk is writable (e.g. for in-place log1p).
        counts = np.array(batch.expression_counts, dtype=dtype)
    matrix = sparse.csr_matrix((counts, columns, offsets), shape=(len(indices), width), dtype=dtype)
    if device == "cpu":
        return matrix
    import cupy as cp
    from cupyx.scipy import sparse as cupyx_sparse

    with cp.cuda.Device(_cupy_device_id(device)):
        return cupyx_sparse.csr_matrix(matrix)


def view_to_anndata(
    corpus: Corpus, *, dataset_id=None, global_row_indices=None,
    obs_columns=None, var_join="exact", lazy=False, chunk_rows=65_536, device="cpu",
):
    """AnnData handoff sharing the same sparse reader and feature coordinates.

    ``device`` ('cpu', 'cuda' or 'cuda:<index>') applies to the lazy export: on a
    GPU each Dask chunk is a CuPy CSR matrix.
    """
    import anndata as ad
    import dask.array as da
    from dask import delayed

    if chunk_rows <= 0 or var_join not in {"exact", "inner", "outer"}:
        raise ValueError("Use positive chunk_rows and var_join='exact', 'inner', or 'outer'")
    selected = list(corpus.dataset_ids) if dataset_id is None else ([dataset_id] if isinstance(dataset_id, str) else list(dataset_id))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select a nonempty set of distinct datasets")
    entries = {entry.dataset_id: entry for entry in corpus.dataset_entries}
    if global_row_indices is not None:
        rows = np.asarray(global_row_indices, dtype=np.int64)
        if rows.ndim != 1 or not len(rows) or np.any(rows < 0) or np.any(rows >= len(corpus.metadata_index)):
            raise ValueError("Invalid selected global row indices")
        owners = corpus.take_metadata(rows, columns=["dataset_id"])["dataset_id"]
        if len(set(owners)) != 1:
            raise ValueError("Selected-row export currently requires one dataset")
        selected = [owners[0]] if dataset_id is None else selected
        if selected != [owners[0]]:
            raise ValueError("Selected rows do not belong to the requested dataset")
    if len(selected) > 1 and not corpus.features_aligned:
        raise ValueError("Supply feature_key to concat before shared-axis AnnData export")
    ds_indices = [corpus.dataset_index_by_id[name] for name in selected]
    registry = corpus.feature_registry
    maps = [registry.local_to_global_map[i, :len(corpus.var_frames[name])] for i, name in zip(ds_indices, selected)]
    if len(selected) == 1:
        axis = maps[0]
        var = corpus.var_frames[selected[0]].copy()
    else:
        if var_join == "exact":
            if any(not np.array_equal(maps[0], m) for m in maps[1:]):
                raise ValueError("Selected datasets do not have identical ordered feature axes")
            axis = maps[0]
        elif var_join == "inner":
            axis = maps[0][np.isin(maps[0], np.flatnonzero(registry.dataset_has_gene[ds_indices].all(axis=0)))]
        else:
            axis = np.flatnonzero(registry.dataset_has_gene[ds_indices].any(axis=0))
        var = corpus.var.iloc[axis].copy()
    if not len(axis):
        raise ValueError("The selected feature axis is empty")
    # int32: the CSR uses these as its column indices without a conversion copy.
    global_to_output = np.full(registry.global_vocab_size, -1, dtype=np.int32)
    global_to_output[axis] = np.arange(len(axis))
    blocks, obs_parts = [], []
    for name, local_to_global in zip(selected, maps):
        entry = entries[name]
        indices = rows if global_row_indices is not None else np.arange(entry.global_start, entry.global_end)
        if corpus.obs_view is not None:
            frame = corpus.obs.iloc[indices].copy()
            if corpus.members:
                frame.index = pd.Index([f"{int(i)}" for i in indices])
        else:
            frame = corpus.obs_frames[name].iloc[indices - entry.global_start].copy()
        if obs_columns is not None:
            frame = frame[list(obs_columns)]
        obs_parts.append(frame)
        mapping = global_to_output[local_to_global]
        if lazy:
            # Dtype is obtained from a bounded read, never a full dataset.
            dtype = corpus.expression_reader.read_expression_flat([int(indices[0])]).expression_counts.dtype
            if device != "cpu" and dtype.kind in "iu":
                # CuPy sparse has no integer dtype.
                dtype = np.dtype(np.float32)
            meta = _lazy_sparse_meta(device, dtype)
        for start in range(0, len(indices), chunk_rows):
            selected_rows = indices[start:start + chunk_rows]
            args = (corpus.expression_reader, selected_rows, mapping, len(axis))
            if lazy:
                blocks.append(da.from_delayed(
                    delayed(_read_aligned)(*args, dtype, device), shape=(len(selected_rows), len(axis)),
                    dtype=dtype, meta=meta,
                ))
            else:
                blocks.append(_read_aligned(*args))
    expression = da.concatenate(blocks, axis=0) if lazy else sparse.vstack(blocks, format="csr")
    if len(selected) == 1 and "highly_variable" in var:
        var["hvg_rank"] = registry.hvg_rank_matrix[ds_indices[0], axis]
    result = ad.AnnData(X=expression, obs=pd.concat(obs_parts), var=var)
    # AnnData sparse matrices encode absence as zero; retain the distinction.
    result.varm["feature_presence"] = registry.dataset_has_gene[ds_indices][:, axis].T.copy()
    result.uns["feature_presence_members"] = selected
    return result
