"""In-memory metadata views over the existing sparse corpus readers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import polars as pl
import yaml
from scipy import sparse

from .corpus_loader import Corpus
from .expression import BaseExpressionReader, DatasetEntry, FederatedLanceReader, LanceDatasetEntry
from .feature_registry import FeatureRegistry
from .index import MetadataIndex


def _registry(named_ids: Mapping[str, list[str]]) -> FeatureRegistry:
    # Canonical names here are internal coordinates, not required user columns.
    return FeatureRegistry({
        name: pl.DataFrame({"origin_index": np.arange(len(ids)), "canonical_gene_id": ids})
        for name, ids in named_ids.items()
    })


def _metadata(obs: pd.DataFrame, name: str, index: int, start: int, factors=None) -> pl.DataFrame:
    structural = {
        "global_row_index": np.arange(start, start + len(obs), dtype=np.int64),
        "dataset_index": np.full(len(obs), index, dtype=np.int32),
        "dataset_id": [name] * len(obs),
        "local_row_index": np.arange(len(obs), dtype=np.int64),
        "cell_id": obs.index.astype(str).to_numpy(),
    }
    if factors is not None:
        structural["size_factor"] = np.asarray(factors, dtype=np.float64)
    # User columns are kept untouched in Corpus.obs. Routing columns belong to
    # MetadataIndex, and take precedence only in that internal table.
    user = pl.from_pandas(obs.drop(columns=list(structural), errors="ignore"), include_index=False)
    return pl.DataFrame(structural).hstack(user) if user.width else pl.DataFrame(structural)


def load_standalone(root: Path) -> Corpus:
    manifest = yaml.safe_load((root / "corpus.yaml").read_text())
    if (manifest["format"], manifest["version"], manifest["backend"]) != ("perturb-data-lab", 1, "lance"):
        raise ValueError("Unsupported standalone corpus format/version/backend")
    obs = pd.read_parquet(root / "obs.parquet")
    var = pd.read_parquet(root / "var.parquet")
    if [len(obs), len(var)] != manifest["shape"]:
        raise ValueError("Stored metadata disagrees with manifest shape")
    name = root.name
    kind = manifest["expression_kind"]
    factors = None if kind == "non_count" else pd.read_parquet(root / "statistics.parquet")["size_factor"]
    if factors is not None and len(factors) != len(obs):
        raise ValueError("Statistics rows disagree with obs")
    ids = var.index.astype(str).tolist()
    if not var.index.is_unique:
        ids = [f"local:{i}" for i in range(len(var))]
    entry = LanceDatasetEntry(name, 0, len(obs), root / "expression.lance")
    return Corpus(
        expression_reader=FederatedLanceReader([entry]), feature_registry=_registry({name: ids}),
        metadata_index=MetadataIndex(_metadata(obs, name, 0, 0, factors)),
        dataset_entries=[entry], dataset_index_by_id={name: 0}, topology="federated",
        backend="lance", corpus_root=root, obs_frames={name: obs}, var_frames={name: var},
        expression_kinds={name: kind}, features_aligned=var.index.is_unique,
    )


class _ViewExpressionReader(BaseExpressionReader):
    """Route view-local rows to member readers; retain dataset-local gene indices."""

    def __init__(self, entries, members):
        super().__init__(entries)
        self.members = dict(members)

    def _read_local_cells(self, entry, local_indices):
        member = self.members[entry.dataset_id]
        batch = member.expression_reader.read_expression_flat(local_indices)
        return [(batch.row_gene_indices(i), batch.row_counts(i)) for i in range(batch.batch_size)]


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
    reserved = {"dataset_id", "dataset_index", "global_row_index", "local_row_index", "cell_id", "size_factor"}
    if reserved.intersection(obs_columns or {}):
        raise ValueError("Collective column names conflict with internal routing/statistics fields")
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
        factors = corpus.metadata_index.df.get_column("size_factor").to_numpy() if "size_factor" in corpus.metadata_index.df.columns else np.full(len(source_obs), np.nan)
        metadata_frames.append(_metadata(selected, name, index, start, factors))
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
    registry = _registry(named_ids)
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


def _read_aligned(reader, indices, local_to_output, width):
    batch = reader.read_expression_flat(list(indices))
    columns = local_to_output[batch.expressed_gene_indices]
    keep = columns >= 0
    # Prefix sums work for consecutive empty rows and a trailing empty row.
    prefix = np.concatenate(([0], np.cumsum(keep, dtype=np.int64)))
    offsets = prefix[batch.row_offsets]
    return sparse.csr_matrix(
        (batch.expression_counts[keep], columns[keep], offsets),
        shape=(len(indices), width),
    )


def view_to_anndata(
    corpus: Corpus, *, dataset_id=None, global_row_indices=None,
    obs_columns=None, var_join="exact", lazy=False, chunk_rows=4096,
):
    """AnnData handoff sharing the same sparse reader and feature coordinates."""
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
    global_to_output = np.full(registry.global_vocab_size, -1, dtype=np.int64)
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
        dtype = corpus.expression_reader.read_expression_flat([int(indices[0])]).expression_counts.dtype if lazy else None
        for start in range(0, len(indices), chunk_rows):
            selected_rows = indices[start:start + chunk_rows]
            args = (corpus.expression_reader, selected_rows, mapping, len(axis))
            if lazy:
                # Dtype is obtained from a bounded read, never a full dataset.
                blocks.append(da.from_delayed(
                    delayed(_read_aligned)(*args), shape=(len(selected_rows), len(axis)),
                    dtype=dtype, meta=sparse.csr_matrix((0, 0), dtype=dtype),
                ))
            else:
                blocks.append(_read_aligned(*args))
    expression = da.concatenate(blocks, axis=0) if lazy else sparse.vstack(blocks, format="csr")
    result = ad.AnnData(X=expression, obs=pd.concat(obs_parts), var=var)
    # AnnData sparse matrices encode absence as zero; retain the distinction.
    result.varm["feature_presence"] = registry.dataset_has_gene[ds_indices][:, axis].T.copy()
    result.uns["feature_presence_members"] = selected
    return result
