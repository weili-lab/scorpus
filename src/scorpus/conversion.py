"""Convert a selected h5ad matrix without imposing a biological metadata schema."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from .materializers.backends.lance import write_lance_federated
from .materializers.chunk_translation import ChunkBundle, _translate_chunk

if TYPE_CHECKING:
    from .loaders.corpus_loader import Corpus


NO_COUNTS = (
    'no raw count found, if proceeding, convert with '
    'layers="raw|X|layer_name" and no_count=True'
)
RECOVERY_POSSIBLE = 'raw count possible, please convert with attempt_conversion=True'

MANIFEST = {"format": "scorpus", "version": 1, "backend": "lance"}
SIZE_FACTOR_METHOD = "library_sum/median_positive_library_sum; empty_cells=1"

# Genes read per pass over an on-disk CSC matrix (see _csc_to_row_buckets).
CSC_GENES_PER_BLOCK = 512


def _matrix_path(name: str) -> str:
    return "raw/X" if name == "raw" else "X" if name == "X" else f"layers/{name}"


def _open(node):
    return ad.io.sparse_dataset(node) if isinstance(node, h5py.Group) else node


def _matrix(handle: h5py.File, name: str):
    return _open(handle[_matrix_path(name)])


def _is_csc(handle: h5py.File, name: str) -> bool:
    return handle[_matrix_path(name)].attrs.get("encoding-type") == "csc_matrix"


def _count_chunk(matrix, *, recover: bool, start: int = 0) -> ChunkBundle:
    chunk = sparse.csr_matrix(matrix)
    chunk.sum_duplicates()
    chunk.eliminate_zeros()
    return _translate_chunk(
        dataset_id="conversion", global_row_start=0, matrix_chunk=chunk,
        chunk_start=start, needs_recovery=recover,
    )


def _append(bundle: ChunkBundle, name: str, folder: Path, states: dict, last: bool) -> None:
    """Append one chunk to ``<folder>/<name>.lance``; ``states`` holds each table's writer state."""
    _, states[folder / name] = write_lance_federated(
        bundle, name, folder, _writer_state=states.get(folder / name), _is_last_chunk=last,
    )


def _choose_matrix(handle: h5py.File, n_obs: int, layers: str | None, attempt_conversion: bool) -> tuple[str, bool]:
    """Probe sampled rows of the candidate matrices; return (selected, needs_recovery).

    Without ``layers``, direct count candidates are preferred over recoverable
    ones; the widest gene axis wins, with counts-named layers, raw, then X
    breaking ties.
    """
    if layers is None:
        layer_names = list(handle.get("layers", {}))
        count_names = [n for n in layer_names if "count" in n.lower()]
        names = count_names + (["raw"] if "raw/X" in handle else [])
        names += (["X"] if "X" in handle else [])
        names += [n for n in layer_names if n not in count_names]
    else:
        names = [layers]
    sample_rows = np.unique(np.linspace(0, n_obs - 1, min(32, n_obs), dtype=int))
    direct, recoverable = [], []
    for name in names:
        # Row samples of CSC would read the whole matrix.
        if _is_csc(handle, name):
            raise ValueError(f"{name} is CSC; select it explicitly with layers=...")
        sample = _matrix(handle, name)[sample_rows, :]
        try:
            _count_chunk(sample, recover=False)
        except ValueError:
            try:
                _count_chunk(sample, recover=True)
            except ValueError:
                continue
            recoverable.append(name)
        else:
            # A sampled all-zero matrix cannot establish count semantics.
            if layers is not None or sparse.csr_matrix(sample).count_nonzero():
                direct.append(name)
    candidates = direct or recoverable
    if not candidates:
        raise ValueError(NO_COUNTS)
    recover = not direct
    if recover and not attempt_conversion:
        raise ValueError(RECOVERY_POSSIBLE + f" (candidate: {recoverable[0]})")
    return max(candidates, key=lambda n: _matrix(handle, n).shape[1]), recover


def _csc_to_row_buckets(group: h5py.Group, folder: Path, rows_per_bucket: int) -> tuple[list[Path], np.dtype]:
    """Regroup an on-disk CSC matrix by rows in one pass over its gene blocks.

    Slicing rows out of CSC reads every column, so a row-chunk loop would read the
    whole matrix once per chunk. Instead each nonzero (row, gene, value) is appended
    to the file of its row bucket; bucket ``b`` holds rows
    ``[b * rows_per_bucket, (b + 1) * rows_per_bucket)``, with rows stored relative
    to the bucket start. Temporary disk use is about 12 bytes per nonzero.
    """
    n_rows, n_genes = (int(n) for n in group.attrs["shape"])
    record = np.dtype([("row", "<i4"), ("gene", "<i4"), ("value", group["data"].dtype)])
    n_buckets = -(-n_rows // rows_per_bucket)
    folder.mkdir(parents=True)
    paths = [folder / f"{b}.bin" for b in range(n_buckets)]
    for path in paths:
        path.touch()
    indptr = group["indptr"][:]
    for g0 in range(0, n_genes, CSC_GENES_PER_BLOCK):
        g1 = min(g0 + CSC_GENES_PER_BLOCK, n_genes)
        rows = group["indices"][indptr[g0]:indptr[g1]]
        values = group["data"][indptr[g0]:indptr[g1]]
        genes = np.repeat(np.arange(g0, g1, dtype=np.int32), np.diff(indptr[g0:g1 + 1]))
        bucket = rows // rows_per_bucket
        order = np.argsort(bucket, kind="stable")
        bounds = np.searchsorted(bucket[order], np.arange(n_buckets + 1))
        for b in np.flatnonzero(np.diff(bounds)):
            selected = order[bounds[b]:bounds[b + 1]]
            out = np.empty(len(selected), dtype=record)
            out["row"] = rows[selected] - b * rows_per_bucket
            out["gene"] = genes[selected]
            out["value"] = values[selected]
            with open(paths[b], "ab") as handle:
                out.tofile(handle)
    return paths, record


def _read_row_bucket(path: Path, record: np.dtype, n_rows: int, n_genes: int) -> sparse.csr_matrix:
    entries = np.fromfile(path, dtype=record)
    return sparse.csr_matrix((entries["value"], (entries["row"], entries["gene"])), shape=(n_rows, n_genes))


def _write_metadata(output: Path, obs, var, row_sums, manifest: dict) -> None:
    """Write obs/var and size factors, then the manifest last.

    An interrupted output therefore has no corpus.yaml and cannot be mistaken
    for a complete corpus.
    """
    obs.to_parquet(output / "obs.parquet")
    var.to_parquet(output / "var.parquet")
    if row_sums is not None:
        # Empty cells retain a finite factor; their raw row sum remains zero.
        positive = row_sums[row_sums > 0]
        median = float(np.median(positive)) if len(positive) else 1.0
        factors = np.where(row_sums > 0, row_sums / median, 1.0)
        pd.DataFrame({"row_sum": row_sums, "size_factor": factors}).to_parquet(
            output / "statistics.parquet", index=False,
        )
    (output / "corpus.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))


def from_h5ad(
    source: str | Path,
    output: str | Path,
    *,
    layers: str | None = None,
    attempt_conversion: bool = False,
    no_count: bool = False,
    round_counts: bool = False,
    features: tuple[str, object] | None = None,
    extra: Mapping[str, tuple[str, Sequence[str]]] | None = None,
    csc_tmp_dir: str | Path | None = None,
    chunk_rows: int = 4096,
) -> Corpus:
    """Stream one h5ad into a standalone Lance corpus with original obs/var.

    ``layers`` selects ``X``, ``raw`` (raw.X), or one layer name; without it the
    count matrix is chosen by probing sampled rows. ``attempt_conversion`` allows
    log1p recovery, ``no_count`` keeps non-count values as they are,
    ``round_counts`` rounds fractional counts (half to even), ``features=(var_column,
    value)`` keeps only matching features, and ``extra={name: (h5_path, names)}``
    also stores per-cell matrices such as guide UMIs (read back with
    ``Corpus.read_extra``). A CSC matrix is regrouped by rows through temporary
    files in ``csc_tmp_dir``. Every chunk is validated. Outputs must not exist;
    the manifest is written last. See docs/composable_corpora.md.
    """
    from .loaders import load_corpus

    source, output = Path(source), Path(output)
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if output.exists():
        raise FileExistsError(output)
    if no_count and (layers is None or attempt_conversion):
        raise ValueError("no_count=True requires layers=... and cannot use attempt_conversion")
    if (round_counts or features is not None) and layers is None:
        raise ValueError("round_counts and features require layers=...")

    with h5py.File(source, "r") as handle:
        obs = ad.io.read_elem(handle["obs"])
        if len(obs) == 0:
            raise ValueError("Cannot convert a dataset with no cells")
        # These explicit routes are not probed; every chunk is validated anyway.
        if layers is not None and (no_count or round_counts or features is not None or _is_csc(handle, layers)):
            selected, recover = layers, False
        else:
            selected, recover = _choose_matrix(handle, len(obs), layers, attempt_conversion)
        matrix = _matrix(handle, selected)
        csc = _is_csc(handle, selected)
        if csc and csc_tmp_dir is None:
            raise ValueError("A CSC matrix needs csc_tmp_dir= for its temporary row buckets")
        var = ad.io.read_elem(handle["raw/var" if selected == "raw" else "var"])
        if matrix.shape != (len(obs), len(var)):
            raise ValueError(f"{selected}: matrix shape {matrix.shape} disagrees with obs/var")
        keep = None
        if features is not None:
            column, value = features
            keep = (var[column] == value).to_numpy()
            var = var[keep]
        kind = ("non_count" if no_count else "recovered_counts" if recover
                else "rounded_counts" if round_counts else "counts")
        extra_matrices = {}
        for name, (path, feature_names) in (extra or {}).items():
            extra_matrices[name] = _open(handle[path])
            # A short or misnamed extra matrix would silently misalign cells or features.
            if extra_matrices[name].shape != (len(obs), len(feature_names)):
                raise ValueError(f"extra {name}: shape {extra_matrices[name].shape} is not "
                                 f"{len(obs)} cells x {len(feature_names)} names")
        output.mkdir(parents=True)
        if extra_matrices:
            (output / "extra").mkdir()
        if csc:
            bucket_dir = Path(csc_tmp_dir) / "csc_row_buckets"
            buckets, record = _csc_to_row_buckets(handle[_matrix_path(selected)], bucket_dir, chunk_rows)
        states = {}
        row_sums = np.empty(len(obs), dtype=np.float64)
        n_values = n_fractional = 0
        for start in range(0, len(obs), chunk_rows):
            stop = min(start + chunk_rows, len(obs))
            last = stop == len(obs)
            if csc:
                chunk = _read_row_bucket(buckets[start // chunk_rows], record, stop - start, matrix.shape[1])
            else:
                chunk = sparse.csr_matrix(matrix[start:stop, :])
            if keep is not None:
                chunk = chunk[:, keep]
            if round_counts:
                n_values += chunk.nnz
                n_fractional += int(np.count_nonzero(chunk.data != np.floor(chunk.data)))
                chunk.data = np.rint(chunk.data)
            if no_count:
                if not np.isfinite(chunk.data).all():
                    raise ValueError("Non-count expression contains non-finite values")
                bundle = ChunkBundle(
                    np.arange(start, stop, dtype=np.int64),
                    np.asarray(chunk.sum(axis=1)).ravel(),
                    chunk.indptr.astype(np.int64), chunk.indices.astype(np.int32),
                    chunk.data, stop - start,
                )
            else:
                bundle = _count_chunk(chunk, recover=recover, start=start)
            row_sums[start:stop] = bundle.row_sums
            _append(bundle, "expression", output, states, last)
            for name, extra_matrix in extra_matrices.items():
                _append(_count_chunk(extra_matrix[start:stop], recover=False, start=start),
                        name, output / "extra", states, last)
        if csc:
            shutil.rmtree(bucket_dir)
        for name, (path, feature_names) in (extra or {}).items():
            pd.DataFrame(index=pd.Index(list(feature_names), name="feature")).to_parquet(
                output / "extra" / f"{name}_features.parquet")
        manifest = {
            **MANIFEST, "source": str(source.resolve()),
            "matrix_source": selected, "expression_kind": kind,
            "shape": [len(obs), len(var)],
            "dtype": str(bundle.counts.dtype),
            "recovery": "expm1/min_nonzero_expm1_per_cell" if recover else None,
            "size_factor_method": None if no_count else SIZE_FACTOR_METHOD,
        }
        if features is not None:
            manifest["features"] = {"var_column": features[0], "value": features[1]}
        if round_counts:
            manifest["rounding"] = "half_to_even"
            manifest["fractional_value_share"] = n_fractional / max(n_values, 1)
        if extra:
            manifest["extra"] = {name: {"source": path, "n_features": len(feature_names)}
                                 for name, (path, feature_names) in extra.items()}
        _write_metadata(output, obs, var, None if no_count else row_sums, manifest)
    return load_corpus(output)


def _feature_columns(ids, types, feature_type, wanted, path) -> np.ndarray:
    """File columns holding the ``wanted`` feature ids, in that order.

    The file's features of this type and ``wanted`` must be the same set, so no
    measured feature is dropped and no requested one is missing.
    """
    of_type = np.flatnonzero(types == feature_type)
    positions = pd.Index(ids[of_type]).get_indexer(wanted)
    if len(of_type) != len(wanted) or (positions < 0).any():
        raise ValueError(f"{path}: {feature_type} feature ids differ from the requested ids")
    return of_type[positions]


def _read_10x_block(path: Path, lo: int, hi: int):
    """Rows [lo, hi) of a CellRanger h5 as cells x features CSR, with feature ids and types."""
    with h5py.File(path, "r") as handle:
        # CellRanger stores features x barcodes CSC: each cell is one contiguous slice.
        m = handle["matrix"]
        indptr = m["indptr"][lo:hi + 1]
        block = sparse.csr_matrix(
            (m["data"][indptr[0]:indptr[-1]], m["indices"][indptr[0]:indptr[-1]], indptr - indptr[0]),
            shape=(hi - lo, int(m["shape"][0])),
        )
        return block, m["features/id"][:].astype(str), m["features/feature_type"][:].astype(str)


def _read_h5ad_block(path: Path, lo: int, hi: int):
    """Rows [lo, hi) of a count h5ad (gene ids in var["gene_ids"]); all features are genes."""
    with h5py.File(path, "r") as handle:
        block = sparse.csr_matrix(ad.io.sparse_dataset(handle["X"])[lo:hi])
        ids = ad.io.read_elem(handle["var"])["gene_ids"].astype(str).to_numpy()
    return block, ids, np.full(len(ids), "Gene Expression")


def from_10x_h5_files(
    files: Mapping[str, str | Path],
    cells: pd.DataFrame,
    output: str | Path,
    *,
    genes: pd.DataFrame,
    guides: pd.DataFrame | None = None,
    chunk_rows: int = 4096,
) -> Corpus:
    """Stream cells from many CellRanger ``.h5`` files (and count h5ads) into one corpus.

    ``files`` maps a file id to its path. ``cells`` has one row per output cell,
    in output order, with columns ``file_id`` and ``local_row`` (the cell's row
    in that file); it is stored unchanged as obs. ``genes`` is the var table,
    indexed by gene id: each file's "Gene Expression" features are matched to it
    by id, so files may list genes in different orders. ``guides`` (optional,
    indexed by guide id) does the same for "CRISPR Guide Capture" features, which
    are stored as the extra matrix ``guide_umi``. h5ad files (counts in X, gene
    ids in ``var["gene_ids"]``) have no guide features and get empty guide rows.
    """
    from .loaders import load_corpus

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    if guides is not None:
        (output / "extra").mkdir()
    file_ids = cells["file_id"].astype(str).to_numpy()
    local_rows = cells["local_row"].to_numpy()
    row_sums = np.empty(len(cells), dtype=np.float64)
    states = {}
    for start in range(0, len(cells), chunk_rows):
        stop = min(start + chunk_rows, len(cells))
        last = stop == len(cells)
        # Consecutive cells from the same file are read together.
        ids = file_ids[start:stop]
        run_starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        run_stops = np.r_[run_starts[1:], len(ids)]
        expression_parts, guide_parts = [], []
        for a, b in zip(run_starts, run_stops):
            path = Path(files[ids[a]])
            rows = local_rows[start + a:start + b]
            lo, hi = int(rows.min()), int(rows.max()) + 1
            is_h5 = path.suffix == ".h5"
            block, feature_ids, types = (_read_10x_block if is_h5 else _read_h5ad_block)(path, lo, hi)
            block = block[rows - lo]
            expression_parts.append(block[:, _feature_columns(feature_ids, types, "Gene Expression", genes.index, path)])
            if guides is not None and is_h5:
                guide_parts.append(block[:, _feature_columns(feature_ids, types, "CRISPR Guide Capture", guides.index, path)])
            elif guides is not None:
                guide_parts.append(sparse.csr_matrix((len(rows), len(guides)), dtype=block.dtype))
        bundle = _count_chunk(sparse.vstack(expression_parts), recover=False, start=start)
        row_sums[start:stop] = bundle.row_sums
        _append(bundle, "expression", output, states, last)
        if guides is not None:
            _append(_count_chunk(sparse.vstack(guide_parts), recover=False, start=start),
                    "guide_umi", output / "extra", states, last)
    manifest = {
        **MANIFEST, "source": sorted(str(Path(files[f]).resolve()) for f in set(file_ids)),
        "matrix_source": "10x Gene Expression features", "expression_kind": "counts",
        "shape": [len(cells), len(genes)], "dtype": str(bundle.counts.dtype), "recovery": None,
        "size_factor_method": SIZE_FACTOR_METHOD,
    }
    if guides is not None:
        guides.to_parquet(output / "extra" / "guide_umi_features.parquet")
        manifest["extra"] = {"guide_umi": {"source": "10x CRISPR Guide Capture features",
                                           "n_features": len(guides)}}
    _write_metadata(output, cells, genes, row_sums, manifest)
    return load_corpus(output)
