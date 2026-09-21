"""Convert a selected h5ad matrix without imposing a biological metadata schema."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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


def _matrix(handle: h5py.File, name: str):
    path = "raw/X" if name == "raw" else "X" if name == "X" else f"layers/{name}"
    node = handle[path]
    return ad.io.sparse_dataset(node) if isinstance(node, h5py.Group) else node


def _count_chunk(matrix, *, recover: bool, start: int = 0) -> ChunkBundle:
    chunk = sparse.csr_matrix(matrix)
    chunk.sum_duplicates()
    chunk.eliminate_zeros()
    return _translate_chunk(
        dataset_id="conversion", global_row_start=0, matrix_chunk=chunk,
        chunk_start=start, needs_recovery=recover,
    )


def from_h5ad(
    source: str | Path,
    output: str | Path,
    *,
    layers: str | None = None,
    attempt_conversion: bool = False,
    no_count: bool = False,
    chunk_rows: int = 4096,
) -> Corpus:
    """Stream one h5ad into a standalone Lance corpus with original obs/var.

    ``layers`` selects ``X``, ``raw`` (raw.X), or one layer name. Without a
    selection, direct count candidates are preferred over recoverable matrices;
    the widest gene axis wins, with counts-named layers, raw, then X breaking ties.
    Recovery is opt-in and validated again on every chunk. ``no_count=True``
    requires an explicit selection and preserves its numeric dtype and values.

    Only the selected expression matrix and its obs/var are converted. Outputs
    must not exist. The manifest is written last; interrupted outputs cannot be
    mistaken for complete corpora. Large files must be converted on Slurm.
    """
    from .loaders import load_corpus

    source, output = Path(source), Path(output)
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if output.exists():
        raise FileExistsError(output)
    if no_count and (layers is None or attempt_conversion):
        raise ValueError("no_count=True requires layers=... and cannot use attempt_conversion")

    with h5py.File(source, "r") as handle:
        obs = ad.io.read_elem(handle["obs"])
        if len(obs) == 0:
            raise ValueError("Cannot convert a dataset with no cells")
        if layers is None:
            layer_names = list(handle.get("layers", {}))
            count_names = [n for n in layer_names if "count" in n.lower()]
            names = count_names + (["raw"] if "raw/X" in handle else [])
            names += (["X"] if "X" in handle else [])
            names += [n for n in layer_names if n not in count_names]
        else:
            names = [layers]
        sample_rows = np.unique(np.linspace(0, len(obs) - 1, min(32, len(obs)), dtype=int))
        direct, recoverable = [], []
        for name in names:
            matrix = _matrix(handle, name)
            if matrix.shape[0] != len(obs):
                raise ValueError(f"{name}: matrix rows disagree with obs")
            if no_count:
                direct.append(name)
                break
            sample = matrix[sample_rows, :]
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
        selected = max(candidates, key=lambda n: _matrix(handle, n).shape[1])
        matrix = _matrix(handle, selected)
        var = ad.io.read_elem(handle["raw/var" if selected == "raw" else "var"])
        if matrix.shape[1] != len(var) or len(var) == 0:
            raise ValueError("Selected matrix and feature metadata have incompatible shapes")
        kind = "non_count" if no_count else "recovered_counts" if recover else "counts"
        if np.dtype(matrix.dtype).kind not in "iuf":
            raise ValueError("Expression must use a real numeric dtype")
        output.mkdir(parents=True)
        writer_state = None
        row_sums = np.empty(len(obs), dtype=np.float64)
        for start in range(0, len(obs), chunk_rows):
            stop = min(start + chunk_rows, len(obs))
            chunk = sparse.csr_matrix(matrix[start:stop, :])
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
            _, writer_state = write_lance_federated(
                bundle, "expression", output, _writer_state=writer_state,
                _is_last_chunk=stop == len(obs),
            )
        obs.to_parquet(output / "obs.parquet")
        var.to_parquet(output / "var.parquet")
        if not no_count:
            # Empty cells retain a finite factor; their raw row sum remains zero.
            positive = row_sums[row_sums > 0]
            median = float(np.median(positive)) if len(positive) else 1.0
            factors = np.where(row_sums > 0, row_sums / median, 1.0)
            pd.DataFrame({"row_sum": row_sums, "size_factor": factors}).to_parquet(
                output / "statistics.parquet", index=False,
            )
        manifest = {
            "format": "perturb-data-lab", "version": 1,
            "backend": "lance", "source": str(source.resolve()),
            "matrix_source": selected, "expression_kind": kind,
            "shape": [len(obs), len(var)],
            "dtype": str(bundle.counts.dtype),
            "recovery": "expm1/min_nonzero_expm1_per_cell" if recover else None,
            "size_factor_method": None if no_count else "library_sum/median_positive_library_sum; empty_cells=1",
        }
        (output / "corpus.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return load_corpus(output)
