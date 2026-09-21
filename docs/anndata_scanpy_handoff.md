# AnnData, Scanpy, and RAPIDS handoff

This document describes the boundary between `scorpus` and standard
single-cell analysis tools.

`scorpus` owns corpus materialization, canonical metadata, sparse count
storage, feature alignment for loaders, durable per-dataset HVG rankings, and
training/runtime access patterns. After selected corpus rows or datasets are exported
as `AnnData`, Scanpy or RAPIDS should own the usual analysis preprocessing.

## Public handoff API

The handoff API has three parts:

```python
adata = corpus.to_anndata(dataset_id="replogle_k562")
adata = corpus.to_anndata(global_row_indices=[10, 25, 10])
adata = corpus.to_anndata_lazy(dataset_id="replogle_k562", chunk_rows=4096)
corpus.add_obs_meta(frame, on=["dataset_id", "cell_id"])
```

- `to_anndata(...)` builds a normal in-memory AnnData with SciPy CSR `X` for whole datasets or selected global rows.
- `to_anndata_lazy(...)` builds an AnnData whose `X` is a Dask array of sparse CSR chunks read from the corpus backend.
- `add_obs_meta(...)` joins selected cell-level results back into the loaded corpus metadata at runtime.

Selected-index export requires all global row indices to belong to one dataset.
It preserves requested order and duplicate rows, and infers the dataset when
`dataset_id` is omitted. Cross-dataset selected-index export fails explicitly.

## Feature-axis rule

Single-dataset handoff is always allowed.

Multi-dataset handoff is allowed only when all selected datasets have exactly the
same ordered feature axis:

- same number of features
- same `canonical_gene_id` values
- same feature order

This check happens before expression is read. If the feature axes differ, the
export fails instead of trying to silently remap sparse indices.

Cross-dataset feature reconciliation is a separate future feature. It would need
a real transformation layer that remaps local feature indices while reading each
chunk.

## Eager AnnData handoff

Use `to_anndata(...)` when the selected rows or whole dataset(s) fit in RAM:

```python
from scorpus.loaders import load_corpus

corpus = load_corpus(
    "/path/to/corpus",
    extra_metadata_columns=["donor_id", "batch_id"],
)

adata = corpus.to_anndata(
    dataset_id="replogle_k562",
    obs_columns=["perturb_label", "donor_id", "batch_id"],
)
```

For a selected row sequence from one dataset:

```python
adata = corpus.to_anndata(
    global_row_indices=[105, 250, 105],
    obs_columns=["perturb_label", "batch_id"],
)
```

The output preserves `[105, 250, 105]`. Observation names are made unique when
indices repeat, while `adata.obs["global_row_index"]` retains the exact values.

For compatible datasets:

```python
adata = corpus.to_anndata(
    dataset_id=["dataset_a", "dataset_b"],
    obs_columns=["perturb_label", "donor_id", "batch_id"],
)
```

The export is counts-only. It builds CSR `adata.X` and does not add normalized or
log-transformed layers.

`adata.obs` includes stable provenance fields such as `dataset_id`,
`dataset_index`, `global_row_index`, `local_row_index`, and `cell_id` when those
columns are present in the loaded metadata. Requested `obs_columns` are added
alongside them.

For single-dataset and selected-index exports, `adata.var` contains the canonical
var metadata plus `hvg_rank` and `highly_variable` from the corpus HVG ranking.
For multi-dataset exports, `adata.var` comes from the first selected dataset after
the shared feature-axis check.

## Lazy AnnData handoff

Use `to_anndata_lazy(...)` when the selected whole dataset(s) should stay on disk
and be read in chunks by Dask:

```python
adata = corpus.to_anndata_lazy(
    dataset_id="replogle_k562",
    obs_columns=["perturb_label", "donor_id", "batch_id"],
    chunk_rows=4096,
)
```

Only `adata.X` is lazy. `adata.obs` and `adata.var` are loaded in memory because
they are small compared with the count matrix.

The Dask chunks are CSR matrices. Each chunk reads a contiguous row block from
the active corpus backend:

- Zarr chunks read `row_offsets`, `indices`, and `counts` through the Zarr expression reader.
- Lance chunks read row blocks through the Lance expression reader and unpack Arrow list columns.

This keeps the handoff lightweight: no new AnnData-Zarr copy is required just to
let AnnData hold a Dask-backed `X`.

For GPU workflows, `to_anndata_lazy(..., device="cuda")` makes each Dask task
build a CuPy CSR chunk directly:

```python
adata = corpus.to_anndata_lazy(
    dataset_id="replogle_k562",
    chunk_rows=4096,
    device="cuda",
)
```

This still reads Lance/Zarr buffers through CPU memory before the CPU-to-GPU
transfer, but it avoids constructing an intermediate SciPy CSR matrix inside the
Dask task. It requires a working CUDA/CuPy runtime.

## Scanpy example

```python
import scanpy as sc

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)
```

Use this route when the selected AnnData representation is supported by the
Scanpy function you are calling. Scanpy Dask support is function-specific.

## RAPIDS example

```python
import rapids_singlecell as rsc

rsc.get.anndata_to_GPU(adata)
rsc.pp.normalize_total(adata, target_sum=1e4)
rsc.pp.log1p(adata)
rsc.pp.highly_variable_genes(adata, n_top_genes=2000)
rsc.tl.pca(adata)
rsc.pp.neighbors(adata)
rsc.tl.umap(adata)
rsc.tl.leiden(adata)
```

RAPIDS can work with GPU-backed chunks for supported functions. RAPIDS Dask
support is still function-specific; for example, `rank_genes_groups` supports
Dask for `t-test`, `t-test_overestim_var`, and `wilcoxon_binned`, but not
regular `wilcoxon` or `logreg`.

## Adding results back to the corpus

`scorpus` does not decide how Scanpy/RAPIDS outputs are saved. The user
owns the AnnData workflow and can add selected cell-level results back to the
loaded corpus with `add_obs_meta(...)`:

```python
corpus.add_obs_meta(
    adata.obs[["dataset_id", "cell_id", "leiden", "doublet_score"]],
    on=["dataset_id", "cell_id"],
)
```

`add_obs_meta(...)` is strict:

- join keys are required
- incoming rows must cover the full loaded corpus exactly once
- duplicate join keys are rejected
- missing corpus rows are rejected
- extra unmatched rows are rejected
- new metadata column names must not already exist in the corpus
- the update is runtime-only and does not persist to disk

Preferred join keys are:

- `on=["dataset_id", "cell_id"]`
- `on=["dataset_id", "local_row_index"]`
- `on=["global_row_index"]`

For a multi-dataset loaded corpus, analyze all feature-compatible datasets
together or combine per-dataset result frames before calling `add_obs_meta(...)`.
Partial subset metadata is rejected because it would leave the corpus in a
half-annotated runtime state.

After adding metadata, downstream loaders can use the new columns for sampling or
pass-through metadata:

```python
from scorpus.loaders import build_loader

loader = build_loader(
    corpus,
    sampler="context",
    batch_size=128,
    seq_len=1024,
    context_columns=["leiden", "perturb_label"],
    metadata_columns=["leiden", "doublet_score"],
)
```

## Corpus-native pp helpers

Use Scanpy or RAPIDS for normalization, log transforms, PCA, neighbors, UMAP,
clustering, plotting, batch correction, and exploratory marker ranking.

Use corpus-native `scorpus.pp` helpers for durable HVG rankings,
streamed summary statistics, quick QA/debug checks, and bounded-memory fallback
workflows when AnnData handoff is not practical.

The most important durable artifact is the per-dataset `hvg.parquet` ranking
table created during materialization or by `recalc-hvg`.
