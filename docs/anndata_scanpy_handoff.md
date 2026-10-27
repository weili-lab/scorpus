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
corpus.add_obs_meta(frame)
corpus.save_ex_obs()
```

- `to_anndata(...)` builds a normal in-memory AnnData with SciPy CSR `X` for whole datasets or selected global rows.
- `to_anndata_lazy(...)` builds an AnnData whose `X` is a Dask array of sparse CSR chunks read from the corpus backend.
- `add_obs_meta(...)` appends per-cell results to a loaded standalone corpus; `save_ex_obs(...)` / `save_ex_var(...)` save its obs / var next to the corpus (see "Saving results with the corpus").

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

## Saving results with the corpus

A standalone corpus keeps per-cell and per-gene results in files next to it:
`ex_obs.parquet` / `ex_var.parquet` (the default), or `ex_obs_<name>.parquet` /
`ex_var_<name>.parquet` (named). Each file holds the full obs or var, the original
columns plus the added ones, in the corpus's cell or gene order. Loading one uses it
instead of `obs.parquet` / `var.parquet`, which are never changed.

```python
corpus = load_corpus("my.corpus")
adata = corpus.to_anndata_lazy()
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000, batch_key="lane")
# ... clustering etc.

corpus.add_obs_meta(adata.obs[["leiden", "doublet_score"]])  # export keeps the corpus cell ids
corpus.save_ex_obs()                     # ex_obs.parquet (created or overwritten)
for column in ["highly_variable", "highly_variable_rank"]:
    corpus.var[column] = adata.var[column].to_numpy()
corpus.save_ex_var("hvg2000")            # ex_var_hvg2000.parquet

corpus = load_corpus("my.corpus", ex_obs=True, ex_var="hvg2000")
```

- `add_obs_meta(meta, overwrite=False)` takes a DataFrame or a parquet path. Its
  index must equal the corpus obs index (the cell ids) element by element: same
  rows, same order. Repeated barcodes are fine, because order is compared, not
  uniqueness. A column that already exists is an error unless `overwrite=True`;
  the routing fields (`cell_id`, `dataset_id`, `dataset_index`, `global_row_index`,
  `local_row_index`) are always refused.
  The new columns join `corpus.obs`, `take_metadata` and AnnData exports, but only
  in the loaded corpus; `save_ex_obs` saves them. A standalone corpus's AnnData
  export keeps its cell ids as the obs index, so its obs can be passed back as is.
- `save_ex_obs(name=None)` / `save_ex_var(name=None)` write `corpus.obs` /
  `corpus.var`. They refuse when its rows (cells or genes) differ from the corpus's
  `obs.parquet` / `var.parquet` in number or order.
- `load_corpus(root, ex_obs=..., ex_var=...)`: `True` loads the default file, a
  string loads that named file; obs and var are chosen separately.
- Both work on one standalone corpus. For a `concat`, load each member with its
  ex files and concat those: `concat({"a": load_corpus(a, ex_obs=True), ...})`.
- The shared corpora under `/beacon-projects/perturbai/data/` are read-only, so ex
  files need a copy of the corpus.
- `reorder_corpus` carries every ex file (see `rewrite_and_stage.md`).

### HVGs from `ex_var`

When the loaded var has a `highly_variable` column, the loader fills the gene
index's HVG ranks from it, so `feature_registry.hvg_mask` /
`hvg_rank_matrix` (used, for example, by pertTF's `"hvg"` gene sampling) and the
AnnData `var["hvg_rank"]` follow it:

- `highly_variable_rank` is required as well. It is scanpy's `seurat_v3` /
  `seurat_v3_paper` column (0-based, NaN outside the top genes, a median across
  batches with `batch_key`). The `seurat` and `cell_ranger` flavors write no rank;
  add one before saving, for example
  `var["highly_variable_rank"] = var["dispersions_norm"].rank(ascending=False, method="first") - 1`.
- Only the `highly_variable` genes are ranked, 1..n in `highly_variable_rank` order
  (ties in gene order); all other genes get 0, so `hvg_mask` is exactly
  `highly_variable` and a top-k above n gives all n HVGs (for more, compute HVGs
  with a larger `n_top_genes`).

After adding metadata, the new columns can define groups for sampling, for example
batches that each hold cells of one cluster and perturbation:

```python
from scorpus.pairing import cells, make_groups

groups = make_groups(corpus.metadata_index.df, by=["leiden", "perturb_label"])
batches = cells(groups, epoch=0, batch_size=128, same_group=True)
```
