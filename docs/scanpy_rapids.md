# Scanpy & RAPIDS Support

This page shows the downstream analysis handoff after the demo corpus has been
canonicalized (see the [Bash walkthrough](bash_demo.md)).

What is validated for the Marson + Xorion demo corpus:

- single-dataset `to_anndata_lazy(...)`
- cross-dataset `to_anndata_lazy(..., var_join="inner")`
- Dask-backed `adata.X`
- CPU Scanpy preprocessing on the combined demo corpus
- optional RAPIDS GPU preprocessing when a CUDA environment is available

For the full API surface, see
[AnnData Handoff API](anndata_scanpy_handoff.md).

## Load the demo corpus

```python
from scorpus.loaders import load_corpus

corpus = load_corpus(
    "./artifacts/demo_corpus",
    extra_metadata_columns=["perturb_label", "condition", "cell_context", "batch_id"],
)
```

## Single-dataset lazy AnnData

Use this when you want one dataset exactly as-is:

```python
adata_marson = corpus.to_anndata_lazy(
    dataset_id="marson_d2_rest",
    obs_columns=["perturb_label", "condition", "cell_context", "batch_id"],
)

print(adata_marson.shape)
print(type(adata_marson.X))
```

`adata_marson.X` stays Dask-backed, so the count matrix is still read in chunks
of `chunk_rows` cells (default 65,536; lower it if memory runs short).

## Cross-dataset lazy AnnData with native gene axes

Marson and Xorion keep their own native feature axes during materialization.
That means cross-dataset export needs an explicit join rule.

```python
adata = corpus.to_anndata_lazy(
    dataset_id=["marson_d2_rest", "xorion_hct116_dual_guide"],
    obs_columns=["perturb_label", "condition", "cell_context", "batch_id"],
    var_join="inner",
)

print(f"adata shape: {adata.shape}")
print(f"adata.X type: {type(adata.X)}")
print(f"obs columns: {list(adata.obs.columns)}")
print(f"intersection genes: {adata.n_vars}")
```

### `var_join="exact"` vs `var_join="inner"`

- `var_join="exact"` only works when the selected datasets already share the
  same ordered `canonical_gene_id` axis.
- `var_join="inner"` keeps only the shared genes and is the right choice for
  this Marson + Xorion demo.

This is not gene harmonization. It is only an intersection of already-canonical
feature identifiers.

## CPU Scanpy smoke

For the validated demo path, basic Scanpy preprocessing runs directly on the
lazy combined AnnData:

```python
import scanpy as sc

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

print(f"HVG selected: {int(adata.var['highly_variable'].sum())}")

# Force one small read so you can see that the lazy matrix really computes.
subset = adata[:100, :100].to_memory()
print(f"Subset shape: {subset.shape}")
print(f"Subset X nnz: {subset.X.nnz}")
```

This gives a simple collaborator-friendly smoke without copying the whole corpus
into a new `.h5ad` file first.

## Optional RAPIDS GPU smoke

RAPIDS is environment-dependent and is **not** part of the basic install path.
Use it only when you already have a CUDA-ready environment.

The simplest stable demo path is to export a normal AnnData, then move it to the
GPU with `rapids_singlecell`:

```python
adata_gpu = corpus.to_anndata(
    dataset_id=["marson_d2_rest", "xorion_hct116_dual_guide"],
    obs_columns=["perturb_label", "condition", "cell_context", "batch_id"],
    var_join="inner",
)

import rapids_singlecell as rsc

rsc.get.anndata_to_GPU(adata_gpu)
rsc.pp.normalize_total(adata_gpu, target_sum=1e4)
rsc.pp.log1p(adata_gpu)
rsc.pp.highly_variable_genes(adata_gpu, n_top_genes=2000)

print(type(adata_gpu.X))
print(int(adata_gpu.var["highly_variable"].sum()))
```

To keep the matrix on disk instead, export lazily with `device="cuda"`: every
Dask chunk is then a CuPy CSR matrix, so no `anndata_to_GPU` call is needed. This
works for a standalone corpus from `from_h5ad`, a `concat` view, and the older
canonical corpora:

```python
corpus = load_corpus("my.corpus")
adata_gpu = corpus.to_anndata_lazy(device="cuda")

rsc.pp.normalize_total(adata_gpu, target_sum=1e4)
rsc.pp.log1p(adata_gpu)
```

Integer counts arrive on the GPU as float32. RAPIDS Dask support is
function-specific. Tested with rapids-singlecell 0.18.1 (the prebuilt wheel is
named `rapids-singlecell-cu12`; plain `rapids-singlecell` is source-only) and
RAPIDS 26.10: `calculate_qc_metrics`, `normalize_total`, `log1p`,
`highly_variable_genes` (`seurat`, `cell_ranger`, `seurat_v3` with `batch_key`),
`scale`, `pca`, `regress_out`, `rank_genes_groups(method="t-test")` and
`get.aggregate` run on the lazy GPU matrix. (With 0.14.1, `seurat_v3` needed an
in-memory matrix.) `pca` returns a Dask `X_pca`; call
`adata.obsm["X_pca"] = adata.obsm["X_pca"].compute()` before `neighbors`,
`leiden`, `umap` or `harmony_integrate`. `wilcoxon` and `pearson_residuals` have no
Dask support.

On the CPU, scanpy 1.12's `calculate_qc_metrics`, `rank_genes_groups` and
`highly_variable_genes(flavor="seurat_v3")` would pull the whole matrix into memory
on a lazy export: two scanpy Dask helpers (`axis_nnz` along genes and
`check_nonnegative_integers`) join all row chunks into one block. When scanpy is
installed, `import scorpus` replaces these two helpers with chunk-by-chunk
versions for the running process (the installed scanpy is not changed), so these
functions stay lazy. On a 98,927-cell export this cut peak memory
from about 24 GB to 3-6 GB with identical results.

### Chunk size and memory

Each Dask chunk is built in memory before it is used, and Dask works on several
chunks at once (one per thread). Budget about `chunk_rows` x threads x 3 times the
size of one chunk's CSR matrix (the read buffers plus one copy of the matrix). If
memory runs short, lower `chunk_rows`.

If your environment does not provide CUDA or `rapids_singlecell`, treat this as
an optional extension and record the blocker explicitly.

## What to expect from the validated demo corpus

- combined demo corpus: `5,440` cells
- requested obs columns present: `perturb_label`, `condition`, `cell_context`, `batch_id`
- cross-dataset `var_join="inner"` produces a non-empty shared gene axis
- `adata.X` stays Dask-backed for the lazy export path

## See also

- [AnnData Handoff API](anndata_scanpy_handoff.md) — complete API reference
- {doc}`Walkthrough notebook <tutorials/demo_walkthrough>` — the curated route in Python, with cell outputs
- [Bash Demo](bash_demo.md) — end-to-end corpus preparation
- [Canonicalization](demo_canonicalization.md) — how canonical labels enable downstream analysis
