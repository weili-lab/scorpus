# Scanpy & RAPIDS Support

This page shows the downstream analysis handoff after the demo corpus has been
canonicalized.

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
    chunk_rows=1024,
)

print(adata_marson.shape)
print(type(adata_marson.X))
```

`adata_marson.X` stays Dask-backed, so the count matrix is still read in chunks.

## Cross-dataset lazy AnnData with native gene axes

Marson and Xorion keep their own native feature axes during materialization.
That means cross-dataset export needs an explicit join rule.

```python
adata = corpus.to_anndata_lazy(
    dataset_id=["marson_d2_rest", "xorion_hct116_dual_guide"],
    obs_columns=["perturb_label", "condition", "cell_context", "batch_id"],
    chunk_rows=1024,
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

If your environment does not provide CUDA or `rapids_singlecell`, treat this as
an optional extension and record the blocker explicitly.

## One-command validation script

The repo includes a small validation script that runs the same demo checks and
writes a JSON summary:

```bash
PYTHONPATH=src python scripts/scanpy_rapids_demo_validate.py \
  --corpus-root ./artifacts/demo_corpus \
  --summary-json ./artifacts/validation/scanpy_rapids_summary.json
```

Add `--require-rapids` when you are on a real GPU node and want the script to
fail fast if the RAPIDS step cannot run.

## What to expect from the validated demo corpus

- combined demo corpus: `5,440` cells
- requested obs columns present: `perturb_label`, `condition`, `cell_context`, `batch_id`
- cross-dataset `var_join="inner"` produces a non-empty shared gene axis
- `adata.X` stays Dask-backed for the lazy export path

## See also

- [AnnData Handoff API](anndata_scanpy_handoff.md) — complete API reference
- [Jupyter Demo](jupyter_demo.md) — the full notebook-style walkthrough
- [Rendered Notebook](demo_walkthrough.ipynb) — the executed walkthrough with cell outputs
- [Bash Demo](bash_demo.md) — end-to-end corpus preparation
- [Canonicalization](demo_canonicalization.md) — how canonical labels enable downstream analysis
