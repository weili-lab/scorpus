# Quickstart

Convert an `.h5ad` file into a corpus, then load it. No inspection file, no
schema YAML, no canonical column names.

## Convert one file

```python
from scorpus import from_h5ad, load_corpus

corpus = from_h5ad("dataset.h5ad", "dataset.corpus")

print(corpus.obs)   # your original obs — same columns, same index, untouched
print(corpus.var)   # the selected matrix's feature axis
```

From the command line:

```bash
python -m scorpus.cli convert --source dataset.h5ad --output dataset.corpus
```

Reopen it later with `load_corpus("dataset.corpus")`.

## How the count matrix is chosen

`scorpus` samples `X`, `raw.X`, and every layer looking for finite, nonnegative,
integer-like counts. It prefers a **direct** count source over a recoverable
one; among direct sources the widest gene axis wins, with counts-named layers,
then `raw.X`, then `X` breaking ties.

Crucially, a sampled pass is not treated as proof: **every chunk is revalidated
during the write.**

When no direct counts exist, conversion stops and tells you what to pass:

=== "Recover counts from log-normalized data"

    ```python
    from_h5ad("log_norm.h5ad", "out.corpus", attempt_conversion=True)
    ```

    Recovery uses `expm1(x) / min_nonzero_expm1_per_cell`, assuming the smallest
    nonzero value is one molecule. Integer checks do **not** prove the original
    absolute counts were recovered — the manifest records this as
    `recovered_counts`, not `counts`.

=== "Pick a matrix explicitly"

    ```python
    from_h5ad("a.h5ad", "out.corpus", layers="counts")
    ```

    `layers` takes one selection: `raw` for `raw.X`, `X` for `X`, or a layer name.

=== "Keep non-count values as they are"

    ```python
    from_h5ad("norm.h5ad", "out.corpus", layers="X", no_count=True)
    ```

    Preserves the numeric dtype and values, and skips size-factor computation.
    Recorded as `non_count`.

The manifest always distinguishes the three cases, so a downstream reader never
has to guess what it is holding.

!!! warning "What conversion does not copy"
    Only the selected matrix and its `obs`/`var` are converted. Other layers,
    `obsm`, `obsp`, and `uns` are **not** copied. Existing outputs are never
    overwritten, and the manifest is written last, so an interrupted run cannot
    be mistaken for a complete corpus.

## Compose several datasets

Convert datasets independently, then join them in memory. Harmonization is
ordinary pandas — there is no schema DSL.

```python
from scorpus import load_corpus, concat

a, b = load_corpus("a.corpus"), load_corpus("b.corpus")

a_obs, b_obs = a.obs.copy(), b.obs.copy()
a_obs["pert"] = a_obs["target"].replace({"NTC": "ctrl"})
b_obs["pert"] = b_obs["genotype"].replace({"WT": "ctrl"})

combined = concat(
    {"a": a, "b": b},
    obs={"a": a_obs, "b": b_obs},
    obs_columns={
        "pert": {"a": "pert", "b": "pert"},
        "cell_type": {"a": "celltype", "b": "annotation"},
    },
    feature_key={"a": "ensembl_id", "b": "gene_id"},
    var_columns={"symbol": {"a": "symbol", "b": "gene_symbol"}},
)
```

The view holds **references** to the member readers. No expression data is
copied, and `combined.members["a"].obs` still gives you the original member
metadata.

### Feature identity is explicit

Without `feature_key`, features stay member-local: `scorpus` will not assume two
similarly named genes are the same gene. Use `"index"` to align on `var_names`.
Mapped IDs must be nonempty, unique within each dataset, and already in the same
identifier namespace.

`combined.feature_presence` records which member actually measured each feature:

```python
adata = combined.to_anndata_lazy(var_join="outer")
adata.varm["feature_presence"]      # per-member availability
adata.uns["feature_presence_members"]
```

!!! danger "Outer-join zeros are not measured zeros"
    An outer export fills unavailable genes with zeros. Those positions are
    *unmeasured*, not observed-as-zero. Check `feature_presence` before treating
    them as data.

## Hand back to AnnData

```python
adata = corpus.to_anndata_lazy(chunk_rows=4096)          # Dask X, stays on disk
small = corpus.to_anndata(global_row_indices=[2, 0, 5])  # eager, selected rows
corpus.write_h5ad("export.h5ad")                         # reopen with backed="r"
```

After export, Scanpy/RAPIDS owns normalization, PCA, neighbors, UMAP, and
clustering. Join results back with `corpus.add_obs_meta(...)`. See
[Scanpy & RAPIDS](scanpy_rapids.md).

## Train

```python
from scorpus.loaders import build_loader

for batch in build_loader(corpus, seq_len=1024, batch_size=128):
    ...
```

## Next

- [Conversion & composition reference](composable_corpora.md) — full API detail
- [AnnData handoff](anndata_scanpy_handoff.md)
- [Curated corpus releases](bash_demo.md) — the reviewed metadata pipeline
