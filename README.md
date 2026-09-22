<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/scorpus_logo.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/scorpus_logo_light.png">
  <img src="docs/assets/scorpus_logo_light.png" alt="scorpus" width="440">
</picture>

**Sparse single-cell corpora — cheap to random-access, easy to compose,
ready to hand back to AnnData.**

[Documentation](https://weili-lab.github.io/scorpus/) ·
[Quickstart](https://weili-lab.github.io/scorpus/quickstart/) ·
[Conversion &amp; composition](docs/composable_corpora.md)

</div>

---

`scorpus` turns single-cell `.h5ad` files into sparse on-disk corpora that are
cheap to random-access, easy to compose, and ready to hand back to AnnData.

It is built for three jobs:

- **Large model training** — sparse batch loading straight off disk.
- **Perturbation model training** — paired sampling with fast random access.
- **Routine single-cell analysis** — hand the corpus back to Scanpy/RAPIDS.

## Install

```bash
pip install -e .
```

## Convert

One call. No inspection file, no schema, no canonical column names.

```python
from scorpus import from_h5ad, load_corpus

corpus = from_h5ad("dataset.h5ad", "dataset.corpus")
print(corpus.obs)   # your original obs, untouched
print(corpus.var)   # the selected matrix's feature axis
```

```bash
python -m scorpus.cli convert --source dataset.h5ad --output dataset.corpus
```

`scorpus` picks the count matrix for you: it samples `X`, `raw.X`, and every
layer for integer-like counts, prefers a direct source over a recoverable one,
and revalidates **every chunk** during the write rather than trusting the
sample. When it cannot find counts it stops and tells you what to pass:

```python
from_h5ad("log_norm.h5ad", "out.corpus", attempt_conversion=True)   # recover counts
from_h5ad("a.h5ad", "out.corpus", layers="counts")                  # pick explicitly
from_h5ad("norm.h5ad", "out.corpus", layers="X", no_count=True)     # keep non-counts as-is
```

The manifest records which of the three you got — `counts`, `recovered_counts`,
or `non_count` — so a downstream reader never has to guess.

## Compose

Convert datasets independently, then join them in memory. Harmonization is
ordinary pandas; there is no schema DSL to learn.

```python
from scorpus import load_corpus, concat

a, b = load_corpus("a.corpus"), load_corpus("b.corpus")

a_obs, b_obs = a.obs.copy(), b.obs.copy()
a_obs["pert"] = a_obs["target"].replace({"NTC": "ctrl"})
b_obs["pert"] = b_obs["genotype"].replace({"WT": "ctrl"})

combined = concat(
    {"a": a, "b": b},
    obs={"a": a_obs, "b": b_obs},
    obs_columns={"pert": {"a": "pert", "b": "pert"},
                 "cell_type": {"a": "celltype", "b": "annotation"}},
    feature_key={"a": "ensembl_id", "b": "gene_id"},
)
```

The view holds **references** to the member readers — no expression data is
copied. Without `feature_key`, features stay member-local: `scorpus` will not
assume two similarly named genes are the same gene. `combined.feature_presence`
records which member actually measured each feature, so zeros introduced by an
outer join are never mistaken for measured zeros.

## Hand back to AnnData

```python
adata = corpus.to_anndata_lazy(chunk_rows=4096)   # Dask X, stays on disk
small = corpus.to_anndata(global_row_indices=[2, 0, 5])   # eager, selected rows
corpus.write_h5ad("export.h5ad")                  # new file, reopen backed="r"
```

After export, Scanpy/RAPIDS owns normalization, PCA, neighbors, UMAP, and
clustering. Results can be joined back with `corpus.add_obs_meta(...)`.

## Train

```python
from scorpus.loaders import build_loader

for batch in build_loader(corpus, seq_len=1024, batch_size=128):
    ...
```

## Documentation

- [Conversion and composition](docs/composable_corpora.md) — start here
- [AnnData / Scanpy / RAPIDS handoff](docs/anndata_scanpy_handoff.md)
- [Architecture](DESIGN.md)
- [Backend policy](docs/backend_note.md)

### Curated corpus releases (advanced)

A longer reviewed pipeline exists for publishing corpora with harmonized,
audited metadata: `inspect → materialize → draft-schema → canonicalize`. Most
users do not need it — `from_h5ad` plus `concat` covers conversion, loading,
and composition. See [Inspection & materialization](docs/inspect_materialize.md)
and the [Canonicalization handbook](docs/canonicalization_handbook.md).

## The name

<img src="docs/assets/scorpus_charioteer.png" alt="" width="150" align="right">

Scorpus was the most celebrated charioteer of first-century Rome, winning over
two thousand races before dying young in the Circus Maximus. The name fits a
library whose whole job is moving single-cell data quickly — and it is a corpus
tool, so the pun was irresistible.

<br clear="right">

## Notes

- Treat source `.h5ad` files as read-only.
- Convert large files on Slurm using the project execution environment.
- Outputs are never overwritten; the manifest is written last, so an
  interrupted run cannot be mistaken for a complete corpus.
- Only the selected matrix and its obs/var are converted. Other layers, `obsm`,
  `obsp`, and `uns` are not copied.
