<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/scorpus_logo.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/scorpus_logo_light.png">
  <img src="docs/assets/scorpus_logo_light.png" alt="scorpus" width="440">
</picture>

**Sparse single-cell corpora — cheap to random-access, easy to compose,
ready to hand back to AnnData.**

[Documentation](https://scorpus.readthedocs.io/en/latest/) ·
[Quickstart](https://scorpus.readthedocs.io/en/latest/quickstart.html) ·
[Conversion &amp; composition](https://scorpus.readthedocs.io/en/latest/composable_corpora.html)

[![tests](https://github.com/weili-lab/scorpus/actions/workflows/tests.yml/badge.svg)](https://github.com/weili-lab/scorpus/actions/workflows/tests.yml)
[![ruff](https://github.com/weili-lab/scorpus/actions/workflows/ruff.yml/badge.svg)](https://github.com/weili-lab/scorpus/actions/workflows/ruff.yml)

</div>

---

`scorpus` turns single-cell `.h5ad` files into sparse on-disk corpora that are
cheap to random-access, easy to compose, and ready to hand back to AnnData.
Counts are stored as [Lance](https://github.com/lancedb/lance) tables, one row
per cell, so any set of cells can be read straight off disk without loading the
rest.

It is built for three jobs:

- **Large model training** — sparse batch loading straight off disk.
- **Perturbation model training** — paired sampling with fast random access.
- **Routine single-cell analysis** — hand the corpus back to Scanpy/RAPIDS.

## Install

```bash
pip install -e .           # conversion, composition, reading, pairing, AnnData export
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
adata = corpus.to_anndata_lazy()                  # Dask X in 8,192-cell chunks, stays on disk
small = corpus.to_anndata(global_row_indices=[2, 0, 5])   # eager, selected rows
corpus.write_h5ad("export.h5ad")                  # new file, reopen backed="r"
```

After export, Scanpy/RAPIDS owns normalization, PCA, neighbors, UMAP, and
clustering. Results can be joined back with `corpus.add_obs_meta(...)`.

## Train

scorpus gives rows and expression; tokenization, gene sampling and tensors belong to the model.
Draw every cell once per epoch, read a batch sparse, or dense on the corpus gene axis:

```python
from scorpus.loaders import dense_counts
from scorpus.pairing import cells, make_groups

groups = make_groups(corpus.metadata_index.df, by=["dataset_id", "cell_type"])
for rows in cells(groups, epoch=0, batch_size=128):            # same_group=True: one group per batch
    batch = corpus.expression_reader.read_expression_flat(rows.tolist())   # sparse, per-dataset genes
    counts, measured = dense_counts(corpus, batch)             # cells x genes, and which were measured
```

To train on pairs of cells (for example controls → perturbed cells of the same cell line),
describe which pairs are allowed and draw them with `scorpus.pairing`:

```python
import polars as pl
from scorpus.pairing import Sampler, cross, make_groups

groups = make_groups(corpus.metadata_index.df, by=["cell_line", "perturbation"])
is_ctrl = pl.col("perturbation") == "ctrl"
pairs = cross(groups, on=["cell_line"], source=is_ctrl, target=~is_ctrl)
Sampler(groups, pairs, n_source=1).sample(step=0)   # [([source row], [target row])]
```

## Documentation

The full documentation, including installation, tutorials and guides, is at
**[scorpus.readthedocs.io](https://scorpus.readthedocs.io/en/latest/)**:

- [Installation](https://scorpus.readthedocs.io/en/latest/installation.html) and [quickstart](https://scorpus.readthedocs.io/en/latest/quickstart.html)
- [Conversion and composition](https://scorpus.readthedocs.io/en/latest/composable_corpora.html): start here
- [Pairing source and target cells](https://scorpus.readthedocs.io/en/latest/pairing.html) and the [pairing walkthrough](https://scorpus.readthedocs.io/en/latest/tutorials/pairing_walkthrough.html)
- [AnnData / Scanpy / RAPIDS handoff](https://scorpus.readthedocs.io/en/latest/anndata_scanpy_handoff.html)
- [Architecture](https://scorpus.readthedocs.io/en/latest/design.html) and [storage backend (Lance)](https://scorpus.readthedocs.io/en/latest/backend_note.html)

The site is built from `docs/` (Sphinx, see `.readthedocs.yaml`); edit the pages
there.

### Curated corpus releases (advanced)

A longer reviewed pipeline exists for publishing corpora with harmonized,
audited metadata: `inspect → materialize → draft-schema → canonicalize`. Most
users do not need it — `from_h5ad` plus `concat` covers conversion, loading,
and composition. See the [bash walkthrough](https://scorpus.readthedocs.io/en/latest/bash_demo.html),
[inspection & materialization](https://scorpus.readthedocs.io/en/latest/inspect_materialize.html) and the
[canonicalization handbook](https://scorpus.readthedocs.io/en/latest/canonicalization_handbook.html).

## The name

<img src="docs/assets/scorpus_charioteer.png" alt="" width="150" align="right">

Scorpus was the most celebrated charioteer of first-century Rome, winning over
two thousand races before dying young in the Circus Maximus. The name fits a
library whose whole job is moving single-cell data quickly — and it is a corpus
tool, so the pun was irresistible.

Look at what our charioteer is holding: a lance, raised high. That is no
accident: every corpus rides on [Lance](https://github.com/lancedb/lance), the
columnar format that lets scorpus pick any cell off disk as quickly as a lance
finds its mark. The lance turns up in the wordmark too, as the leg of the **R**,
and the three cells riding in the chariot are your data, along for the race.

<br clear="right">

## Notes

- Treat source `.h5ad` files as read-only.
- Convert large files on Slurm using the project execution environment.
- Outputs are never overwritten; the manifest is written last, so an
  interrupted run cannot be mistaken for a complete corpus.
- Only the selected matrix and its obs/var are converted. Other layers, `obsm`,
  `obsp`, and `uns` are not copied.
