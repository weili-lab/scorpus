# scorpus

Sparse single-cell corpora that are cheap to random-access, easy to compose, and
ready to hand back to AnnData.

`scorpus` turns `.h5ad` files into sparse on-disk corpora. Conversion is one
call — no schema, no intermediate artifacts — and composing several datasets
into a training corpus is ordinary pandas plus an explicit column mapping.

```python
from scorpus import from_h5ad

corpus = from_h5ad("dataset.h5ad", "dataset.corpus")
adata = corpus.to_anndata_lazy()   # original obs/var; expression stays on disk
```

<div class="grid cards" markdown>

-   :material-flash:{ .lg .middle } __Quickstart__

    ---

    Convert one `.h5ad`, compose several, hand back to AnnData.

    [:octicons-arrow-right-24: Start here](quickstart.md)

-   :material-download:{ .lg .middle } __Installation__

    ---

    Set up your environment with one command.

    [:octicons-arrow-right-24: Install](installation.md)

-   :material-layers-triple:{ .lg .middle } __Conversion & Composition__

    ---

    Count policies, recovery, feature alignment, and temporary federation.

    [:octicons-arrow-right-24: Reference](composable_corpora.md)

-   :material-chart-scatter-plot:{ .lg .middle } __Scanpy & RAPIDS__

    ---

    Export AnnData, run Scanpy preprocessing, explore GPU acceleration.

    [:octicons-arrow-right-24: Analysis handoff](scanpy_rapids.md)

</div>

## What it is for

- **Large model training** — sparse batch loading straight off disk.
- **Perturbation model training** — paired sampling with fast random access.
- **Routine single-cell analysis** — hand the corpus back to Scanpy/RAPIDS.

## Two routes to a corpus

Conversion is the default and covers most use:

```text
source h5ad -> from_h5ad() -> concat() -> load_corpus() -> loader or AnnData
```

Curation is an optional reviewed pipeline for publishing corpora with audited,
harmonized metadata:

```text
source h5ad -> inspect -> materialize -> canonicalize -> load_corpus()
```

Both return the same `Corpus` object and share the same sparse expression
readers. `load_corpus()` works out which one it is from the manifest on disk.

Most users only need the first. The curated route is documented under
[Curated Corpora](bash_demo.md).

## Reference

- [Conversion & composition](composable_corpora.md) — count policies, recovery,
  native backed h5ad export, temporary federation
- [AnnData handoff](anndata_scanpy_handoff.md) — corpus-to-AnnData export and
  the Scanpy/RAPIDS boundary
- [Backend notes](backend_note.md) — storage backend policy and selection
- [pertTF integration](perttf_loader.md) — where the paired loader now lives
