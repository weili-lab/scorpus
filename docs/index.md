# scorpus

**Sparse single-cell corpora: cheap to random-access, easy to compose, ready to
hand back to AnnData.**

`scorpus` turns `.h5ad` files into sparse on-disk corpora. Conversion is one
call, with no schema and no intermediate files, and composing several datasets
into a training corpus is ordinary pandas plus an explicit column mapping.

```python
from scorpus import from_h5ad

corpus = from_h5ad("dataset.h5ad", "dataset.corpus")
adata = corpus.to_anndata_lazy()   # original obs/var; expression stays on disk
```

It is built for three jobs:

- **Large model training**: sparse batches read straight off disk.
- **Perturbation model training**: paired sampling of source and target cells
  with fast random access.
- **Routine single-cell analysis**: hand the corpus back to Scanpy or RAPIDS.

## Two routes to a corpus

Conversion is the default and covers most use:

```text
source h5ad -> from_h5ad() -> concat() -> load_corpus() -> reading, pairing or AnnData
```

Curation is an optional reviewed pipeline for publishing corpora with audited,
harmonized metadata:

```text
source h5ad -> inspect -> materialize -> canonicalize -> load_corpus()
```

Both give the same `Corpus` object and share the same sparse expression
readers; `load_corpus()` tells them apart from the manifest on disk. Most users
only need the first route. The second is documented under
[Curated Corpora](bash_demo.md).

```{toctree}
:maxdepth: 1
:caption: Get Started

installation
quickstart
```

```{toctree}
:maxdepth: 1
:caption: Guides

composable_corpora
pairing
tutorials/pairing_walkthrough
anndata_scanpy_handoff
perttf_loader
backend_note
```

```{toctree}
:maxdepth: 1
:caption: Curated Corpora (advanced)

bash_demo
tutorials/demo_walkthrough
demo_canonicalization
scanpy_rapids
inspect_materialize
canonicalization_handbook
```

```{toctree}
:maxdepth: 1
:caption: About

design
```

## The name

```{image} assets/scorpus_charioteer.png
:alt: ""
:width: 150px
:align: right
```

Scorpus was the most celebrated charioteer of first-century Rome, winning over
two thousand races before dying young in the Circus Maximus. The name fits a
library whose whole job is moving single-cell data quickly, and it is a corpus
tool, so the pun was irresistible.

## Links

- [GitHub repository](https://github.com/weili-lab/scorpus)
- [Demo data on HuggingFace](https://huggingface.co/datasets/weililab/scorpus-demo)
- [pertTF](https://perttf.readthedocs.io/en/latest/), which trains on scorpus corpora
- [Li lab website](https://weililab.org)
