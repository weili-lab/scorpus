# Installation

`scorpus` needs Python 3.11 or newer. It does not need torch: models build
their own tensors from what scorpus reads.

## Install the package

Install a release straight from GitHub (the tag pins the version):

```bash
# conda/mamba, or any other environment manager
conda create -n scorpus python=3.12 -y
conda activate scorpus

pip install "scorpus @ git+https://github.com/weili-lab/scorpus.git@v0.5.0"
```

or, to work on scorpus itself, from a clone:

```bash
git clone https://github.com/weili-lab/scorpus.git
cd scorpus
pip install -e .
```

The base install covers conversion, composition, reading, training batches,
pairing and in-memory AnnData export (`to_anndata`). Optional extras add the
rest; with a clone, write them as `pip install -e ".[scanpy]"`:

| Extra | Adds | For |
|---|---|---|
| `lazy` | dask | `to_anndata_lazy()` and `write_h5ad()`: AnnData whose counts stay on disk |
| `scanpy` | dask, scanpy | Scanpy on the lazy export (CPU) |
| `gpu` | dask, cupy, rapids-singlecell | rapids-singlecell on a lazy export whose chunks live on the GPU (CUDA 12) |
| `all` | all of the above | |
| `demo` | `scanpy` plus requests | the walkthrough notebooks |
| `docs` | sphinx, myst-nb, furo | building this site |

```bash
pip install "scorpus[scanpy] @ git+https://github.com/weili-lab/scorpus.git@v0.5.0"
```

RAPIDS wheels come from NVIDIA's package index, so the GPU extra needs it as
well:

```bash
pip install "scorpus[gpu] @ git+https://github.com/weili-lab/scorpus.git@v0.5.0" \
    --extra-index-url https://pypi.nvidia.com
```

Check the install:

```python
from scorpus import from_h5ad, load_corpus, concat
from scorpus.pairing import Sampler, cells, make_groups
```

## Demo data

The walkthroughs use two small `.h5ad` subsets hosted on HuggingFace
([weililab/scorpus-demo](https://huggingface.co/datasets/weililab/scorpus-demo)):
Marson D2 Rest (CRISPRi, 2,720 cells) and Xorion HCT116 (dual guide, 2,720
cells). Download them from the repository root:

```bash
python scripts/download_demo_data.py --output-dir ./demo_data                 # huggingface_hub
python scripts/download_demo_data.py --output-dir ./demo_data --method wget   # no extra packages
```

The script downloads `h5ad/demo_marson_d2_rest.h5ad`,
`h5ad/demo_xorion_hct116_dual_guide.h5ad` and `checksums.txt`, then checks the
files against the checksums. Alternatively, with `huggingface_hub` installed:

```bash
huggingface-cli download weililab/scorpus-demo --local-dir ./demo_data --repo-type dataset
```

## Build this site locally

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## Next

- [Quickstart](quickstart.md): convert a file, compose several, read and pair cells
- [Bash walkthrough](bash_demo.md): the curated route on the demo data
