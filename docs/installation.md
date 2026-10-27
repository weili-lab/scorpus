# Installation

`scorpus` needs Python 3.11 or newer. It does not need torch: models build
their own tensors from what scorpus reads.

## Install the package

```bash
git clone https://github.com/weili-lab/scorpus.git
cd scorpus

# conda/mamba, or any other environment manager
conda create -n scorpus python=3.12 -y
conda activate scorpus

pip install -e .
```

This covers conversion, composition, reading, pairing and AnnData export.

Optional extras:

```bash
pip install -e ".[demo]"   # scanpy, dask[dataframe], requests, tqdm: the walkthroughs
pip install -e ".[docs]"   # sphinx, myst-nb, furo: build this site
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
