# Jupyter Materialization Demo

This page walks through the full demo workflow as a series of Python cells you
can run in a Jupyter notebook, IPython session, or plain script. Each cell is
self-contained and builds on the previous one.

An executed version of this walkthrough is available as a rendered notebook:
[demo walkthrough notebook](demo_walkthrough.ipynb).

## Prerequisites

- [Installation](installation.md) completed and `pip install -e ".[demo]"` succeeded.
- A working Python environment with `scorpus`, `anndata`, and `scanpy` importable.
- Demo data downloaded (see the download cell below).

---

## Download demo data

```python
from pathlib import Path
import sys
import requests

repo_root = next(
    p for p in (Path.cwd(), *Path.cwd().parents)
    if (p / "src" / "scorpus").exists()
)
sys.path.insert(0, str(repo_root / "src"))

demo_root = repo_root / "demo_data"
base_url = "https://huggingface.co/datasets/weililab/scorpus-demo/resolve/main"
files = [
    "h5ad/demo_marson_d2_rest.h5ad",
    "h5ad/demo_xorion_hct116_dual_guide.h5ad",
    "checksums.txt",
]

for rel in files:
    out = demo_root / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"  {rel}: already exists")
        continue
    response = requests.get(f"{base_url}/{rel}", stream=True)
    response.raise_for_status()
    with open(out, "wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            fh.write(chunk)

for rel in ["h5ad/demo_marson_d2_rest.h5ad", "h5ad/demo_xorion_hct116_dual_guide.h5ad"]:
    f = demo_root / rel
    size_mb = f.stat().st_size / 1e6 if f.exists() else 0
    print(f"  {rel}: {size_mb:.1f} MB {'✓' if f.exists() else '✗ missing'}")
```

---

## Inspect raw h5ad metadata

Inspection reads metadata profiles and samples matrix candidates without loading
full count matrices. The output `dataset-summary.yaml` tells you whether the
file is ready for materialization.

```python
from pathlib import Path
from scorpus.inspectors import inspect_target
from scorpus.inspectors.models import DatasetSummaryDocument, InspectionTarget

review_dir = repo_root / "artifacts" / "review"
review_dir.mkdir(parents=True, exist_ok=True)

datasets = [
    ("marson_d2_rest", str(demo_root / "h5ad" / "demo_marson_d2_rest.h5ad")),
    ("xorion_hct116_dual_guide", str(demo_root / "h5ad" / "demo_xorion_hct116_dual_guide.h5ad")),
]

for ds_id, source_path in datasets:
    artifacts = inspect_target(
        InspectionTarget(dataset_id=ds_id, source_path=source_path, source_release=ds_id),
        review_dir,
    )
    summary = DatasetSummaryDocument.from_yaml_file(artifacts.inspection_summary)
    print(f"\n{ds_id}:")
    print(f"  cells: {summary.dataset.obs_rows}, features: {summary.dataset.var_rows}")
    print(f"  count source: {summary.count_source_decision.selected_candidate}")
    print(f"  readiness: {summary.materialization_readiness}")
    print(f"  obs fields ({len(summary.obs_fields)}): {[f.name for f in summary.obs_fields[:10]]}...")
```

---

## Quick peek at obs metadata

Let's look at a few rows of raw metadata to understand what each dataset
contains before canonicalization:

```python
import anndata

for ds_id, source_path in datasets:
    adata = anndata.read_h5ad(source_path, backed="r")
    print(f"\n--- {ds_id} ---")
    print(f"Shape: {adata.shape}")
    print(f"obs columns: {list(adata.obs.columns)}")

    # Show a few key columns
    cols = [c for c in ["guide_id", "guide_type", "perturbed_gene_name",
                        "guide_target", "gene_target", "sample", "pass_guide_filter"]
            if c in adata.obs.columns]
    if cols:
        display_cols = cols[:6]
        print(adata.obs[display_cols].head(5).to_string())
```

Expected observations:

- **Marson**: has `guide_id`, `guide_type`, `perturbed_gene_name`, `lane_id`, QC columns
- **Xorion**: has `guide_target`, `gene_target`, `sample`, `pass_guide_filter`, QC columns

---

## Materialize a federated Lance corpus (both datasets at once)

This builds a **federated** Lance corpus: each dataset gets its own isolated
`matrix/` and `meta/` directory under `{corpus}/{dataset_id}/`. Both datasets
are materialized in one Python loop using the materializer API.

```python
from scorpus.materializers import DatasetMaterializer
from scorpus.materializers.models import OutputRoots, CorpusIndexDocument
from scorpus.materializers.paths import resolve_corpus_paths

corpus_root = repo_root / "artifacts" / "demo_corpus"
corpus_root.mkdir(parents=True, exist_ok=True)

for i, (ds_id, source_path) in enumerate(datasets):
    paths = resolve_corpus_paths("federated", corpus_root, ds_id)
    mode = "create" if i == 0 else "append"

    materializer = DatasetMaterializer(
        source_path=source_path,
        inspection_summary_path=str(review_dir / ds_id / "dataset-summary.yaml"),
        output_roots=OutputRoots(
            metadata_root=str(paths.meta_root),
            matrix_root=str(paths.matrix_root),
        ),
        dataset_id=ds_id,
        backend="lance",
        topology="federated",
        corpus_index_path=str(corpus_root / "corpus-index.yaml"),
        corpus_id="demo_corpus",
        register=True,
        mode=mode,
        dataset_index=i,
        global_row_start=0,  # corpus-index bookkeeping handles federated global ranges
    )
    manifest = materializer.materialize()
    print(f"{mode}: {ds_id} -> {manifest.cell_count} cells, {manifest.feature_count} features")

index_doc = CorpusIndexDocument.from_yaml_file(corpus_root / "corpus-index.yaml")
print([d.dataset_id for d in index_doc.datasets])
```

The federated layout writes:

```text
artifacts/demo_corpus/
├── corpus-index.yaml
├── global-metadata.yaml
├── marson_d2_rest/
│   ├── matrix/marson_d2_rest.lance
│   └── meta/
│       ├── raw-obs.parquet
│       ├── raw-var.parquet
│       ├── size-factor.parquet
│       └── hvg.parquet
└── xorion_hct116_dual_guide/
    ├── matrix/xorion_hct116_dual_guide.lance
    └── meta/
        ├── raw-obs.parquet
        ├── raw-var.parquet
        ├── size-factor.parquet
        └── hvg.parquet
```

For the command-line equivalent, use the [Bash Demo](bash_demo.md).

---

## Install reviewed schemas

Copy the reviewed demo final schemas into the corpus. These contain the
biological decisions that turn raw metadata columns into canonical labels:

```python
import shutil

schema_root = repo_root / "examples" / "demo_canonicalization"
for ds_id, _ in datasets:
    source = schema_root / f"{ds_id}.final-schema.yaml"
    target = corpus_root / ds_id / "meta" / "final-schema.yaml"
    shutil.copyfile(source, target)
    print(f"installed {ds_id} -> {target}")
```

```python
# Verify they landed
for ds_id in ["marson_d2_rest", "xorion_hct116_dual_guide"]:
    schema_path = corpus_root / ds_id / "meta" / "final-schema.yaml"
    print(f"  {ds_id}: {'✓' if schema_path.exists() else '✗ missing'}")
```

Read the quick decision sheets that explain what each schema does:

```python
import yaml
examples_root = Path("examples/demo_canonicalization")
for ds_id in ["marson_d2_rest", "xorion_hct116_dual_guide"]:
    hints = yaml.safe_load((examples_root / f"{ds_id}.schema-hints.yaml").read_text())
    print(f"\n--- {ds_id} hints ---")
    for field, info in hints.items():
        print(f"  {field}: {info}")
```

For a deeper explanation, see [Canonicalization](demo_canonicalization.md).

---

## Canonicalize

Apply the reviewed schemas to produce canonical obs/var metadata:

```python
from scorpus.canonical import run_canonicalization
from scorpus.materializers.models import MaterializationManifest

index_doc = CorpusIndexDocument.from_yaml_file(corpus_root / "corpus-index.yaml")
for ds in index_doc.datasets:
    manifest = MaterializationManifest.from_yaml_file(corpus_root / ds.manifest_path)
    paths = resolve_corpus_paths("federated", corpus_root, ds.dataset_id)

    result = run_canonicalization(
        dataset_id=ds.dataset_id,
        raw_obs_path=corpus_root / manifest.raw_cell_meta_path,
        raw_var_path=corpus_root / manifest.raw_feature_meta_path,
        size_factor_path=corpus_root / manifest.size_factor_parquet_path,
        schema_path=paths.meta_root / "final-schema.yaml",
        output_root=paths.canonical_meta_root,
    )
    print(f"{result.dataset_id}: {result.obs_rows} obs rows, {result.var_rows} var rows")
```

---

## Validate and load the corpus

```python
from scorpus.loaders.validation import validate_corpus_structure
from scorpus.loaders import load_corpus

report = validate_corpus_structure(corpus_root)
print(report["status"], report["topology"], report["total_rows"])

corpus = load_corpus(str(corpus_root))
print(f"Datasets: {corpus.dataset_ids}")
print(f"Total cells: {len(corpus.metadata_index)}")
print(f"Global vocab size: {corpus.feature_registry.global_vocab_size}")
```

---

## Inspect canonical metadata

Look at a few rows to confirm canonical labels are sensible:

```python
import polars as pl

# Grab first 10 rows from each dataset
marson_rows = corpus.take_metadata(
    list(range(0, 10)),
    columns=["dataset_id", "perturb_label", "condition", "perturb_type",
             "cell_context", "batch_id", "gene_id"],
)
xorion_rows = corpus.take_metadata(
    list(range(2720, 2730)),
    columns=["dataset_id", "perturb_label", "condition", "perturb_type",
             "cell_context", "batch_id", "gene_id"],
)

print("--- Marson ---")
print(marson_rows)
print("\n--- Xorion ---")
print(xorion_rows)
```

```python
# Quick stats
meta = corpus.metadata_index.df
all_perturb_labels = meta.get_column("perturb_label")

# Count control vs treated
ctrl_count = (pl.Series(all_perturb_labels) == "ctrl").sum()
total = len(all_perturb_labels)
print(f"Control rows: {ctrl_count} / {total} ({100*ctrl_count/total:.0f}%)")

# Unique conditions
conditions = meta.get_column("condition")
unique = set(conditions.to_list())
print(f"Unique conditions: {len(unique)}")
print(f"Sample conditions: {sorted(unique)[:10]}...")
```

---

## pertTF paired loading

The paired loader, perturbation sampler, and model-specific vocabulary adapter
are no longer part of `scorpus`. They live in pertTF as
`perttf.model.corpus_adapter`, and `perttf.model.corpus_data.produce_corpus_datasets`
builds the full training/validation dictionary from a loaded corpus.

For a model-independent sparse batch loader, `scorpus` provides:

```python
from scorpus.loaders import build_loader

batch = next(iter(build_loader(corpus, seq_len=64, batch_size=4)))
print(sorted(batch.keys()))
```

See [pertTF integration](perttf_loader.md).

---

## AnnData handoff (Dask-backed)

Export the combined corpus as a Dask-backed AnnData with inner-join on features:

```python
adata = corpus.to_anndata_lazy(
    dataset_id=list(corpus.dataset_ids),
    obs_columns=["perturb_label", "condition", "cell_context", "batch_id"],
    chunk_rows=1024,
    var_join="inner",
)

print(f"adata shape: {adata.shape}")
print(f"adata.X type: {type(adata.X)}")
print(f"obs columns: {list(adata.obs.columns)}")
print(f"var columns: {list(adata.var.columns)}")

# Show the intersection gene axis (first few)
print(f"\nIntersection features: {adata.n_vars}")
print(adata.var.head(5).to_string())
```

```python
# Quick Scanpy smoke on the Dask-backed AnnData
import scanpy as sc

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

print(f"HVG selected: {adata.var['highly_variable'].sum()}")

# Force a small computation to confirm Dask works
subset = adata[:100, :100].to_memory()
print(f"Subset shape: {subset.shape}")
print(f"Subset X nnz: {subset.X.nnz}")
```

For the full Scanpy/RAPIDS story, see [Scanpy & RAPIDS](scanpy_rapids.md).

---

## Re-running from scratch

Delete the artifacts directory and re-run from the download cell:

```python
import shutil
shutil.rmtree("./artifacts", ignore_errors=True)
```

The demo data under `./demo_data/` does not need to be re-downloaded.

## Next steps

- **[Bash Demo](bash_demo.md)** — the CLI equivalent for scripting
- **[Canonicalization](demo_canonicalization.md)** — understand the two schema decisions
- **[pertTF integration](perttf_loader.md)** — where the paired loader now lives
- **[Scanpy & RAPIDS](scanpy_rapids.md)** — Dask-backed AnnData, CPU Scanpy, GPU RAPIDS
