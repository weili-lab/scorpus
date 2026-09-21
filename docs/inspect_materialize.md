# Inspection and Materialization

This document shows how to turn raw `.h5ad` files into a materialized corpus.
Canonical schema review and canonicalization are covered separately in
`canonicalization_handbook.md`.

## Workflow

```text
inspect raw h5ad
  -> review dataset-summary.yaml
  -> materialize sparse counts + raw sidecars
  -> validate corpus registration
```

Inspection is metadata-first and samples matrix candidates. Materialization uses
the inspection decision to stream the selected count matrix into a corpus.

## Direct CLI Inspection

```bash
PYTHONPATH=src python -m scorpus.cli inspect \
  --source /path/to/dataset.h5ad \
  --dataset-id my_dataset \
  --output-dir ./artifacts/review
```

This writes:

```text
./artifacts/review/my_dataset/dataset-summary.yaml
```

The summary contains:

- source identity and dimensions
- obs and var field profiles
- `.X`, `.raw.X`, and layer count-source candidates
- selected count source
- recovery decision, if needed
- `materialization_readiness`
- likely control-label candidates for later canonical schema review

Materialization only proceeds when `materialization_readiness: pass`.

## Batch CLI Inspection

Batch inspection reads a YAML config:

```yaml
output_root: ./artifacts/review
datasets:
  - dataset_id: dataset_a
    source_path: /path/to/dataset_a.h5ad
    source_release: dataset_a
  - dataset_id: dataset_b
    source_path: /path/to/dataset_b.h5ad
    source_release: dataset_b
```

Run:

```bash
PYTHONPATH=src python -m scorpus.cli inspect \
  --config ./artifacts/inspection-batch.yaml \
  --workers 1
```

Outputs:

```text
./artifacts/review/inspection-manifest.yaml
./artifacts/review/dataset_a/dataset-summary.yaml
./artifacts/review/dataset_b/dataset-summary.yaml
```

For large `.h5ad` files, run this on Slurm CPU in the project environment.

## Reviewing `dataset-summary.yaml`

Before materialization, check these fields:

- `materialization_readiness`: must be `pass`
- `count_source_decision.selected_candidate`: `.X`, `.raw.X`, or `.layers[name]`
- `count_source_decision.uses_recovery`: whether reverse-normalization is expected
- `count_source_candidates`: sampled integer and non-negativity evidence
- `obs_fields` and `var_fields`: field names and example values for later schema review
- `control_label_candidates`: possible control labels for canonicalization

If readiness is `needs-review` or `fail`, do not materialize until the source or
inspection decision is resolved.

## Create a New Corpus

Create a new aggregate Lance corpus from one dataset:

```bash
PYTHONPATH=src python -m scorpus.cli materialize \
  --mode create \
  --source /path/to/dataset.h5ad \
  --dataset-id my_dataset \
  --inspection-summary ./artifacts/review/my_dataset/dataset-summary.yaml \
  --output-corpus ./artifacts/corpus \
  --backend lance \
  --topology aggregate \
  --corpus-id my_corpus
```

For `--mode create`, `--backend` is required. If `--topology` is omitted, the CLI
defaults to federated, so pass `--topology aggregate` when creating the default
recommended aggregate Lance corpus.

## Append to an Existing Corpus

Append later datasets into the same corpus:

```bash
PYTHONPATH=src python -m scorpus.cli materialize \
  --mode append \
  --source /path/to/later_dataset.h5ad \
  --dataset-id later_dataset \
  --inspection-summary ./artifacts/review/later_dataset/dataset-summary.yaml \
  --output-corpus ./artifacts/corpus
```

For append, the existing `corpus-index.yaml` supplies backend and topology. If
`--backend` or `--topology` are passed, they must match the existing corpus.

Dataset IDs are immutable within one corpus. The CLI refuses to append a dataset
ID that already exists in the corpus index.

## Materialize Multiple Datasets

Use a CSV input list:

```csv
source,dataset_id,inspection_summary
/path/to/dataset_a.h5ad,dataset_a,./artifacts/review/dataset_a/dataset-summary.yaml
/path/to/dataset_b.h5ad,dataset_b,./artifacts/review/dataset_b/dataset-summary.yaml
```

Then run:

```bash
PYTHONPATH=src python -m scorpus.cli materialize \
  --mode create \
  --input-list ./artifacts/materialize-inputs.csv \
  --output-corpus ./artifacts/corpus \
  --backend lance \
  --topology aggregate
```

The first dataset creates the corpus and later rows are appended in CSV order.

`--input-dir` can also scan a directory of `.h5ad` files, but it expects summary
files named `{stem}-summary.yaml` in the input directory or in
`--inspection-summary-dir`. Use `--input-list` when summaries are stored in the
normal inspection output layout.

## Dry Runs

Check inputs and print the planned materialization without writing data:

```bash
PYTHONPATH=src python -m scorpus.cli materialize \
  --mode create \
  --source /path/to/dataset.h5ad \
  --dataset-id my_dataset \
  --inspection-summary ./artifacts/review/my_dataset/dataset-summary.yaml \
  --output-corpus ./artifacts/corpus \
  --backend lance \
  --topology aggregate \
  --dry-run
```

## Python Inspection API

```python
from pathlib import Path

from scorpus.inspectors import inspect_target
from scorpus.inspectors.models import InspectionTarget

artifacts = inspect_target(
    InspectionTarget(
        dataset_id="my_dataset",
        source_path="/path/to/dataset.h5ad",
        source_release="my_dataset",
    ),
    Path("./artifacts/review"),
)

print(artifacts.inspection_summary)
print(artifacts.materialization_readiness)
```

For batch inspection in Python, use `InspectionBatchConfig` and `run_batch(...)`.

## Python Materialization API

```python
from pathlib import Path

from scorpus.materializers import DatasetMaterializer
from scorpus.materializers.models import OutputRoots
from scorpus.materializers.paths import resolve_corpus_paths

corpus_root = Path("./artifacts/corpus")
dataset_id = "my_dataset"
paths = resolve_corpus_paths(
    topology="aggregate",
    corpus_root=corpus_root,
    dataset_id=dataset_id,
)

materializer = DatasetMaterializer(
    source_path="/path/to/dataset.h5ad",
    inspection_summary_path="./artifacts/review/my_dataset/dataset-summary.yaml",
    output_roots=OutputRoots(
        metadata_root=str(paths.meta_root),
        matrix_root=str(paths.matrix_root),
    ),
    dataset_id=dataset_id,
    backend="lance",
    topology="aggregate",
    corpus_index_path=str(corpus_root / "corpus-index.yaml"),
    corpus_id="my_corpus",
    register=True,
    mode="create",
    dataset_index=0,
    global_row_start=0,
)

manifest = materializer.materialize()
print(manifest.cell_count, manifest.feature_count)
```

Most users should use the CLI because it handles append bookkeeping,
`dataset_index`, `global_row_start`, and aggregate writer state across multiple
datasets.

## Output Layouts

Aggregate topology:

```text
corpus/
├── corpus-index.yaml
├── global-metadata.yaml
├── matrix/
└── meta/
    └── my_dataset/
        ├── dataset-summary.yaml
        ├── materialization-manifest.yaml
        ├── raw-obs.parquet
        ├── raw-var.parquet
        ├── size-factor.parquet
        └── hvg.parquet
```

Federated topology:

```text
corpus/
├── corpus-index.yaml
├── global-metadata.yaml
└── my_dataset/
    ├── matrix/
    └── meta/
        ├── dataset-summary.yaml
        ├── materialization-manifest.yaml
        ├── raw-obs.parquet
        ├── raw-var.parquet
        ├── size-factor.parquet
        └── hvg.parquet
```

## Validation After Materialization

Run corpus validation:

```bash
PYTHONPATH=src python -m scorpus.cli corpus-validate \
  ./artifacts/corpus/corpus-index.yaml
```

Manual checks:

- `corpus-index.yaml` lists the expected dataset IDs in the expected order
- each dataset has `materialization-manifest.yaml`
- each manifest has `integer_verified: true`
- each manifest records the expected `backend` and `topology`
- `raw-obs.parquet`, `raw-var.parquet`, `size-factor.parquet`, and `hvg.parquet` exist

After this, continue with `canonicalization_handbook.md`.

## Safety Notes

- Treat source `.h5ad` files as read-only.
- Do not write outputs into protected symlink roots such as `data/`, `pertTF/`, or `perturb/`.
- Materialization does not apply `obs_filter`. Pre-filter the source `.h5ad` if only a subset of cells should enter the corpus.
- Keep dataset IDs stable; they become part of corpus paths and runtime metadata.
