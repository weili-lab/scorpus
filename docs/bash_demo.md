# Bash Materialization Demo

This page walks through the full demo workflow using only copy-paste CLI
commands. When you finish you will have a validated, canonicalized corpus ready
for loading.

**Estimated time**: 5–10 minutes on a modern machine.

## Prerequisites

- [Installation](installation.md) completed and `pip install -e .` succeeded.
- Demo data downloaded under `./demo_data/` (see step below).
- All commands are run from the repository root (`scorpus/`).

## Step 0 — Download the demo data

If you haven't downloaded the demo `.h5ad` files yet:

```bash
python scripts/download_demo_data.py --output-dir ./demo_data
```

Expected contents:

```text
demo_data/
├── checksums.txt
├── h5ad/
│   ├── demo_marson_d2_rest.h5ad       (~25 MB, 2,720 cells × 18,130 features)
│   └── demo_xorion_hct116_dual_guide.h5ad  (~33 MB, 2,720 cells × 38,606 features)
```

## Step 1 — Inspect each dataset

Inspection reads metadata and samples matrix candidates without loading the full
count matrices:

```bash
PYTHONPATH=src python -m scorpus.cli inspect \
  --source ./demo_data/h5ad/demo_marson_d2_rest.h5ad \
  --dataset-id marson_d2_rest \
  --output-dir ./artifacts/review

PYTHONPATH=src python -m scorpus.cli inspect \
  --source ./demo_data/h5ad/demo_xorion_hct116_dual_guide.h5ad \
  --dataset-id xorion_hct116_dual_guide \
  --output-dir ./artifacts/review
```

This writes two summaries:

```text
artifacts/review/marson_d2_rest/dataset-summary.yaml
artifacts/review/xorion_hct116_dual_guide/dataset-summary.yaml
```

Each summary records source dimensions, obs/var field profiles, the selected
count source, and `materialization_readiness`. Materialization requires
`readiness: pass`.

## Step 2 — Create the corpus (first dataset)

Create a new federated Lance corpus with the Marson dataset:

```bash
PYTHONPATH=src python -m scorpus.cli materialize \
  --mode create \
  --source ./demo_data/h5ad/demo_marson_d2_rest.h5ad \
  --dataset-id marson_d2_rest \
  --inspection-summary ./artifacts/review/marson_d2_rest/dataset-summary.yaml \
  --output-corpus ./artifacts/demo_corpus \
  --backend lance \
  --topology federated
```

This writes:

- `artifacts/demo_corpus/corpus-index.yaml` — corpus registration
- `artifacts/demo_corpus/global-metadata.yaml` — global metadata
- `artifacts/demo_corpus/marson_d2_rest/matrix/` — sparse Lance count table
- `artifacts/demo_corpus/marson_d2_rest/meta/` — raw obs/var, size factors, HVG

## Step 3 — Append the second dataset

Append the Xorion dataset into the same corpus:

```bash
PYTHONPATH=src python -m scorpus.cli materialize \
  --mode append \
  --source ./demo_data/h5ad/demo_xorion_hct116_dual_guide.h5ad \
  --dataset-id xorion_hct116_dual_guide \
  --inspection-summary ./artifacts/review/xorion_hct116_dual_guide/dataset-summary.yaml \
  --output-corpus ./artifacts/demo_corpus
```

For append, the existing `corpus-index.yaml` supplies backend and topology — no
need to repeat `--backend` or `--topology`.

## Step 3.5 — Materialize both datasets at once (alternative)

Instead of running Step 2 and Step 3 separately, you can materialize both
datasets in a single CLI call using an input-list CSV. First create the CSV:

```bash
cat > ./artifacts/demo_inputs.csv << 'EOF'
source,dataset_id,inspection_summary
./demo_data/h5ad/demo_marson_d2_rest.h5ad,marson_d2_rest,./artifacts/review/marson_d2_rest/dataset-summary.yaml
./demo_data/h5ad/demo_xorion_hct116_dual_guide.h5ad,xorion_hct116_dual_guide,./artifacts/review/xorion_hct116_dual_guide/dataset-summary.yaml
EOF
```

Then materialize both in one call (the first row creates the corpus, the rest
are appended automatically):

```bash
PYTHONPATH=src python -m scorpus.cli materialize \
  --mode create \
  --input-list ./artifacts/demo_inputs.csv \
  --output-corpus ./artifacts/demo_corpus \
  --backend lance \
  --topology federated
```

This is the recommended one-call equivalent of Step 2 + Step 3.

## Step 4 — Install the reviewed demo schemas

Copy the reviewed final schemas from the repo into the corpus metadata layout:

```bash
PYTHONPATH=src python scripts/install_demo_schemas.py \
  --corpus ./artifacts/demo_corpus
```

Expected output:

```text
installed marson_d2_rest -> ...meta/marson_d2_rest/final-schema.yaml
installed xorion_hct116_dual_guide -> ...meta/xorion_hct116_dual_guide/final-schema.yaml
```

These schemas contain the biological decisions explained in
[Canonicalization](demo_canonicalization.md). The helper copies them from
`examples/demo_canonicalization/` into the corpus.

## Step 5 — Canonicalize

First run a dry-run to check schema resolution:

```bash
PYTHONPATH=src python -m scorpus.cli canonicalize \
  --corpus ./artifacts/demo_corpus \
  --dry-run
```

Then run the real canonicalization:

```bash
PYTHONPATH=src python -m scorpus.cli canonicalize \
  --corpus ./artifacts/demo_corpus
```

This applies the reviewed final schemas to each dataset's raw metadata and
writes canonical obs/var parquet files under:

```text
artifacts/demo_corpus/marson_d2_rest/meta/canonical_meta/
artifacts/demo_corpus/xorion_hct116_dual_guide/meta/canonical_meta/
```

Key canonical fields include `perturb_label`, `condition`, `perturb_type`,
`cell_context`, and `batch_id`. Control rows in both datasets normalize to
`ctrl`.

## Step 6 — Validate the corpus

Run the built-in corpus validator:

```bash
PYTHONPATH=src python -m scorpus.cli corpus-validate \
  ./artifacts/demo_corpus/corpus-index.yaml
```

You should see `PASS`. The validator checks:

- Corpus index integrity
- Materialization manifests present and consistent
- Canonical metadata available for all datasets
- Backend file structure completeness

## Step 7 — Load a few rows

A quick smoke to confirm the corpus loads and canonical metadata is accessible:

```python
from scorpus.loaders import load_corpus

corpus = load_corpus("./artifacts/demo_corpus")

print(f"Datasets: {corpus.dataset_ids}")
print(f"Total cells: {len(corpus.metadata_index)}")
print(f"Global vocab: {corpus.feature_registry.global_vocab_size}")

# Inspect the first few rows
meta = corpus.take_metadata(
    [0, 1, 2],
    columns=["dataset_id", "perturb_label", "condition", "cell_context"],
)
print(meta)
```

## Where outputs go

All generated artifacts live under `./artifacts/`:

| Path | Contents |
|------|----------|
| `artifacts/review/` | Inspection summaries |
| `artifacts/demo_corpus/` | Materialized federated corpus (per-dataset Lance matrices, raw sidecars, canonical metadata) |

The raw demo `.h5ad` inputs under `./demo_data/` are never modified.

## Re-running the demo

To start from scratch:

```bash
rm -rf ./artifacts
```

Then re-run from Step 1. The demo data under `./demo_data/` does not need to be
re-downloaded.

## Next steps

- **[Jupyter Demo](jupyter_demo.md)** — the same workflow with interactive Python cells
- **[Rendered Notebook](demo_walkthrough.ipynb)** — the executed walkthrough with cell outputs
- **[Canonicalization](demo_canonicalization.md)** — understand the two demo schema decisions
- **[pertTF Loading](perttf_loader.md)** — build paired batches from the corpus
- **[Scanpy & RAPIDS](scanpy_rapids.md)** — export AnnData and run downstream analysis
