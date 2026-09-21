# scorpus Design

This document describes how the current repository works. It is not a future
plan and it does not document removed backend experiments.

`scorpus` has one main job: turn single-cell `.h5ad` files into a corpus that
can be loaded consistently for model training or analysis.

There are two routes to a corpus. **Conversion** is the default and covers most
use: one call, no schema. **Curation** is an optional reviewed pipeline for
publishing corpora with audited, harmonized metadata.

## Core Ideas

- A **dataset** is one source `.h5ad` plus its materialized matrix, raw metadata sidecars, canonical metadata, and manifest.
- A **corpus** is a collection of datasets registered in one `corpus-index.yaml`.
- A **backend** is the physical matrix format. Current maintained choices are `lance` and `zarr`.
- A **topology** is how datasets are organized. Current choices are `aggregate` and `federated`.
- **Raw sidecars** preserve what the source dataset contained.
- **Canonical sidecars** are reviewed, loader-ready views of obs/var metadata.
- **Heavy expression data** stays sparse and stores dataset-local feature indices.
- **Feature alignment** happens at runtime from `canonical-var.parquet`, not by rewriting materialized expression rows.

Backend and topology are separate axes:

```text
backend:  lance | zarr
topology: aggregate | federated
```

Aggregate topology stores matrix data under one corpus-level `matrix/` root and
metadata under `meta/<dataset_id>/`. Federated topology stores each dataset as a
self-contained `<dataset_id>/meta` and `<dataset_id>/matrix` directory.

## Data Flow

The default route is direct conversion:

```text
source h5ad
  -> from_h5ad()            standalone conversion, no schema required
  -> concat()               optional in-memory composition
  -> load_corpus()
  -> model loader or AnnData handoff
```

The curated route adds reviewed metadata for corpus releases:

```text
source h5ad
  -> inspection
  -> materialization
  -> canonical schema review
  -> canonicalization
  -> load_corpus()
  -> downstream loaders or analysis helpers
```

Both routes return the same `Corpus` object and share the same sparse
expression readers. `load_corpus()` dispatches on which manifest it finds:
`corpus.yaml` for a converted corpus, `corpus-index.yaml` for a curated one.

The user-facing how-to docs are split by task:

- Conversion and composition: `docs/composable_corpora.md`
- AnnData/Scanpy/RAPIDS handoff and corpus-native pp helpers: `docs/anndata_scanpy_handoff.md`
- Backend policy: `docs/backend_note.md`
- Inspection and materialization (curated route): `docs/inspect_materialize.md`
- Canonical schema review and canonicalization (curated route): `docs/canonicalization_handbook.md`
- pertTF paired loading: `docs/perttf_loader.md` and the pertTF worktree's tutorials

## 1. Inspection

Inspection is the metadata-first pass over a source `.h5ad` file.

It reads the file in backed mode, profiles obs/var metadata, samples matrix
sources, and writes a `dataset-summary.yaml`. It does not materialize the full
matrix and it does not decide the final canonical schema.

Inspection records:

- dataset shape and source identity
- obs and var field names, dtypes, null counts, and example values
- matrix candidates from `.X`, `.raw.X`, and named layers
- sampled integer/count-like behavior for each candidate
- selected count source and whether recovery is needed
- materialization readiness: `pass`, `needs-review`, or `fail`
- likely control-label candidates for later schema review

The materializer uses `dataset-summary.yaml` as the gate. A dataset whose
`materialization_readiness` is not `pass` is rejected.

See `docs/inspect_materialize.md` for CLI and Python examples.

## 2. Materialization

Materialization turns one inspected dataset into corpus artifacts.

The materializer:

- opens the source `.h5ad` in backed mode
- selects the inspected count source
- streams the matrix in row chunks
- writes sparse count data through the selected backend and topology
- writes raw obs and raw var sidecars
- writes size factors and per-dataset HVG rankings
- writes a per-dataset `materialization-manifest.yaml`
- registers the dataset in `corpus-index.yaml`

Materialization is count-first and schema-independent. It does not need a
finalized canonical metadata schema, and it does not rewrite expression features
into a shared global feature space.

Important materialization artifacts:

- `corpus-index.yaml`: corpus membership, dataset order, backend, topology, and global row ranges
- `global-metadata.yaml`: corpus-level metadata mirror used by tools and humans
- `materialization-manifest.yaml`: per-dataset source, output, count-source, and QA metadata
- `raw-obs.parquet`: raw obs metadata plus stable row identity fields
- `raw-var.parquet`: raw var metadata plus `origin_index`
- `size-factor.parquet`: per-cell size factors
- `hvg.parquet`: per-dataset HVG ranking table keyed by `origin_index`

Materialization does not apply cell filtering. If a dataset should be filtered,
create a pre-filtered `.h5ad` first and inspect/materialize that file.

See `docs/inspect_materialize.md` for the command-line and Python APIs.

## 3. Canonical Schema Review

Canonicalization is the most manual part of the workflow because raw dataset
metadata varies across sources.

After materialization, `draft-schema` reads the materialized raw sidecars and
inspection hints and writes `draft-schema.yaml`. The draft is only a starting
point. A user or agent should review it, edit mappings, and save the approved
schema as `final-schema.yaml`.

The reviewed schema decides:

- how perturbation labels are formed
- which labels are controls
- how cell context, assay, tissue, species, donor, batch, dose, and timepoint are represented
- which optional raw fields are carried through as canonical extras
- how raw feature identifiers become `gene_id` and `canonical_gene_id`

See `docs/canonicalization_handbook.md` for the exact schema structure and
review checklist.

## 4. Canonicalization

Canonicalization applies one `final-schema.yaml` per dataset.

It reads:

- `raw-obs.parquet`
- `raw-var.parquet`
- `size-factor.parquet` when present
- `final-schema.yaml`

It writes:

- `canonical_meta/canonical-obs.parquet`
- `canonical_meta/canonical-var.parquet`

`canonical-obs.parquet` contains required loader metadata such as
`dataset_id`, `dataset_index`, `global_row_index`, `local_row_index`,
`perturb_label`, `cell_context`, and `size_factor`.

`canonical-var.parquet` contains at least:

- `origin_index`: dataset-local feature order
- `gene_id`: raw or lightly cleaned source feature identifier
- `canonical_gene_id`: harmonized identifier used for runtime alignment
- `global_id`: per-dataset deterministic field written by canonicalization

Runtime global feature IDs are assigned by `FeatureRegistry` when a corpus is
loaded. The registry walks datasets in `corpus-index.yaml` order, sorts each
`canonical-var.parquet` by numeric `origin_index`, and appends first-seen
`canonical_gene_id` values to a corpus-global vocabulary.

## 5. Runtime Loading

`load_corpus(path)` reconstructs a corpus from the index, canonical metadata,
feature metadata, and backend matrix artifacts.

It builds:

- an expression reader for aggregate or federated Lance/Zarr
- a `MetadataIndex` from `canonical-obs.parquet`
- a `FeatureRegistry` from `canonical-var.parquet` and optional `hvg.parquet`
- a `Corpus` object that exposes public loading and inspection methods

Common runtime calls:

```python
from scorpus.loaders import load_corpus

corpus = load_corpus("/path/to/corpus")
expr = corpus.expression_reader.read_expression_flat([0, 1, 2])
meta = corpus.take_metadata([0, 1, 2], columns=["dataset_id", "perturb_label"])
```

Curated corpora require canonical obs/var parquet files. A standalone corpus
created by direct conversion retains original obs/var metadata and is suitable
for explicit runtime composition, export, or a model-specific adapter after
the required metadata mapping is supplied.

## 6. Downstream Paths

The repo has three main downstream usage paths.

The generic sparse loader uses `build_loader(...)` and returns sparse batches
with dataset-aware feature mapping.

The pertTF-owned adapter uses `PertTFPairedBatchLoader` to form source/target
paired batches with configurable labels, control definitions, row pools, and
pairing groups. See `docs/perttf_loader.md` and the pertTF worktree's tutorials.

The AnnData/Scanpy/RAPIDS path uses `corpus.to_anndata(...)` for eager
counts-only export of whole selected dataset(s), or `corpus.to_anndata_lazy(...)`
for Dask-backed `X` over whole selected dataset(s). Multi-dataset handoff is
allowed only when the selected datasets share the same ordered feature axis.
After export, Scanpy/RAPIDS should own normalization, log transforms, PCA,
neighbors, UMAP, clustering, plotting, and exploratory differential expression.
Selected full-corpus cell-level results can be joined back into the loaded corpus
with `corpus.add_obs_meta(...)`. Corpus-native `pp` helpers remain for durable
HVG rankings, streamed stats, QA/debug checks, and bounded-memory fallbacks. See
`docs/anndata_scanpy_handoff.md`.

## Corpus Management Helpers

`corpus-validate` checks that a corpus index and registered manifests are
consistent.

`corpus-compose` creates a new federated corpus from selected whole-dataset
directories. It supports symlink or copy mode and requires all selected inputs
to use the same backend.

`recalc-hvg` recomputes per-dataset `hvg.parquet` rankings from an existing
corpus and can update manifests.

`corpus-gc` removes unregistered per-dataset directories after failed corpus
writes. It is intentionally conservative and does not clean registered backend
internals.

## Design Boundaries

- Materialization preserves sparse counts and raw metadata; it does not solve cross-dataset metadata harmonization.
- Canonicalization adds reviewed metadata; it does not mutate materialized expression rows.
- The curated loader requires canonical metadata; direct conversion does not infer a final biological schema.
- Feature identity is dataset-local at materialization time and corpus-global at load time.
- The repo does not aim to replace Scanpy/RAPIDS for full exploratory preprocessing.
- Current mainline docs describe Lance/Zarr only; direct conversion currently writes standalone Lance corpora.
