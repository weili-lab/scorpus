# Canonicalization Handbook

Canonicalization turns materialized raw metadata sidecars into loader-ready
canonical obs and var parquet files. It is the main manual review step in the
repo because every perturb-seq source uses different metadata fields.

## Where Canonicalization Fits

```text
inspect
  -> materialize
  -> draft-schema
  -> review and write final-schema.yaml
  -> canonicalize
  -> load_corpus()
```

Materialization preserves raw source truth. Canonicalization adds a reviewed,
consistent view used by runtime loaders. It does not rewrite the sparse count
matrix.

## Inputs and Outputs

For each dataset, canonicalization reads:

- `dataset-summary.yaml`
- `materialization-manifest.yaml`
- `raw-obs.parquet`
- `raw-var.parquet`
- `size-factor.parquet` when present
- `final-schema.yaml`

It writes:

- `canonical_meta/canonical-obs.parquet`
- `canonical_meta/canonical-var.parquet`

`canonicalize` does not write a corpus-level vocabulary YAML as part of the
public CLI path. Runtime feature alignment is reconstructed by `load_corpus()`
from `canonical-var.parquet` files.

## Paths by Topology

Aggregate topology:

```text
corpus/
└── meta/
    └── <dataset_id>/
        ├── raw-obs.parquet
        ├── raw-var.parquet
        ├── size-factor.parquet
        ├── draft-schema.yaml
        ├── final-schema.yaml
        └── canonical_meta/
            ├── canonical-obs.parquet
            └── canonical-var.parquet
```

Federated topology:

```text
corpus/
└── <dataset_id>/
    └── meta/
        ├── raw-obs.parquet
        ├── raw-var.parquet
        ├── size-factor.parquet
        ├── draft-schema.yaml
        ├── final-schema.yaml
        └── canonical_meta/
            ├── canonical-obs.parquet
            └── canonical-var.parquet
```

The CLI resolves these paths from `corpus-index.yaml`, so most users only pass
the corpus root.

## CLI Workflow

Draft schemas after materialization:

```bash
PYTHONPATH=src python -m scorpus.cli draft-schema \
  --corpus /path/to/corpus
```

This writes one `draft-schema.yaml` per uncanonicalized dataset. Existing drafts
are skipped unless `--force-all` is used.

Review each draft and save the approved version as `final-schema.yaml` in the
same dataset metadata directory.

Dry-run canonicalization:

```bash
PYTHONPATH=src python -m scorpus.cli canonicalize \
  --corpus /path/to/corpus \
  --dry-run
```

Canonicalize all datasets that have `final-schema.yaml`:

```bash
PYTHONPATH=src python -m scorpus.cli canonicalize \
  --corpus /path/to/corpus
```

Canonicalize one dataset:

```bash
PYTHONPATH=src python -m scorpus.cli canonicalize \
  --corpus /path/to/corpus \
  --dataset-id my_dataset
```

Bulk canonicalization continues past failures and reports all failed datasets at
the end.

## Draft Schema Versus Final Schema

`draft-schema.yaml` is heuristic. It uses inspection profiles and raw sidecar
field names to produce a starting point.

`final-schema.yaml` is the reviewed schema executed by `canonicalize`.

Recommended review rule: keep the smallest faithful mapping that represents the
dataset. Do not promote high-cardinality or ambiguous raw fields unless a
downstream workflow needs them.

Set `status: ready` in reviewed schemas for human clarity. The current schema
loader accepts both `draft` and `ready`, but the filename `final-schema.yaml` is
what the CLI discovers.

## Schema Structure

Minimal shape:

```yaml
kind: canonicalization-schema
contract_version: 0.3.0
dataset_id: my_dataset
status: ready
description: Reviewed canonicalization schema for my_dataset.

gene_mapping:
  enabled: false
  engine: identity
  source_namespace: gene_symbol
  target_namespace: gene_symbol
  mapping_file: null

obs_column_mappings: []
obs_extensible: []
var_column_mappings: []
var_extensible: []
notes: []
```

The runner validates duplicate canonical names across normal mappings and
extensible columns. It also requires all required canonical obs and var fields
to be produced.

## Required Canonical Obs Fields

Every `canonical-obs.parquet` must contain:

- `assay`
- `batch_id`
- `cell_context`
- `cell_id`
- `cell_line_or_type`
- `condition`
- `dataset_id`
- `dataset_index`
- `disease_state`
- `donor_id`
- `dose`
- `dose_unit`
- `global_row_index`
- `local_row_index`
- `perturb_label`
- `perturb_type`
- `sex`
- `size_factor`
- `species`
- `timepoint`
- `timepoint_unit`
- `tissue`

Typed obs fields:

- `global_row_index`: int64
- `dataset_index`: int32
- `local_row_index`: int64
- `size_factor`: float64

Nullable string obs fields:

- `dose`
- `dose_unit`
- `timepoint`
- `timepoint_unit`

All other required obs fields are written as strings. Missing string values use
the mapping fallback, usually `NA`.

## Required Canonical Var Fields

Every `canonical-var.parquet` must contain:

- `origin_index`
- `gene_id`
- `canonical_gene_id`
- `global_id`

Meanings:

- `origin_index`: dataset-local feature order from the materialized count matrix
- `gene_id`: source gene/feature identifier used as the input to gene mapping
- `canonical_gene_id`: harmonized identifier used for corpus-level feature alignment
- `global_id`: deterministic per-dataset field written during canonicalization

Runtime loaders do not trust `global_id` as the corpus-global feature contract.
`FeatureRegistry` rebuilds corpus-global feature IDs from `canonical_gene_id`.

## Raw Sidecar Loading

`raw-obs.parquet` stores stable top-level fields plus a JSON `raw_fields` column.
During canonicalization, the runner expands `raw_fields` so schema mappings can
refer to the original obs columns by name.

`raw-var.parquet` stores `origin_index`, `feature_id`, and JSON `raw_var`. The
runner expands `raw_var` so schema mappings can refer to original var columns by
name.

`size-factor.parquet` is aligned by `cell_id` when needed. If no size-factor file
is present, `size_factor` defaults to `1.0`.

## Obs Mapping Strategies

### `source-field`

Read one raw obs column and apply transforms.

```yaml
- canonical_name: cell_line_or_type
  strategy: source-field
  source_column: cell_type
  transforms:
    - name: strip_whitespace
      args: {}
```

If the source column is missing, the runner fills the field with `fallback` and
records a warning.

Special case: `size_factor` with `source-field` reads from the size-factor vector
rather than raw obs.

### `literal`

Fill one constant value for every row.

```yaml
- canonical_name: species
  strategy: literal
  literal_value: human
```

### `passthrough`

Copy a raw column. The runner first looks for a raw column named the same as the
canonical field, then uses `source_column` if provided.

```yaml
- canonical_name: batch_id
  strategy: passthrough
  source_column: batch
```

If no column is found, the field is filled with `fallback` and a warning is
recorded.

### `row-index`

Write zero-based row order for the materialized dataset.

```yaml
- canonical_name: local_row_index
  strategy: row-index
```

Use this for `local_row_index`. `global_row_index` should usually come from the
raw obs sidecar because materialization registered corpus-global row ranges.

### `null`

Fill every row with `fallback`.

```yaml
- canonical_name: disease_state
  strategy: null
  fallback: NA
```

### `coalesce`

Pick the first non-null-like value from ordered source columns, then apply
transforms.

```yaml
- canonical_name: perturb_label
  strategy: coalesce
  source_columns: [perturbation, target_gene, guide_id]
  transforms:
    - name: strip_whitespace
      args: {}
    - name: strip_guide_suffix
      args: {}
    - name: map_control_labels
      args:
        candidates: [NTC, non-targeting]
        output: ctrl
```

All listed source columns must exist.

### `join`

Join multiple source columns into one string.

```yaml
- canonical_name: condition
  strategy: join
  source_columns: [treatment, dose]
  separator: "_"
  skip_nulls: true
```

All listed source columns must exist.

### `template`

Render a Python-style format string from raw source columns.

```yaml
- canonical_name: condition
  strategy: template
  template: "{treatment}_{timepoint}h"
  missing_value_behavior: fallback
```

Template fields are discovered from braces and must exist in raw obs.

`missing_value_behavior` can be:

- `fallback`: return the mapping fallback when any template field is null-like
- `empty`: render missing fields as empty strings
- `literal`: use `missing_value` for missing fields

### `conditional`

Evaluate ordered cases and use the first matching result.

```yaml
- canonical_name: perturb_type
  strategy: conditional
  cases:
    - source_column: is_control
      predicate: equals
      value: "true"
      result_literal: control
    - source_column: perturbation
      predicate: not_null
      result_literal: crispr
  default_literal: unknown
```

Supported predicates:

- `equals`
- `in`
- `not_null`

Each case must provide exactly one of `result_literal` or
`result_source_column`. Defaults can use `default_literal` or
`default_source_column`.

## Var Mapping Strategies

The schema allows these var strategies:

- `source-field`
- `literal`
- `passthrough`
- `gene-mapping`
- `auto`
- `null`

The current runner has special handling for the four required var fields by
canonical name:

- `origin_index`: read from raw var `origin_index` or the configured source column
- `gene_id`: read from the configured `source_column` or raw `gene_id`
- `canonical_gene_id`: produced from `gene_id` through the top-level `gene_mapping` block
- `global_id`: assigned deterministically within the dataset from sorted canonical gene IDs

Practical required-var block:

```yaml
var_column_mappings:
  - canonical_name: origin_index
    strategy: passthrough
  - canonical_name: gene_id
    strategy: source-field
    source_column: feature_id
  - canonical_name: canonical_gene_id
    strategy: gene-mapping
  - canonical_name: global_id
    strategy: auto
```

Current caveat: transforms listed on the required `gene_id` mapping are not
applied before gene mapping because the runner collects `gene_id` values before
generic var transform handling. If exact gene cleanup is needed, provide a raw
source column that is already in the desired form, use a mapping file, or
preprocess the source metadata before materialization.

## Extensible Columns

Use `obs_extensible` or `var_extensible` to carry optional raw fields into
canonical parquet files without making them part of the required schema.

```yaml
obs_extensible:
  - raw_source_column: donor_age
    canonical_name: donor_age

var_extensible:
  - raw_source_column: highly_variable
    canonical_name: source_highly_variable
```

If an extensible raw source column is missing, the output column is filled with
`NA` and a warning is recorded.

## Transform Execution

Transforms run after a mapping strategy resolves a value.

For obs mappings:

- `source-field`: transforms apply to the extracted value
- `coalesce`: transforms apply to the selected value
- `join`: transforms apply to the joined value
- `template`: transforms apply to the rendered value
- `conditional`: transforms apply to the selected case/default result

Transforms run in YAML order. Unknown transform names are logged and skipped.
If a transform raises an exception or the final value is null-like, the mapping
uses its `fallback`.

Runtime-dispatched transforms:

- `map_control_labels`
- `strip_whitespace`
- `replace_empty_with_null`
- `strip_prefix`
- `strip_suffix`
- `strip_guide_suffix`
- `regex_sub`
- `normalize_case`
- `map_values`
- `split_on_delimiter`
- `dose_parse`
- `dose_unit`
- `timepoint_parse`
- `timepoint_unit`
- `normalize_time_unit`
- `normalize_dose_unit`
- `strip_ensembl_version`
- `normalize_boolean`

Practical ordering: clean first, then harmonize.

```yaml
transforms:
  - name: strip_whitespace
    args: {}
  - name: strip_guide_suffix
    args: {}
  - name: map_control_labels
    args:
      candidates: [NTC, non-targeting]
      output: ctrl
```

## Gene Mapping

The top-level `gene_mapping` block controls how `gene_id` becomes
`canonical_gene_id`.

Supported engines:

- `identity`: `canonical_gene_id = gene_id`
- `mapping_file`: tab-separated mapping file, first column raw gene ID, second column canonical gene ID
- `gget`: `gget.convert(...)` when `gget` is installed

Identity example:

```yaml
gene_mapping:
  enabled: false
  engine: identity
  source_namespace: gene_symbol
  target_namespace: gene_symbol
  mapping_file: null
```

Mapping-file example:

```yaml
gene_mapping:
  enabled: true
  engine: mapping_file
  source_namespace: gene_symbol
  target_namespace: ensembl_gene_id
  mapping_file: ./artifacts/gene_symbol_to_ensembl.tsv
```

If `gget` is unavailable or conversion fails, the current runner logs a warning
and falls back to identity mapping.

## Runtime Feature Alignment

`load_corpus()` builds `FeatureRegistry` from every dataset's
`canonical-var.parquet`.

The registry:

1. Reads datasets in `corpus-index.yaml` order.
2. Casts `origin_index` to integer and sorts rows by `origin_index`.
3. Requires `origin_index` to be contiguous `0..n_vars-1` after sorting.
4. Walks `canonical_gene_id` values in local feature order.
5. Assigns the next corpus-global feature ID when a `canonical_gene_id` is first seen.
6. Builds each dataset's local-feature-index to global-feature-ID mapping.

This means `origin_index` and `canonical_gene_id` are the two var fields that
most directly affect runtime feature correctness.

## Review Checklist

Before copying a draft to `final-schema.yaml`, check:

- `dataset_id` matches the corpus dataset ID exactly.
- Every required obs field has one mapping.
- Every required var field has one mapping.
- No canonical field appears both in mappings and extensible columns.
- `perturb_label` values make biological sense.
- Control labels are mapped consistently across datasets.
- `perturb_type` separates controls from treated cells when possible.
- `cell_context`, `cell_line_or_type`, `assay`, `species`, and `tissue` are not guessed beyond source evidence.
- `dose`, `dose_unit`, `timepoint`, and `timepoint_unit` are null when unavailable rather than invented.
- `gene_id` source matches the matrix feature order.
- `canonical_gene_id` namespace is consistent across datasets intended to share a feature vocabulary.
- Optional high-cardinality fields are kept as extensible fields only when needed.

## Worked Minimal Schema

```yaml
kind: canonicalization-schema
contract_version: 0.3.0
dataset_id: my_dataset
status: ready
description: Minimal reviewed schema.

gene_mapping:
  enabled: false
  engine: identity
  source_namespace: gene_symbol
  target_namespace: gene_symbol
  mapping_file: null

obs_column_mappings:
  - canonical_name: assay
    strategy: literal
    literal_value: Perturb-seq
  - canonical_name: batch_id
    strategy: source-field
    source_column: batch
  - canonical_name: cell_context
    strategy: source-field
    source_column: cell_type
  - canonical_name: cell_id
    strategy: passthrough
  - canonical_name: cell_line_or_type
    strategy: source-field
    source_column: cell_type
  - canonical_name: condition
    strategy: null
  - canonical_name: dataset_id
    strategy: passthrough
  - canonical_name: dataset_index
    strategy: source-field
    source_column: dataset_index
  - canonical_name: disease_state
    strategy: null
  - canonical_name: donor_id
    strategy: null
  - canonical_name: dose
    strategy: null
  - canonical_name: dose_unit
    strategy: null
  - canonical_name: global_row_index
    strategy: source-field
    source_column: global_row_index
  - canonical_name: local_row_index
    strategy: source-field
    source_column: local_row_index
  - canonical_name: perturb_label
    strategy: coalesce
    source_columns: [perturbation, target_gene]
    transforms:
      - name: strip_whitespace
        args: {}
      - name: strip_guide_suffix
        args: {}
      - name: map_control_labels
        args:
          candidates: [NTC, non-targeting]
          output: ctrl
  - canonical_name: perturb_type
    strategy: conditional
    cases:
      - source_column: perturbation
        predicate: in
        values: [NTC, non-targeting]
        result_literal: control
      - source_column: perturbation
        predicate: not_null
        result_literal: crispr
    default_literal: unknown
  - canonical_name: sex
    strategy: null
  - canonical_name: size_factor
    strategy: source-field
    source_column: size_factor
  - canonical_name: species
    strategy: literal
    literal_value: human
  - canonical_name: timepoint
    strategy: null
  - canonical_name: timepoint_unit
    strategy: null
  - canonical_name: tissue
    strategy: null

obs_extensible: []

var_column_mappings:
  - canonical_name: origin_index
    strategy: passthrough
  - canonical_name: gene_id
    strategy: source-field
    source_column: feature_id
  - canonical_name: canonical_gene_id
    strategy: gene-mapping
  - canonical_name: global_id
    strategy: auto

var_extensible: []
notes: []
```

## Common Failure Modes

Missing required canonical obs fields:

- Add mappings for every field listed in the required obs section.

Missing required canonical var fields:

- Add `origin_index`, `gene_id`, `canonical_gene_id`, and `global_id` mappings.

Duplicate canonical names:

- Remove duplicate mappings or duplicate extensible entries.

Missing raw columns:

- Check `raw-obs.parquet`, `raw-var.parquet`, and `dataset-summary.yaml` for actual field names.
- `coalesce`, `join`, `template`, and `conditional` fail when referenced source columns are missing.

Bad `origin_index` at load time:

- `FeatureRegistry` requires numeric, contiguous `origin_index` after sorting.
- If this fails, check `canonical-var.parquet` and the var mapping for `origin_index`.

Wrong controls:

- Review `control_label_candidates` in `dataset-summary.yaml`.
- Make sure `map_control_labels` candidates match raw values exactly enough for the configured case sensitivity.

Wrong gene namespace:

- Confirm whether `gene_id` is a symbol, Ensembl ID, guide ID, or other feature ID.
- Keep `canonical_gene_id` consistent across datasets that should share model tokens.

Loader cannot find canonical parquet files:

- Run `canonicalize` after writing `final-schema.yaml`.
- Check topology-specific paths under `meta/<dataset_id>/canonical_meta` or `<dataset_id>/meta/canonical_meta`.
