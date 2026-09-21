# Demo Canonicalization

This page keeps the demo canonicalization step short.

The demo does **not** ask a new user to read the full schema contract first.
Instead, it gives a reviewed final schema for each demo dataset plus a short
decision sheet explaining the biological choices behind that schema.

## Demo files

- Reviewed executable schemas:
  - `examples/demo_canonicalization/marson_d2_rest.final-schema.yaml`
  - `examples/demo_canonicalization/xorion_hct116_dual_guide.final-schema.yaml`
- Short decision sheets:
  - `examples/demo_canonicalization/marson_d2_rest.schema-hints.yaml`
  - `examples/demo_canonicalization/xorion_hct116_dual_guide.schema-hints.yaml`
- Copy helper:
  - `scripts/install_demo_schemas.py`

The executable truth is the `final-schema.yaml` files. The `schema-hints.yaml`
files are the short reading layer for the demo.

## Workflow

After materializing the demo corpus, copy in the reviewed schemas and validate
before writing canonical metadata:

```bash
PYTHONPATH=src python scripts/install_demo_schemas.py --corpus ./artifacts/demo_corpus

PYTHONPATH=src python -m scorpus.cli canonicalize \
  --corpus ./artifacts/demo_corpus \
  --dry-run

PYTHONPATH=src python -m scorpus.cli canonicalize \
  --corpus ./artifacts/demo_corpus
```

These demo schemas assume the tutorial order is:

1. create corpus with `marson_d2_rest`
2. append `xorion_hct116_dual_guide`

That is why the reviewed demo schemas fix `dataset_index` to `0` and `1`.
If you reverse the order, update those two literals before canonicalizing.

## Marson demo decisions

| Canonical field | Demo decision | Why |
|---|---|---|
| `batch_id` | `lane_id` | sequencing lane is the clearest batch unit in the subset |
| `condition` | `coalesce(perturbed_gene_name, guide_group)` with `NTC/no sgRNA -> ctrl` | keeps treated gene names while collapsing controls to one label |
| `perturb_label` | `perturbed_gene_name` with `NTC* -> ctrl` | simple gene-level label for the demo |
| `perturb_type` | `guide_type: targeting -> crispri`, `non-targeting -> control` | matches the CRISPRi design and separates controls cleanly |
| `cell_context` | literal `CD4+ T cells` | fixed biological context for this subset |
| `gene_id` | `gene_name` | reviewed Marson schema already uses gene symbols safely here |
| demo extras | keep `guide_id`, `lane_id`, QC columns | enough raw provenance for the tutorial without extra clutter |

Notes:

- The demo subset only sampled `NTC` control rows, but the schema still keeps
  `no sgRNA -> ctrl` for compatibility with the reviewed source schema.
- Native Marson features stay untouched; canonicalization only changes metadata.

## Xorion demo decisions

| Canonical field | Demo decision | Why |
|---|---|---|
| `batch_id` | `sample` | sample already tracks the Xorion batch identity |
| `condition` | `gene_target` with `Non-Targeting -> ctrl` | simple biological condition label |
| `perturb_label` | `guide_target` with all-non-targeting guide pairs collapsed to `ctrl` | keeps guide-pair detail for treated rows but makes controls easy to read |
| `perturb_type` | `gene_target == Non-Targeting -> control`, otherwise `genetic` | keeps the demo conservative while separating controls |
| `cell_context` | literal `HCT116` | fixed cell context for the subset |
| `gene_id` | `feature_id` | the raw Xorion feature axis lives in `var.index`, which materialization preserves as `feature_id` |
| demo extras | keep `guide_target`, `gene_target`, `sample`, QC columns | enough context to understand dual-guide rows |

Notes:

- Xorion raw `feature_id` values are mixed: some look like gene symbols and
  some look like Ensembl-like IDs. The demo keeps that raw namespace instead of
  pretending it has already been harmonized.
- Control normalization is tutorial-specific: only rows where **both** guides
  are non-targeting collapse to `ctrl`.

## What this demo does not claim

- It does **not** claim that arbitrary new datasets can be auto-reviewed.
- It does **not** replace the full handbook in `docs/canonicalization_handbook.md`.
- It does **not** harmonize Marson and Xorion onto one shared gene panel.

Use this page when teaching the demo. Use the handbook when editing or extending
canonicalization rules more broadly.
