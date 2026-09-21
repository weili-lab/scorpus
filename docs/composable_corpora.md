# Convert independently, compose in memory

The simple route does not require inspection files, canonical column names, or
schema YAML. The curated inspect/materialize/canonicalize route remains usable
for existing corpus releases. Both routes return the same `Corpus` and reuse
the same sparse expression readers.

## One h5ad

```python
from scorpus import from_h5ad, load_corpus, concat

a = from_h5ad("a.h5ad", output="a.corpus")
a = load_corpus("a.corpus")
print(a.obs)  # original columns, index, and categorical/numeric metadata
print(a.var)  # metadata for the selected matrix's feature axis
```

CLI equivalent:

```bash
python -m scorpus.cli convert --source a.h5ad --output a.corpus
```

Conversion currently writes **standalone Lance corpora**. Existing Lance/Zarr
curated corpora remain readable. Conversion opens matrices through h5py and
AnnData's backed sparse reader, including layers; it does not eagerly load all
AnnData layers. Run conversion of large h5ad files on Slurm using the project
execution environment.

The selection policy is:

1. Sample X, raw.X, and layers for finite, nonnegative integer-like counts.
2. Prefer a direct source over a recovery source. Among direct sources, choose
   the largest feature axis; ties prefer counts-named layers, then raw.X, X,
   then other layers.
3. When recovery appears possible, stop with
   `raw count possible, please convert with attempt_conversion=True`.
4. Otherwise stop with
   `no raw count found, if proceeding, convert with layers="raw|X|layer_name" and no_count=True`.
5. Validate every chunk during conversion. A sampled pass is not a guarantee
   that the complete matrix will pass.

Explicit options:

```python
recovered = from_h5ad("log_normalized.h5ad", "recovered.corpus", attempt_conversion=True)
selected = from_h5ad("a.h5ad", "counts.corpus", layers="counts")
non_count = from_h5ad("normalized.h5ad", "normalized.corpus", layers="X", no_count=True)
```

CLI flags are `--attempt-conversion`, `--layers`, and `--no-count`.
`layers` accepts one selection: `raw` means raw.X, `X` means X, otherwise it is
a layer name. The explicit non-count route preserves the selected numeric dtype
and values, and does not compute count size factors.

Recovery uses `expm1(x) / min_nonzero_expm1_per_cell`. It assumes the smallest
nonzero value corresponds to one molecule; integer checks do **not** prove
recovery of the original absolute counts. The manifest distinguishes original
counts, recovered counts, and non-count expression.

Only the selected matrix and its obs/var are converted. Other layers, obsm,
obsp, and uns are not copied. Existing outputs are never overwritten. A failed
conversion can leave an incomplete directory, but no completed `corpus.yaml`;
inspect that directory and choose a fresh output for a retry.

## AnnData access and export

```python
adata = a.to_anndata_lazy(chunk_rows=4096)  # disk-reading Dask X; obs/var in RAM
small = a.to_anndata(global_row_indices=[2, 0, 5])  # eager, selected rows

a.write_h5ad("export.h5ad", chunk_rows=4096)  # writes a new physical copy
import anndata as ad
backed = ad.read_h5ad("export.h5ad", backed="r")
# ... use backed ...
backed.file.close()
```

The lazy object is not a native `isbacked=True` AnnData. Explicit export uses
AnnData's Dask writer, then the new file can be opened in native backed mode.

## Temporary federation and collective metadata

```python
b = load_corpus("b.corpus")

# Ordinary pandas transformations; no new expression conversion or schema DSL.
a_obs = a.obs.copy()
b_obs = b.obs.copy()
a_obs["clean_target"] = a_obs["target"].replace({"NTC": "ctrl"})
b_obs["clean_target"] = b_obs["genotype"].replace({"WT": "ctrl"})

combined = concat(
    {"a": a, "b": b},
    obs={"a": a_obs, "b": b_obs},
    obs_columns={
        "perturbation": {"a": "clean_target", "b": "clean_target"},
        "cell_type": {"a": "celltype", "b": "annotation"},
        "batch": {"a": "batch", "b": "batch"},
    },
    feature_key={"a": "ensembl_id", "b": "gene_id"},
    var_columns={"symbol": {"a": "symbol", "b": "gene_symbol"}},
)
```

Each input currently contains one dataset. Member names are view-local aliases;
the view holds references to the existing readers and never copies expression.
Prepared obs/var tables must retain the exact original indices and row order.
Pass prepared feature tables using `var={"a": a_var, "b": b_var}` similarly.

- `combined.obs` contains only the chosen columns, indexed by `(member, row)`.
- `combined.var` contains the shared feature axis and requested annotations.
- `combined.members["a"].obs` retains the original member metadata.
- `combined.feature_presence` records which member measured each feature.
- `combined.metadata_index` also contains internal routing coordinates and
  count size factors. These do not replace original columns in member obs.

Without `feature_key`, `concat` connects member-local axes but does not claim
that similarly named genes are biologically equivalent. Shared-axis export and
model training require explicit alignment. Use `"index"` as a feature key to
align on var_names. Mapped IDs must be nonempty, unique within each dataset,
and already use the same identifier namespace. Conflicting collective feature
annotations and incompatible collective obs types raise errors.

```python
adata = combined.to_anndata_lazy(var_join="outer")
# Alternatives: "inner" for intersection; "exact" (default) for identical axes.
```

Outer export fills unavailable genes with matrix zeros but preserves their
availability in `adata.varm["feature_presence"]`, with dataset names in
`adata.uns["feature_presence_members"]`. Consumers must not interpret those
unavailable positions as measured zeros.

Metadata is materialized in memory; temporary federation avoids expression
copies, not all metadata allocation. Views are intentionally temporary and are
recreated from the explicit Python mappings for another run.

## Model boundary

pertTF-specific code now belongs to pertTF. Import its paired adapter from
`perttf.model.corpus_adapter`, or use
`perttf.model.corpus_data.produce_corpus_datasets` for the complete training data
dictionary. There are no pertTF adapter exports in scorpus.
