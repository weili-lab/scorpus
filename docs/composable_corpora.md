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

Conversion adds one obs column, `scorpus_total_counts`: each cell's sum of
stored counts (float64). It stops if the source obs already has a column of
that name. No size factor is stored; compute one from `scorpus_total_counts`
when needed. Older corpora have no such column.

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
and values, and adds no `scorpus_total_counts` column.

Recovery uses `expm1(x) / min_nonzero_expm1_per_cell`. It assumes the smallest
nonzero value corresponds to one molecule; integer checks do **not** prove
recovery of the original absolute counts. The manifest distinguishes original
counts, recovered counts, and non-count expression.

Only the selected matrix and its obs/var are converted. Other layers, obsm,
obsp, and uns are not copied unless named as extra matrices (below). Existing
outputs are never overwritten. A failed conversion can leave an incomplete
directory, but no completed `corpus.yaml`; inspect that directory and choose a
fresh output for a retry.

### Fractional counts: `round_counts`

```python
from_h5ad("kallisto.h5ad", "rounded.corpus", layers="X", round_counts=True)
```

Some pipelines store fractional count estimates (e.g. kallisto `--mm`, which
splits multi-mapping reads between genes). `round_counts=True` rounds every
value to the nearest integer, with halves rounded to even (`numpy.rint`, so the
many 0.5 splits are not all rounded up), and
then applies the usual nonnegative-count checks. The manifest records
`expression_kind: rounded_counts` and `fractional_value_share`, the share of
stored nonzero values that were fractional. It needs an explicit `layers=`.

### Keeping some features: `features`

```python
from_h5ad("cite.h5ad", "rna.corpus", layers="X", features=("modality", "RNA"))
```

`features=(var_column, value)` keeps only the features whose var column equals
the value, e.g. to drop normalised antibody (CITE) columns stored next to RNA
counts. Both the count checks and the stored var use only the kept features.
It needs an explicit `layers=`.

### Guide and other per-cell matrices: `extra`

```python
names = list(h5py.File("screen.h5ad")["uns/guide_names"][:].astype(str))
c = from_h5ad("screen.h5ad", "screen.corpus",
              extra={"guide_umi": ("obsm/guide_umi", names)})
guide_umi, guides = c.read_extra("guide_umi", rows=[10, 3, 7])
```

Each entry names an h5 path (sparse or dense, one row per cell) and its feature
names. It is stored as `extra/<name>.lance` in the same row order as the
expression, with `extra/<name>_features.parquet` indexed by feature name.
Values must be nonnegative integer counts. `Corpus.read_extra(name, rows)`
returns a CSR matrix of those rows, in the given order, and the feature table.
Extra matrices belong to a standalone corpus; `concat` does not use them.

### CSC matrices

Reading rows out of a CSC matrix touches every column, so a CSC source (select
it with `layers=`) is first regrouped by rows: one pass over blocks of genes
appends each nonzero to a temporary file per `chunk_rows` rows, and each file
then becomes one chunk. Temporary files take about 12 bytes per nonzero, live
in a `csc_row_buckets` folder under the required `csc_tmp_dir=`
and are removed at the end.

## Many CellRanger files: `from_10x_h5_files`

```python
from scorpus import from_10x_h5_files

cells = pd.read_parquet("cells.parquet")   # one row per cell: file_id, local_row, ...
files = {"lane1": "lane1.h5", "lane2": "lane2.h5", "rescued": "rescued.h5ad"}
genes = pd.read_parquet("genes.parquet").set_index("gene_id")
guides = pd.read_parquet("guide_features.parquet").set_index("sgrna_id")
c = from_10x_h5_files(files, cells, "screen.corpus", genes=genes, guides=guides)
```

Cells are written in the order of `cells`, which becomes obs unchanged; each
row points to a cell by `file_id` (a key of `files`) and `local_row` (its row
in that file). Every file's "Gene Expression" features are matched to the
index of `genes` by id, so files that list genes in different orders still
land on one gene axis; a file whose gene ids are not exactly that set is an
error. "CRISPR Guide Capture" features go the same way to the `guide_umi`
extra matrix. h5ad files (counts in `X`, gene ids in `var["gene_ids"]`) have
no guide features and get empty guide rows.

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
- `combined.metadata_index` also contains internal routing coordinates. These
  do not replace original columns in member obs.

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
