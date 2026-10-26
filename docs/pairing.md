# Pairing source and target cells

`scorpus.pairing` describes which source cells may be paired with which target cells, and
draws them. It works on any metadata table (pandas or polars), for example
`corpus.metadata_index.df`, and does not use the rest of scorpus. It returns row positions;
reading expression, batching and padding are up to the loader and the model.

Every example below is a test in `tests/test_pairing.py`. A runnable walkthrough on a 27-cell
toy table is in `notebooks/pairing_walkthrough.ipynb`.

## Three steps

1. **Groups.** Choose the columns that define a group of cells.
2. **Pair table.** Build a table with one row per allowed (source group, target group),
   using `cross` and ordinary polars filters and joins. Write only what must hold: a
   column you do not mention is free.
3. **Sampler.** Choose how many cells to take from each side, which pair rows to draw
   together, and how cells are linked.

```python
import polars as pl
from scorpus.pairing import make_groups, cross, combine, describe, Sampler, src, tgt

groups = make_groups(obs, by=["cell_line", "perturbation"])
is_ctrl = pl.col("perturbation") == "ctrl"
pairs = cross(groups, on=["cell_line"], source=is_ctrl, target=~is_ctrl)
print(describe(groups, pairs))

sampler = Sampler(groups, pairs, n_source=1)
sampler.sample(step=0)            # [([1], [6])]
```

`cross(groups, on=..., source=..., target=...)` returns every combination of a source
group and a target group. Columns get the prefix `source_` or `target_`; `src.time` is
short for `pl.col("source_time")`, `tgt.time` for `pl.col("target_time")`.

- `source`, `target`: filters on the group table, applied before combining.
- `on`: columns that must be equal on both sides (missing equals missing). This is a join,
  so it stays small with many groups.

`combine(*tables)` stacks several pair tables (several rules). A pair found in more than
one keeps its first row. Add a label column to each table to tell the rules apart.

## Examples

**Controls to perturbed cells, same cell line.** As above.

**Populations.** `Sampler(groups, pairs, n_source=32)` gives 32 source and 32 target cells.

**Time course with dose, three rules.**

```python
groups = make_groups(obs, by=["perturbation", "time", "dose", "replicate"])
from_origin = cross(groups, on=["replicate"], source=is_ctrl & (pl.col("time") == -1), target=pl.col("time") > -1)
continue_treatment = cross(groups, on=["replicate", "perturbation", "dose"], source=~is_ctrl).filter(tgt.time > src.time)
dose_up = cross(groups, on=["replicate", "perturbation", "time"], source=~is_ctrl).filter(tgt.dose > src.dose)
pairs = combine(
    from_origin.with_columns(rule=pl.lit("from_origin")),
    continue_treatment.with_columns(rule=pl.lit("continue_perturbation")),
    dose_up.with_columns(rule=pl.lit("dose_response")),
)
```

This gives 63, 60 and 30 pairs for the three rules (tested).

**Dose as a cell-level condition.** Keep dose in the groups, draw the rows of one
(perturbation, time) pair together, and link cells one to one:

```python
Sampler(groups, pairs, n_source=4, links="random",
        bundle_by=["rule", "source_perturbation", "source_time", "target_perturbation", "target_time"])
```

**Combinations (A → A+B).**

```python
parts = lambda e: e.str.split("+")
cross(groups, source=singles, target=doubles).filter(
    parts(src.perturbation).list.set_difference(parts(tgt.perturbation)).list.len() == 0)
```

## More cases

**Fallback matching.** Same donor when possible, otherwise the same cell type:

```python
same_donor = cross(groups, on=["donor", "celltype"], source=is_ctrl, target=~is_ctrl)
fallback = cross(groups, on=["celltype"], source=is_ctrl, target=~is_ctrl).join(
    same_donor.select("target_group"), on="target_group", how="anti")
pairs = combine(same_donor.with_columns(rule=pl.lit("same_donor")),
                fallback.with_columns(rule=pl.lit("fallback")))
```

**Next time point only.** Add a group-level column, then filter on it:

```python
groups = groups.with_columns(next_time=pl.col("time").shift(-1).over("perturbation", order_by="time"))
pairs = cross(groups, on=["perturbation"]).filter(tgt.time == src.next_time)
```

**Several pairs in one sample with shared controls (dose series).**

```python
Sampler(groups, pairs, n_source=4, bundle_by=["source_group", "target_perturbation"])
# one sample: [(controls, dose 1 cells), (same controls, dose 5 cells), (same controls, dose 10 cells)]
```

**Population to one cell.** `Sampler(groups, pairs, n_source="all", n_target=1)`.

**Real cell links (lineage) or computed ones (OT).** Pass a table with `source_row` and
`target_row`:

```python
Sampler(groups, pairs, n_source=3, links=parent_child_links)
```

`links_from` computes such a table only inside the allowed pairs, with your own matching
function. `match(source_rows, target_rows)` returns positions `(i, j)`:

```python
ot_links = links_from(groups, pairs, lambda s, t: linear_sum_assignment(cdist(pca[s], pca[t])))
Sampler(groups, pairs, links=ot_links)
```

**Weighted links (for example a soft OT plan).** A `weight` column in the link table makes a
cell pick among its links into the chosen target group in proportion to the weights. `match` can
return `(i, j, w)`; `top_k` keeps a link only if it is among the k heaviest of its source cell
or of its target cell, so every cell keeps links whichever side is covered:

```python
def ot_plan(s, t):                      # one transport plan per allowed pair of groups
    P = ot.sinkhorn(ot.unif(len(s)), ot.unif(len(t)), ot.dist(pca[s], pca[t]), reg=0.05)
    i, j = np.nonzero(P)
    return i, j, P[i, j]                # raw plan mass; the sampler normalizes per cell

ot_links = links_from(groups, pairs, ot_plan, top_k=10)   # columns source_row, target_row, weight
```

A cell with no link into the chosen target group is left out of that sample (or epoch); there
is no fallback.

**Different links per rule (pertTF's default).** Put the link kind in a pair-table column:

```python
pairs = combine(
    cross(groups, on=ctx, source=is_ctrl, target=~is_ctrl).with_columns(link=pl.lit("random")),   # or "table" with link_table=ot_links
    cross(groups, on=[*ctx, "perturbation"], source=~is_ctrl).with_columns(link=pl.lit("self")),  # a perturbed cell is its own target
)
Sampler(groups, pairs, links="link")
```

**Weights.** Any pair-table column: `Sampler(groups, pairs, weight="weight")`.

## Sampler

| Option | Meaning |
|---|---|
| `n_source`, `n_target` | Cells per source / target group in a sample: an integer or `"all"`. `n_target` defaults to `n_source`. |
| `bundle_by` | Pair-table columns. Rows with equal values are drawn together as one sample. Default: each row alone. |
| `links` | `"none"`: sets. `"random"`: each covered cell gets one random partner cell. `"self"`: each cell is its own partner (source group = target group). A table (`source_row`, `target_row`): partners from that table. Or the name of a pair-table column giving one of `"none"`, `"random"`, `"self"`, `"table"` per row. |
| `link_table` | The table used by `"table"` rows (see `links_from`), with an optional `weight` column. |
| `weight` | A pair-table column. A bundle's weight is the sum of its rows. Default: uniform. |
| `drop_small` | Leave out pair rows whose source or target group has fewer than `n_source` / `n_target` cells. Without it, such a group gives all its cells; padding is up to the model. |

Inside one sample, a group used by several rows gives the same cells to all of them.

- `sample(step, seed=0)`: one bundle, reproducible from `(seed, step)`.
- `sample_batch(step, n_bundles, seed=0)`: a whole batch in one call. With single cells and
  no bundles it is fully vectorized.
- `epoch(epoch, seed=0, cover="source", drop_last=False)`: every cell of every covered group
  once. Each group is cut into chunks of `n_source` (or `n_target`) cells; each chunk takes one
  bundle that contains its group, by weight. `drop_last` leaves out each group's short last chunk.
- `epoch_rows(...)`: the cell pairs of `epoch()` as two arrays, source rows and target rows.
  With single cells and no bundles it is fully vectorized, link tables included, which matters
  for tens of millions of cells.

The sampler only returns rows. Batching, padding and what a pair means for training (for example
which perturbation label to use) are up to the model, which reads the metadata of those rows.

Output is a list of (source rows, target rows) tuples: one tuple per pair row for `"none"`
(sets), or one tuple per linked cell for every other link. Inside a bundle each group gives its
own `n_source` / `n_target` cells, so a bundle of two dose rows with cell links yields twice as
many links.

## Without pairs

`cells(groups, epoch=0, seed=0, batch_size=None, same_group=False, drop_last=False)` uses
every cell once per epoch, shuffled, for training on single cells. With `batch_size` it
returns a list of batches; with `same_group=True` every batch holds cells of one group
(for example one dataset and cell type, or a set of cells as one input):

```python
groups = make_groups(metadata, by=["dataset_id", "cell_type"])
for rows in cells(groups, epoch=0, batch_size=256, same_group=True):
    ...
```

Read the rows with `corpus.expression_reader.read_expression_flat(rows.tolist())`, and use
`scorpus.loaders.dense_counts` for dense counts on the corpus gene axis.

Cells filtered out at read time leave gaps between the rows of a group, and each stretch
between gaps becomes a separate, slow read on network storage.
`corpus.expression_reader.max_gap = 16` (Lance; default 0) also reads gaps of up to 16 rows and
drops them in memory, so batches are unchanged, but memory per read grows up to about
1 / (fraction of cells kept). On Marson runs on network storage (8 processes, half the cells
kept) it recovered about half of the filter's slowdown. Use 16 when workers have spare CPU,
4 when they are CPU-bound, and 0 on node-local NVMe or without a filter, where reading the
gaps only costs time; avoid large values with `run_length=1`. Rewriting with the filter
applied, `reorder_corpus(..., write_order(groups, seed, keep=rows), ...)`, removes the gaps
for good.

## Speed (1M cells, 10,000 pairs)

| | time |
|---|---|
| build groups and pair table | 0.22 s |
| one step, size 1 | 30 µs |
| one step, size 64 | 32 µs |
| batch of 64 single-cell pairs (`sample_batch`) | 0.05 ms |
| epoch, size 1 (100k samples) | 0.96 s |
| `epoch_rows`, single cells (1M cells, 18,000 pairs) | 0.16 s |
| `epoch_rows`, single cells with 9M weighted links on 9,000 of the pairs | 0.4 s (0.9 s the first time, building the link index) |
| `links_from` for those 9M links with `top_k=10` | 2.0 s |
