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
| `link_table` | The table used by `"table"` rows (see `links_from`). |
| `weight` | A pair-table column. A bundle's weight is the sum of its rows. Default: uniform. |
| `drop_small` | Leave out pair rows whose source or target group has fewer than `n_source` / `n_target` cells. Without it, such a group gives all its cells; padding is up to the model. |

Inside one sample, a group used by several rows gives the same cells to all of them.

- `sample(step, seed=0)`: one bundle, reproducible from `(seed, step)`.
- `sample_batch(step, n_bundles, seed=0)`: a whole batch in one call. With single cells and
  no bundles it is fully vectorized.
- `epoch(epoch, seed=0, cover="source", drop_last=False)`: every cell of every covered group
  once. Each group is cut into chunks of `n_source` (or `n_target`) cells; each chunk takes one
  bundle that contains its group, by weight. `drop_last` leaves out each group's short last chunk.

The sampler only returns rows. Batching, padding and what a pair means for training (for example
which perturbation label to use) are up to the model, which reads the metadata of those rows.

Output is a list of (source rows, target rows) tuples: one tuple per pair row for `"none"`
(sets), or one tuple per linked cell for every other link. Inside a bundle each group gives its
own `n_source` / `n_target` cells, so a bundle of two dose rows with cell links yields twice as
many links.

## Speed (1M cells, 10,000 pairs)

| | time |
|---|---|
| build groups and pair table | 0.22 s |
| one step, size 1 | 30 µs |
| one step, size 64 | 32 µs |
| batch of 64 single-cell pairs (`sample_batch`) | 0.05 ms |
| epoch, size 1 (100k samples) | 0.96 s |
