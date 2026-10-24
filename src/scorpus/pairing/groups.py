"""Groups of cells, and pair tables built from them with ordinary polars operations.

A pair table has one row per allowed (source group, target group). It must have the
columns ``source_group`` and ``target_group``; any other column (a rule label, a
weight, the groups' metadata) is free. Build it with ``cross`` plus polars filters
and joins, and write only the conditions that must hold: a column you do not
mention is unconstrained.
"""

from __future__ import annotations

import numpy as np
import polars as pl


class Groups:
    """Cells grouped by metadata columns.

    ``table``: one row per group, with ``group`` (its id), the grouping columns,
    ``n_cells`` and any columns added with ``with_columns``.
    ``rows``: for each group id, the 0-based positions of its cells in the input table.
    ``by``: the grouping columns.
    """

    def __init__(self, table: pl.DataFrame, rows: list[np.ndarray], n_rows: int, by: list[str]):
        self.table = table
        self.rows = rows
        self.n_rows = n_rows
        self.by = by

    def with_columns(self, *exprs, **named) -> Groups:
        """Add group-level columns (for example a time rank) to use in filters and joins."""
        return Groups(self.table.with_columns(*exprs, **named), self.rows, self.n_rows, self.by)


def make_groups(table, by: list[str]) -> Groups:
    """One group per distinct combination of the ``by`` columns; missing values form their own group."""
    frame = table if isinstance(table, pl.DataFrame) else pl.from_pandas(table)
    grouped = frame.with_row_index("__row").group_by(by, maintain_order=True).agg(pl.col("__row"))
    rows = [np.asarray(r, dtype=np.int64) for r in grouped["__row"].to_list()]
    group_table = (
        grouped.drop("__row")
        .with_row_index("group")
        .with_columns(pl.col("group").cast(pl.Int64), n_cells=pl.Series([len(r) for r in rows], dtype=pl.Int64))
    )
    return Groups(group_table, rows, frame.height, list(by))


def write_order(groups: Groups, seed: int) -> tuple[np.ndarray, pl.DataFrame]:
    """A row order for rewriting a table so that every group is one contiguous block, shuffled within.

    Groups are placed in order of their grouping columns (missing values first), and the rows
    inside each group are shuffled with ``seed``. Returns ``order``, the old row position for
    each new position (``new_table = old_table[order]``), and the group table in the new order
    with each group's block ``[start, stop)`` in the new table.

    On a table written in this order, ``Sampler(..., run_length=r)`` reads runs of neighbouring
    rows, and each run is a uniform random subset of its group.
    """
    rng = np.random.default_rng(seed)
    table = groups.table.sort(groups.by, maintain_order=True)
    blocks = [rng.permutation(groups.rows[g]) for g in table["group"].to_list()]
    n_cells = table["n_cells"].to_numpy()
    stop = np.cumsum(n_cells)
    return np.concatenate(blocks), table.with_columns(start=pl.Series(stop - n_cells), stop=pl.Series(stop))


def cross(groups: Groups, *, on: list[str] = (), source=None, target=None) -> pl.DataFrame:
    """Every (source group, target group) combination, as one table.

    The source group's columns get the prefix ``source_``, the target group's ``target_``.
    ``source`` / ``target``: filters on the group table (plain column names), applied before combining.
    ``on``: columns that must be equal on both sides (missing equals missing). These use a join
    instead of the full cross product, which keeps the table small for many groups.
    """
    source_table = groups.table if source is None else groups.table.filter(source)
    target_table = groups.table if target is None else groups.table.filter(target)
    source_table = source_table.select(pl.all().name.prefix("source_"))
    target_table = target_table.select(pl.all().name.prefix("target_"))
    if not on:
        pairs = source_table.join(target_table, how="cross")
    else:
        pairs = source_table.join(
            target_table,
            left_on=[f"source_{c}" for c in on],
            right_on=[f"target_{c}" for c in on],
            how="inner",
            coalesce=False,
            nulls_equal=True,
        )
    return pairs.sort(["source_group", "target_group"])


def combine(*tables: pl.DataFrame) -> pl.DataFrame:
    """Stack pair tables. A (source group, target group) found in several keeps its first row."""
    return pl.concat(tables, how="diagonal_relaxed").unique(
        ["source_group", "target_group"], keep="first", maintain_order=True
    )


def links_from(groups: Groups, pairs: pl.DataFrame, match, top_k: int | None = None) -> pl.DataFrame:
    """Cell links computed inside every allowed (source group, target group), for ``Sampler``.

    ``match(source_rows, target_rows)`` gets the cells of the two groups and returns two
    equal-length position arrays ``(i, j)``, or three with a weight ``(i, j, w)``:
    ``source_rows[i[k]]`` is linked to ``target_rows[j[k]]`` with weight ``w[k]`` (1 if not given).
    For example an optimal transport assignment on PCA coordinates:
    ``lambda s, t: linear_sum_assignment(cdist(pca[s], pca[t]))``, or the entries of a transport
    plan ``P``: ``i, j = np.nonzero(P)`` and ``w = P[i, j]``.

    top_k: keep a link only if it is among the k heaviest links of its source cell or of its
    target cell, so every cell keeps its best links whichever side the sampler covers.
    """
    source_rows, target_rows, weights = [], [], []
    for s, t in pairs.select("source_group", "target_group").iter_rows():
        rows_s, rows_t = groups.rows[s], groups.rows[t]
        i, j, *w = match(rows_s, rows_t)
        i, j = np.asarray(i, dtype=np.int64), np.asarray(j, dtype=np.int64)
        w = np.asarray(w[0], dtype=float) if w else np.ones(len(i))
        if top_k is not None:
            keep = _heaviest(i, w, top_k) | _heaviest(j, w, top_k)
            i, j, w = i[keep], j[keep], w[keep]
        source_rows.append(rows_s[i])
        target_rows.append(rows_t[j])
        weights.append(w)
    return pl.DataFrame({
        "source_row": np.concatenate(source_rows),
        "target_row": np.concatenate(target_rows),
        "weight": np.concatenate(weights),
    })


def _heaviest(cell: np.ndarray, weight: np.ndarray, k: int) -> np.ndarray:
    """True for each link among the k heaviest links of its cell."""
    order = np.lexsort((-weight, cell))
    sorted_cell = cell[order]
    rank = np.arange(len(cell)) - np.searchsorted(sorted_cell, sorted_cell, side="left")
    keep = np.zeros(len(cell), dtype=bool)
    keep[order[rank < k]] = True
    return keep


class _Side:
    """``src.time`` is ``pl.col("source_time")``; ``tgt["cell type"]`` is ``pl.col("target_cell type")``."""

    def __init__(self, prefix: str):
        self._prefix = prefix

    def __getattr__(self, name: str) -> pl.Expr:
        if name.startswith("__"):
            raise AttributeError(name)
        return pl.col(f"{self._prefix}{name}")

    def __getitem__(self, name: str) -> pl.Expr:
        return pl.col(f"{self._prefix}{name}")


src = _Side("source_")
tgt = _Side("target_")


def describe(groups: Groups, pairs: pl.DataFrame, by: str = "rule") -> str:
    """What the pair table allows, to check against what was intended."""
    n_groups = groups.table.height
    lines = [f"cells: {groups.n_rows:,}   groups: {n_groups:,}   pairs: {pairs.height:,}"]
    if by in pairs.columns:
        lines.append(f"{by:<28}{'pairs':>10}{'source groups':>16}{'target groups':>16}")
        stats = pairs.group_by(by, maintain_order=True).agg(
            pl.len(), pl.col("source_group").n_unique(), pl.col("target_group").n_unique()
        )
        for name, n, n_source, n_target in stats.iter_rows():
            lines.append(f"{str(name):<28}{n:>10,}{n_source:>16,}{n_target:>16,}")
    never_source = n_groups - pairs["source_group"].n_unique()
    never_target = n_groups - pairs["target_group"].n_unique()
    lines.append(f"groups never a source: {never_source:,}   groups never a target: {never_target:,}")
    return "\n".join(lines)
