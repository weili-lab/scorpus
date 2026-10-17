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
    """

    def __init__(self, table: pl.DataFrame, rows: list[np.ndarray], n_rows: int):
        self.table = table
        self.rows = rows
        self.n_rows = n_rows

    def with_columns(self, *exprs, **named) -> Groups:
        """Add group-level columns (for example a time rank) to use in filters and joins."""
        return Groups(self.table.with_columns(*exprs, **named), self.rows, self.n_rows)


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
    return Groups(group_table, rows, frame.height)


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


def links_from(groups: Groups, pairs: pl.DataFrame, match) -> pl.DataFrame:
    """Cell links computed inside every allowed (source group, target group), for ``Sampler``.

    ``match(source_rows, target_rows)`` gets the cells of the two groups and returns two
    equal-length position arrays ``(i, j)``: ``source_rows[i[k]]`` is linked to ``target_rows[j[k]]``.
    For example an optimal transport assignment on PCA coordinates:
    ``lambda s, t: linear_sum_assignment(cdist(pca[s], pca[t]))``.
    """
    source_rows, target_rows = [], []
    for s, t in pairs.select("source_group", "target_group").iter_rows():
        rows_s, rows_t = groups.rows[s], groups.rows[t]
        i, j = match(rows_s, rows_t)
        source_rows.append(rows_s[np.asarray(i, dtype=np.int64)])
        target_rows.append(rows_t[np.asarray(j, dtype=np.int64)])
    return pl.DataFrame({"source_row": np.concatenate(source_rows), "target_row": np.concatenate(target_rows)})


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
