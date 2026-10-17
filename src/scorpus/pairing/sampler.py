"""Sample cells from a pair table.

A sample is a list of (source rows, target rows) tuples, rows being 0-based positions
in the input table:

- link "none":   one tuple per pair row in the bundle, with n_source and n_target cells
- link "random", "self" or "table": one tuple per linked cell, ([i], [j]); with "table", a cell
  that has no link into the partner group gives no tuple

A bundle is the set of pair rows drawn together (``bundle_by``); without ``bundle_by``
every pair row is its own bundle. Inside one sample, a group used by several pair rows
contributes the same cells to all of them, so a bundle like "one control group -> doses
1, 5 and 10" shares its control cells.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .groups import Groups

Sample = list[tuple[list[int], list[int]]]
LINK_KINDS = ("none", "random", "self", "table")


class Sampler:
    def __init__(self, groups: Groups, pairs: pl.DataFrame, *, n_source=1, n_target=None,
                 bundle_by: list[str] | None = None, links="none", link_table=None,
                 weight: str | None = None, drop_small: bool = False):
        """
        n_source, n_target: cells per source / target group in a sample; an integer or "all".
            n_target defaults to n_source.
        bundle_by: pair-table columns; pair rows with equal values are drawn together.
        links: how the cells of a pair row are matched, the same for every row:
            "none" (two sets), "random" (each covered cell gets one random partner cell),
            "self" (each cell is its own partner; source and target group must be the same),
            or a link table (see link_table).
            Or the name of a pair-table column holding one of "none", "random", "self", "table"
            per row.
        link_table: columns ``source_row`` and ``target_row`` (for example lineage or OT links,
            see ``links_from``), used by rows whose link is "table". An optional ``weight``
            column makes a cell pick among its links into the partner group by weight; uniform
            otherwise.
        weight: a pair-table column; a bundle's weight is the sum over its rows. Uniform if None.
        drop_small: leave out pair rows whose source group has fewer than n_source cells or whose
            target group has fewer than n_target cells. Otherwise such a group gives all its cells
            and the model decides whether to pad.
        """
        self.groups = groups
        self.n_source = n_source
        self.n_target = n_source if n_target is None else n_target
        if not isinstance(links, str):
            links, link_table = "table", links

        if drop_small:
            group_size = np.array([len(r) for r in groups.rows])
            keep = np.ones(pairs.height, dtype=bool)
            if self.n_source != "all":
                keep &= group_size[pairs["source_group"].to_numpy()] >= self.n_source
            if self.n_target != "all":
                keep &= group_size[pairs["target_group"].to_numpy()] >= self.n_target
            pairs = pairs.filter(pl.Series(keep))

        self.pair_source = pairs["source_group"].to_numpy().astype(np.int64)
        self.pair_target = pairs["target_group"].to_numpy().astype(np.int64)
        if links in LINK_KINDS:
            self.pair_link = np.full(pairs.height, links)
        else:
            self.pair_link = pairs[links].to_numpy().astype(str)
        unknown = set(self.pair_link.tolist()) - set(LINK_KINDS)
        if unknown:
            raise ValueError(f"unknown link kind(s) {sorted(unknown)}; use one of {LINK_KINDS}")
        # A "self" row between different groups would silently pair cells outside the target group.
        if np.any((self.pair_link == "self") & (self.pair_source != self.pair_target)):
            raise ValueError('"self" links need the same source and target group')

        if bundle_by is None:
            self.bundle_pairs = [np.array([p]) for p in range(pairs.height)]
        else:
            bundles = pairs.with_row_index("__pair").group_by(bundle_by, maintain_order=True).agg(pl.col("__pair"))
            self.bundle_pairs = [np.asarray(b, dtype=np.int64) for b in bundles["__pair"].to_list()]
        self.single_pair_bundles = bundle_by is None

        pair_weight = np.ones(pairs.height) if weight is None else pairs[weight].to_numpy().astype(float)
        self.bundle_weight = np.array([pair_weight[b].sum() for b in self.bundle_pairs])
        # Cumulative weights computed once; each draw is then one binary search.
        self._cdf = np.cumsum(self.bundle_weight) / self.bundle_weight.sum()

        # All group rows in one flat array, so a random cell of group g is flat[start[g] + k], k < count[g].
        self._count = np.array([len(r) for r in groups.rows], dtype=np.int64)
        self._start = np.concatenate([[0], np.cumsum(self._count)[:-1]])
        self._flat = np.concatenate(groups.rows)
        self._group_of_row = np.empty(groups.n_rows, dtype=np.int64)
        for g, rows in enumerate(groups.rows):
            self._group_of_row[rows] = g

        self._is_self, self._is_table = self.pair_link == "self", self.pair_link == "table"
        self.uses_link_table = bool(np.any(self._is_table))
        # One cell per side and no bundles: sampling can be done with array operations.
        self.single_cells = self.single_pair_bundles and self.n_source == 1 and self.n_target == 1
        if self.uses_link_table:
            link_table = link_table if isinstance(link_table, pl.DataFrame) else pl.from_pandas(link_table)
            self._link_table = link_table
            self._link_index = {}  # covered side -> _Links, built when first needed

    @property
    def n_bundles(self) -> int:
        return len(self.bundle_pairs)

    # ---- step mode -------------------------------------------------------------------------

    def sample(self, step: int, seed: int = 0) -> Sample:
        """One bundle, drawn by weight with replacement; reproducible from (seed, step)."""
        rng = np.random.default_rng([seed, step])
        return self._realize(self._draw_bundles(rng, 1)[0], rng, cover="source")

    def sample_batch(self, step: int, n_bundles: int, seed: int = 0) -> list[Sample]:
        """``n_bundles`` bundles for one batch, drawn in one call; reproducible from (seed, step)."""
        rng = np.random.default_rng([seed, step])
        bundles = self._draw_bundles(rng, n_bundles)
        if self.single_cells:
            # Single cells on both sides: pick every source and target cell with one array operation each.
            s, t = self.pair_source[bundles], self.pair_target[bundles]
            source_rows = self._flat[self._start[s] + rng.integers(self._count[s])]
            target_rows = self._partners(source_rows, bundles, "source", rng)
            return [[([int(i)], [int(j)])] if j >= 0 else [] for i, j in zip(source_rows, target_rows)]
        return [self._realize(b, rng, cover="source") for b in bundles]

    def _draw_bundles(self, rng, n: int) -> np.ndarray:
        drawn = np.searchsorted(self._cdf, rng.random(n), side="right")
        return np.minimum(drawn, self.n_bundles - 1)

    # ---- epoch mode ------------------------------------------------------------------------

    def epoch(self, epoch: int, seed: int = 0, cover: str = "source", drop_last: bool = False) -> list[Sample]:
        """Every cell of every group on the covered side is used once, in shuffled order.

        Each covered group is shuffled and cut into chunks of n_source (or n_target) cells; the
        last chunk keeps what is left, or is left out with ``drop_last``. Each chunk takes one
        bundle that contains its group on the covered side, by bundle weight, and the other
        groups' cells are drawn as in step mode.
        """
        if self.single_cells:
            source_rows, target_rows = self._epoch_single(epoch, seed, cover)
            return [[([int(i)], [int(j)])] for i, j in zip(source_rows, target_rows)]
        rng = np.random.default_rng([seed, epoch])
        covered_of_pair = self.pair_source if cover == "source" else self.pair_target
        size = self.n_source if cover == "source" else self.n_target
        bundles_of_group = {}
        for b, pairs in enumerate(self.bundle_pairs):
            for g in np.unique(covered_of_pair[pairs]):
                bundles_of_group.setdefault(int(g), []).append(b)

        samples = []
        for g in sorted(bundles_of_group):
            bundles = np.asarray(bundles_of_group[g])
            cdf = np.cumsum(self.bundle_weight[bundles]) / self.bundle_weight[bundles].sum()
            rows = rng.permutation(self.groups.rows[g])
            chunk = len(rows) if size == "all" else size
            starts = range(0, len(rows) - chunk + 1 if drop_last else len(rows), chunk)
            chosen = bundles[np.minimum(np.searchsorted(cdf, rng.random(len(starts)), side="right"), len(bundles) - 1)]
            for start, b in zip(starts, chosen):
                samples.append(self._realize(b, rng, cover, fixed=(g, rows[start:start + chunk])))
        return [samples[i] for i in rng.permutation(len(samples))]

    def epoch_rows(self, epoch: int, seed: int = 0, cover: str = "source",
                   drop_last: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """The cell pairs of ``epoch()``, in order, as a source-row array and a target-row array.

        A set sample adds its source cells and its target cells side by side, so the two arrays
        line up only when each sample has as many source as target cells. With single cells this
        skips building the samples.
        """
        if self.single_cells:
            return self._epoch_single(epoch, seed, cover)
        source_rows, target_rows = [], []
        for sample in self.epoch(epoch, seed, cover, drop_last):
            for source, target in sample:
                source_rows.extend(source)
                target_rows.extend(target)
        return np.asarray(source_rows, dtype=np.int64), np.asarray(target_rows, dtype=np.int64)

    def _epoch_single(self, epoch: int, seed: int, cover: str) -> tuple[np.ndarray, np.ndarray]:
        """``epoch()`` for single cells, one array operation per step instead of one call per cell."""
        rng = np.random.default_rng([seed, epoch])
        covered_of_pair, other_of_pair = ((self.pair_source, self.pair_target) if cover == "source"
                                          else (self.pair_target, self.pair_source))
        n_groups = len(self._count)

        # Pair rows sorted by covered group, with a running total of their weights: the rows of
        # group g fill the stretch [weight_before[g], weight_before[g] + weight_of[g]).
        order = np.argsort(covered_of_pair, kind="stable")
        cum = np.cumsum(self.bundle_weight[order])
        row_end = np.searchsorted(covered_of_pair[order], np.arange(n_groups), side="right")
        row_start = np.searchsorted(covered_of_pair[order], np.arange(n_groups), side="left")
        cum0 = np.concatenate([[0.0], cum])
        weight_before = cum0[row_start]
        weight_of = cum0[row_end] - weight_before

        # Every cell of every covered group once, in shuffled order.
        is_covered = row_end > row_start
        cells = rng.permutation(self._flat[np.repeat(is_covered, self._count)])
        g = self._group_of_row[cells]

        # A pair row for every cell by weight: a random point in its group's stretch, then one
        # binary search for all cells. Rounding can put the point on the stretch end; clip it back.
        point = weight_before[g] + rng.random(len(cells)) * weight_of[g]
        p = order[np.minimum(np.searchsorted(cum, point, side="right"), row_end[g] - 1)]

        partner = self._partners(cells, p, cover, rng)
        linked = partner >= 0
        cells, partner = cells[linked], partner[linked]
        return (cells, partner) if cover == "source" else (partner, cells)

    def _partners(self, cells: np.ndarray, p: np.ndarray, cover: str, rng) -> np.ndarray:
        """One partner per covered cell for pair rows ``p``; -1 where a "table" cell has no link."""
        other = (self.pair_target if cover == "source" else self.pair_source)[p]
        partner = self._flat[self._start[other] + rng.integers(self._count[other])]
        partner = np.where(self._is_self[p], cells, partner)
        if self.uses_link_table:
            is_table = self._is_table[p]
            partner[is_table] = self._links(cover).pick(cells[is_table], other[is_table], rng)
        return partner

    def _links(self, cover: str) -> "_Links":
        if cover not in self._link_index:
            cell, partner = ("source_row", "target_row") if cover == "source" else ("target_row", "source_row")
            table = self._link_table
            weight = table["weight"].to_numpy() if "weight" in table.columns else np.ones(table.height)
            self._link_index[cover] = _Links(table[cell].to_numpy(), table[partner].to_numpy(), weight,
                                             self._group_of_row, len(self._count))
        return self._link_index[cover]

    # ---- shared ----------------------------------------------------------------------------

    def _realize(self, b: int, rng, cover: str, fixed=None) -> Sample:
        """Cells for every pair row of bundle ``b``. Each group's cells are drawn once per sample."""
        other_side = "target" if cover == "source" else "source"
        size = {"source": self.n_source, "target": self.n_target}
        drawn = {}

        def cells(side, g):
            if (side, g) not in drawn:
                if fixed is not None and side == cover and g == fixed[0]:
                    drawn[(side, g)] = fixed[1]
                else:
                    drawn[(side, g)] = self._draw_cells(g, size[side], rng)
            return drawn[(side, g)]

        sample = []
        for p in self.bundle_pairs[b]:
            groups_of_pair = {"source": self.pair_source[p], "target": self.pair_target[p]}
            covered_group, other_group = groups_of_pair[cover], groups_of_pair[other_side]
            covered_rows = cells(cover, covered_group)
            link_kind = self.pair_link[p]

            if link_kind == "none":
                other_rows = cells(other_side, other_group)
                rows = {cover: covered_rows, other_side: other_rows}
                sample.append((rows["source"].tolist(), rows["target"].tolist()))
                continue

            partners = self._partners(covered_rows, np.full(len(covered_rows), p), cover, rng)
            for cell, partner in zip(covered_rows, partners):
                if partner < 0:
                    continue  # this cell has no link into the partner group
                link = {cover: [int(cell)], other_side: [int(partner)]}
                sample.append((link["source"], link["target"]))
        return sample

    def _draw_cells(self, g: int, n, rng) -> np.ndarray:
        """``n`` distinct cells of group ``g``, or all of them (shuffled) if it has fewer."""
        rows = self.groups.rows[g]
        if n == "all" or n >= len(rows):
            return rng.permutation(rows)
        return rows[rng.choice(len(rows), n, replace=False)]


class _Links:
    """A link table sorted by (cell, partner group): a cell's links into one group are one block."""

    def __init__(self, cell, partner, weight, group_of_row, n_groups: int):
        cell, partner = cell.astype(np.int64), partner.astype(np.int64)
        self.n_groups = n_groups
        key = cell * n_groups + group_of_row[partner]
        order = np.argsort(key, kind="stable")
        self.key, self.partner = key[order], partner[order]
        # cum[m] is the weight of links before link m, so link m covers [cum[m], cum[m + 1]).
        self.cum = np.concatenate([[0.0], np.cumsum(weight[order].astype(float))])

    def pick(self, cells: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
        """One linked partner per cell inside the given group, by weight; -1 where there is none."""
        want = cells * self.n_groups + groups
        start = np.searchsorted(self.key, want, side="left")
        end = np.searchsorted(self.key, want, side="right")
        point = self.cum[start] + rng.random(len(cells)) * (self.cum[end] - self.cum[start])
        m = np.minimum(np.maximum(np.searchsorted(self.cum, point, side="right") - 1, start), end - 1)
        return np.where(end > start, self.partner[m], -1)
