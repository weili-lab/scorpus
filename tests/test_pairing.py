import numpy as np
import pandas as pd
import polars as pl
import pytest

from scorpus.pairing import Sampler, cells, combine, cross, links_from, make_groups, src, tgt, write_order

is_ctrl = pl.col("perturbation") == "ctrl"


def development_table() -> pd.DataFrame:
    """40 controls at -1, 20 controls at each later time, 10 cells per (perturbation, time, dose)."""
    rows = [("ctrl", -1, None)] * 40
    rows += [("ctrl", t, None) for t in (3, 7, 10) for _ in range(20)]
    rows += [(f"P{k}", t, d) for k in range(10) for t in (3, 7, 10) for d in (1.0, 10.0) for _ in range(10)]
    table = pd.DataFrame(rows, columns=["perturbation", "time", "dose"])
    table["replicate"] = "r1"
    return table


def cell_lines_table() -> pd.DataFrame:
    return pd.DataFrame({
        "cell_line": ["K562"] * 9 + ["RPE1"] * 6,
        "perturbation": ["ctrl"] * 3 + ["KLF1"] * 3 + ["GATA1"] * 3 + ["ctrl"] * 3 + ["KLF1"] * 3,
    })


def development_pairs(groups) -> pl.DataFrame:
    """A time course with dose: from the origin, continue the same treatment, raise the dose."""
    from_origin = cross(groups, on=["replicate"], source=is_ctrl & (pl.col("time") == -1), target=pl.col("time") > -1)
    continue_treatment = cross(groups, on=["replicate", "perturbation", "dose"], source=~is_ctrl).filter(tgt.time > src.time)
    dose_up = cross(groups, on=["replicate", "perturbation", "time"], source=~is_ctrl).filter(tgt.dose > src.dose)
    return combine(
        from_origin.with_columns(rule=pl.lit("from_origin")),
        continue_treatment.with_columns(rule=pl.lit("continue_perturbation")),
        dose_up.with_columns(rule=pl.lit("dose_response")),
    )


def test_control_to_perturbed_single_cells():
    table = cell_lines_table()
    groups = make_groups(table, by=["cell_line", "perturbation"])
    pairs = cross(groups, on=["cell_line"], source=is_ctrl, target=~is_ctrl)
    got = sorted(zip(pairs["source_cell_line"], pairs["target_perturbation"]))
    assert got == [("K562", "GATA1"), ("K562", "KLF1"), ("RPE1", "KLF1")]

    sampler = Sampler(groups, pairs, n_source=1)
    for step in range(50):
        ((s, t),) = sampler.sample(step)
        assert table.iloc[s[0]]["perturbation"] == "ctrl" and table.iloc[t[0]]["perturbation"] != "ctrl"
        assert table.iloc[s[0]]["cell_line"] == table.iloc[t[0]]["cell_line"]


def test_development_rule_counts():
    groups = make_groups(development_table(), by=["perturbation", "time", "dose", "replicate"])
    pairs = development_pairs(groups)
    assert dict(pairs.group_by("rule").len().iter_rows()) == {
        "from_origin": 63, "continue_perturbation": 60, "dose_response": 30,
    }


def test_dose_as_cell_level_condition():
    table = development_table()
    groups = make_groups(table, by=["perturbation", "time", "dose", "replicate"])
    pairs = development_pairs(groups).filter(pl.col("rule") != "dose_response")
    sampler = Sampler(groups, pairs, n_source=4, links="random",
                      bundle_by=["rule", "source_perturbation", "source_time", "target_perturbation", "target_time"])
    for step in range(100):
        for (s,), (t,) in sampler.sample(step):
            source, target = table.iloc[s], table.iloc[t]
            assert source["time"] < target["time"]
            if source["perturbation"] != "ctrl":
                assert source["dose"] == target["dose"] and source["perturbation"] == target["perturbation"]


def test_combinations_with_plain_polars():
    groups = make_groups(pd.DataFrame({"perturbation": ["ctrl", "A", "B", "A+B", "C"]}), by=["perturbation"])
    def parts(e):
        return e.str.split("+")

    singles = pl.col("perturbation").is_in(["A", "B", "C"])
    add_one = cross(groups, source=singles, target=pl.col("perturbation") == "A+B").filter(
        parts(src.perturbation).list.set_difference(parts(tgt.perturbation)).list.len() == 0
    )
    assert sorted(add_one["source_perturbation"]) == ["A", "B"]


def test_fallback_matching():
    """Controls only exist in donor D1; D2's perturbed cells fall back to cell-type matching."""
    table = pd.DataFrame({
        "donor": ["D1", "D1", "D2"], "celltype": ["T", "T", "T"], "perturbation": ["ctrl", "KLF1", "KLF1"],
    })
    groups = make_groups(table, by=["donor", "celltype", "perturbation"])
    same_donor = cross(groups, on=["donor", "celltype"], source=is_ctrl, target=~is_ctrl)
    fallback = cross(groups, on=["celltype"], source=is_ctrl, target=~is_ctrl).join(
        same_donor.select("target_group"), on="target_group", how="anti"
    )
    pairs = combine(same_donor.with_columns(rule=pl.lit("same_donor")), fallback.with_columns(rule=pl.lit("fallback")))
    assert sorted(zip(pairs["target_donor"], pairs["rule"])) == [("D1", "same_donor"), ("D2", "fallback")]


def test_next_time_point_only():
    table = pd.DataFrame({"perturbation": "X", "time": [3, 7, 10]})
    groups = make_groups(table, by=["perturbation", "time"]).with_columns(
        next_time=pl.col("time").shift(-1).over("perturbation", order_by="time")
    )
    pairs = cross(groups, on=["perturbation"]).filter(tgt.time == src.next_time)
    assert sorted(zip(pairs["source_time"], pairs["target_time"])) == [(3, 7), (7, 10)]


def test_dose_series_shares_control_cells():
    table = pd.DataFrame({"perturbation": ["ctrl"] * 10 + ["X"] * 30, "dose": [None] * 10 + [1] * 10 + [5] * 10 + [10] * 10})
    groups = make_groups(table, by=["perturbation", "dose"])
    pairs = cross(groups, source=is_ctrl, target=~is_ctrl)
    sampler = Sampler(groups, pairs, n_source=4, bundle_by=["source_group", "target_perturbation"])
    sample = sampler.sample(0)
    assert len(sample) == 3
    assert sample[0][0] == sample[1][0] == sample[2][0]  # the same control cells for every dose
    assert sorted(table.iloc[t[0]]["dose"] for _, t in sample) == [1, 5, 10]


def test_population_to_single_cell():
    table = cell_lines_table()
    groups = make_groups(table, by=["cell_line", "perturbation"])
    sampler = Sampler(groups, cross(groups, on=["cell_line"], source=is_ctrl, target=~is_ctrl), n_source="all", n_target=1)
    ((s, t),) = sampler.sample(0)
    assert len(s) == 3 and len(t) == 1


def test_lineage_links():
    """Real cell links from a parent_id column: every link is a true parent -> child pair."""
    table = pd.DataFrame({
        "cell_id": [f"p{i}" for i in range(5)] + [f"c{i}" for i in range(10)],
        "parent_id": [None] * 5 + [f"p{i // 2}" for i in range(10)],
        "time": [0] * 5 + [1] * 10,
    })
    row_of = {cell: i for i, cell in enumerate(table["cell_id"])}
    children = table.index[table["parent_id"].notna()]
    links = pd.DataFrame({"source_row": [row_of[table.loc[i, "parent_id"]] for i in children], "target_row": children})
    groups = make_groups(table, by=["time"])
    sampler = Sampler(groups, cross(groups, source=pl.col("time") == 0, target=pl.col("time") == 1), n_source=3, links=links)
    for step in range(20):
        for (s,), (t,) in sampler.sample(step):
            assert table.loc[t, "parent_id"] == table.loc[s, "cell_id"]


def test_epoch_uses_every_covered_cell_once():
    table = development_table()
    groups = make_groups(table, by=["perturbation", "time", "dose", "replicate"])
    pairs = development_pairs(groups)
    sampler = Sampler(groups, pairs, n_source=8)
    for cover, side, column in (("source", 0, "source_group"), ("target", 1, "target_group")):
        used = [row for sample in sampler.epoch(0, cover=cover) for row in sample[0][side]]
        expected = np.concatenate([groups.rows[g] for g in pairs[column].unique().to_list()])
        assert sorted(used) == sorted(expected.tolist())


def test_weights_and_batch_sampling():
    table = cell_lines_table()
    groups = make_groups(table, by=["cell_line", "perturbation"])
    pairs = cross(groups, on=["cell_line"], source=is_ctrl, target=~is_ctrl).with_columns(
        weight=pl.when(tgt.perturbation == "GATA1").then(0.0).otherwise(1.0)
    )
    sampler = Sampler(groups, pairs, weight="weight")
    batch = sampler.sample_batch(step=0, n_bundles=500)
    assert len(batch) == 500
    assert all(table.iloc[t[0]]["perturbation"] == "KLF1" for ((s, t),) in batch)
    assert all(table.iloc[s[0]]["cell_line"] == table.iloc[t[0]]["cell_line"] for ((s, t),) in batch)


def perttf_training_pairs(groups) -> pl.DataFrame:
    """pertTF's default: controls -> a perturbation of the same cell line, perturbed cells -> themselves."""
    ctrl_to_pert = cross(groups, on=["cell_line"], source=is_ctrl, target=~is_ctrl).with_columns(link=pl.lit("random"))
    pert_self = cross(groups, on=["cell_line", "perturbation"], source=~is_ctrl).with_columns(link=pl.lit("self"))
    return combine(ctrl_to_pert, pert_self)


def test_self_links_per_row():
    table = cell_lines_table()
    groups = make_groups(table, by=["cell_line", "perturbation"])
    sampler = Sampler(groups, perttf_training_pairs(groups), n_source=2, links="link")
    for step in range(100):
        for (s,), (t,) in sampler.sample(step):
            if table.iloc[s]["perturbation"] == "ctrl":
                assert table.iloc[t]["perturbation"] != "ctrl" and table.iloc[t]["cell_line"] == table.iloc[s]["cell_line"]
            else:
                assert s == t
    used = [s for sample in sampler.epoch(0) for (s,), _ in sample]
    assert sorted(used) == list(range(len(table)))


def test_self_links_in_batch_fast_path():
    table = cell_lines_table()
    groups = make_groups(table, by=["cell_line", "perturbation"])
    sampler = Sampler(groups, perttf_training_pairs(groups), links="link")
    for ((s,), (t,)), in sampler.sample_batch(step=0, n_bundles=300):
        assert (s == t) == (table.iloc[s]["perturbation"] != "ctrl")


def test_self_links_when_covering_targets():
    """pertTF validation: every target once; perturbed targets get a control, control targets themselves."""
    table = cell_lines_table()
    groups = make_groups(table, by=["cell_line", "perturbation"])
    pairs = combine(
        cross(groups, on=["cell_line"], source=is_ctrl, target=~is_ctrl).with_columns(link=pl.lit("random")),
        cross(groups, on=["cell_line", "perturbation"], source=is_ctrl).with_columns(link=pl.lit("self")),
    )
    samples = Sampler(groups, pairs, links="link").epoch(0, cover="target")
    assert sorted(t for sample in samples for _, (t,) in sample) == list(range(len(table)))
    for sample in samples:
        for (s,), (t,) in sample:
            if table.iloc[t]["perturbation"] == "ctrl":
                assert s == t
            else:
                assert table.iloc[s]["perturbation"] == "ctrl" and table.iloc[s]["cell_line"] == table.iloc[t]["cell_line"]


def test_self_link_needs_same_group():
    groups = make_groups(cell_lines_table(), by=["cell_line", "perturbation"])
    with pytest.raises(ValueError):
        Sampler(groups, cross(groups, on=["cell_line"], source=is_ctrl, target=~is_ctrl), links="self")


def test_links_from_and_mixed_links():
    """OT-style links for control -> perturbed, self links for perturbed cells, in one sampler."""
    table = cell_lines_table()
    groups = make_groups(table, by=["cell_line", "perturbation"])
    ctrl_to_pert = cross(groups, on=["cell_line"], source=is_ctrl, target=~is_ctrl)
    # stand-in for an OT assignment: the k-th control cell is linked to the k-th target cell
    links = links_from(groups, ctrl_to_pert, lambda s, t: (np.arange(min(len(s), len(t))),) * 2)
    assert links.height == 3 * ctrl_to_pert.height
    for s, t in links.select("source_row", "target_row").iter_rows():
        assert table.iloc[s]["perturbation"] == "ctrl" and table.iloc[t]["perturbation"] != "ctrl"
        assert table.iloc[s]["cell_line"] == table.iloc[t]["cell_line"]

    pairs = combine(
        ctrl_to_pert.with_columns(link=pl.lit("table")),
        cross(groups, on=["cell_line", "perturbation"], source=~is_ctrl).with_columns(link=pl.lit("self")),
    )
    linked = set(links.select("source_row", "target_row").iter_rows())
    sampler = Sampler(groups, pairs, n_source=3, links="link", link_table=links)
    for step in range(100):
        for (s,), (t,) in sampler.sample(step):
            assert (s, t) in linked or s == t


def test_drop_small_groups():
    table = pd.DataFrame({"perturbation": ["ctrl"] * 10 + ["A"] * 10 + ["B"] * 2})
    groups = make_groups(table, by=["perturbation"])
    pairs = cross(groups, source=is_ctrl, target=~is_ctrl)
    kept = Sampler(groups, pairs, n_source=5, drop_small=True)
    assert kept.n_bundles == 1
    for step in range(50):
        ((s, t),) = kept.sample(step)
        assert len(s) == len(t) == 5 and set(table.iloc[t]["perturbation"]) == {"A"}
    # without drop_small the small group gives the cells it has
    sizes = {len(t) for step in range(50) for ((s, t),) in [Sampler(groups, pairs, n_source=5).sample(step)]}
    assert sizes == {2, 5}


def test_epoch_drop_last():
    table = pd.DataFrame({"perturbation": ["ctrl"] * 10 + ["A"] * 10})
    groups = make_groups(table, by=["perturbation"])
    sampler = Sampler(groups, cross(groups, source=is_ctrl, target=~is_ctrl), n_source=4)
    assert sorted(len(s) for ((s, t),) in sampler.epoch(0)) == [2, 4, 4]
    assert sorted(len(s) for ((s, t),) in sampler.epoch(0, drop_last=True)) == [4, 4]


def test_single_cell_epoch_rows():
    """The array epoch: every covered cell once, partners in the allowed group, by weight."""
    table = pd.DataFrame({
        "cell_line": ["K562"] * 4000 + ["RPE1"] * 30,
        "perturbation": ["ctrl"] * 2000 + ["KLF1"] * 1000 + ["GATA1"] * 1000 + ["ctrl"] * 10 + ["KLF1"] * 20,
    })
    groups = make_groups(table, by=["cell_line", "perturbation"])
    pairs = perttf_training_pairs(groups).with_columns(weight=pl.when(tgt.perturbation == "GATA1").then(3.0).otherwise(1.0))
    sampler = Sampler(groups, pairs, links="link", weight="weight")
    assert sampler.single_cells

    s, t = sampler.epoch_rows(0)
    assert sorted(s.tolist()) == list(range(len(table)))
    source, target = table.iloc[s].reset_index(drop=True), table.iloc[t].reset_index(drop=True)
    is_ctrl_source = (source["perturbation"] == "ctrl").to_numpy()
    assert (source["cell_line"] == target["cell_line"]).all()
    assert (s[~is_ctrl_source] == t[~is_ctrl_source]).all()
    assert (target["perturbation"][is_ctrl_source] != "ctrl").all()
    k562_ctrl = is_ctrl_source & (source["cell_line"] == "K562").to_numpy()
    assert abs((target["perturbation"][k562_ctrl] == "GATA1").mean() - 0.75) < 0.05

    assert [(i, j) for ((i,), (j,)), in sampler.epoch(0)] == list(zip(s.tolist(), t.tolist()))
    assert not np.array_equal(sampler.epoch_rows(1)[0], s)


def test_epoch_rows_for_sets():
    table = cell_lines_table()
    groups = make_groups(table, by=["cell_line", "perturbation"])
    sampler = Sampler(groups, perttf_training_pairs(groups), n_source=2, links="link")
    assert not sampler.single_cells
    s, t = sampler.epoch_rows(0)
    flat = [(i, j) for sample in sampler.epoch(0) for (i,), (j,) in sample]
    assert list(zip(s.tolist(), t.tolist())) == flat


def test_links_from_top_k():
    """A transport plan's entries as weighted links; top_k keeps every cell's heaviest links."""
    table = pd.DataFrame({"perturbation": ["ctrl"] * 4 + ["A"] * 3})
    groups = make_groups(table, by=["perturbation"])
    pairs = cross(groups, source=is_ctrl, target=~is_ctrl)
    plan = np.array([[5, 1, 0.5], [4, 2, 0.1], [3, 2.5, 0.2], [6, 0.3, 0.4]])

    def match(s, t):
        i, j = np.nonzero(plan)
        return i, j, plan[i, j]

    links = links_from(groups, pairs, match, top_k=1)
    got = {(s, t): w for s, t, w in links.iter_rows()}
    # heaviest per control (rows 0-3): all go to A cell 4; heaviest per A cell: 3->4, 2->5, 0->6
    assert got == {(0, 4): 5, (1, 4): 4, (2, 4): 3, (3, 4): 6, (2, 5): 2.5, (0, 6): 0.5}
    assert links_from(groups, pairs, match).height == plan.size


def test_weighted_links():
    """A cell picks among its links into the partner group by weight; a cell without links is skipped."""
    table = pd.DataFrame({"perturbation": ["ctrl"] * 3 + ["A"] * 3})
    groups = make_groups(table, by=["perturbation"])
    pairs = cross(groups, source=is_ctrl, target=~is_ctrl)
    links = pl.DataFrame({"source_row": [0, 0, 1], "target_row": [3, 4, 5], "weight": [0.7, 0.3, 1.0]})
    sampler = Sampler(groups, pairs, links=links)
    assert sampler.single_cells

    batch = [pair for sample in sampler.sample_batch(step=0, n_bundles=6000) for pair in sample]
    from_0 = [t for (s,), (t,) in batch if s == 0]
    assert abs(np.mean(np.array(from_0) == 3) - 0.7) < 0.03
    assert {t for (s,), (t,) in batch if s == 1} == {5}
    assert {s for (s,), _ in batch} == {0, 1}  # control 2 has no link

    s, t = sampler.epoch_rows(0)
    assert sorted(s.tolist()) == [0, 1]
    s, t = sampler.epoch_rows(0, cover="target")
    assert sorted(zip(t.tolist(), s.tolist())) == [(3, 0), (4, 0), (5, 1)]

    # the per-cell path (sets of covered cells) picks by the same weights
    sets = Sampler(groups, pairs, n_source=3, links=links)
    from_0 = [t for step in range(2000) for (s,), (t,) in sets.sample(step) if s == 0]
    assert abs(np.mean(np.array(from_0) == 3) - 0.7) < 0.04


def test_cells_without_pairs():
    table = pd.DataFrame({"perturbation": ["ctrl"] * 7 + ["A"] * 5})
    groups = make_groups(table, by=["perturbation"])
    rows = cells(groups, epoch=0)
    assert sorted(rows.tolist()) == list(range(12))
    assert not np.array_equal(rows, cells(groups, epoch=1))
    np.testing.assert_array_equal(rows, cells(groups, epoch=0))

    batches = cells(groups, epoch=0, batch_size=5)
    assert [len(b) for b in batches] == [5, 5, 2]
    assert [len(b) for b in cells(groups, epoch=0, batch_size=5, drop_last=True)] == [5, 5]

    same = cells(groups, epoch=0, batch_size=3, same_group=True)
    assert sorted(len(b) for b in same) == [1, 2, 3, 3, 3]
    assert all(table.iloc[b]["perturbation"].nunique() == 1 for b in same)
    assert sorted(np.concatenate(same).tolist()) == list(range(12))
    assert sorted(len(b) for b in cells(groups, epoch=0, batch_size=3, same_group=True, drop_last=True)) == [3, 3, 3]


def test_write_order():
    """Every group one contiguous block, groups sorted by their columns, rows shuffled within."""
    table = pd.DataFrame({"perturbation": ["B", "A", None] * 1000, "lane": [1, 2] * 1500})
    groups = make_groups(table, by=["perturbation", "lane"])
    order, blocks = write_order(groups, seed=0)
    assert blocks.select("perturbation", "lane").rows() == [
        (None, 1), (None, 2), ("A", 1), ("A", 2), ("B", 1), ("B", 2)]

    # make_groups numbers groups by first appearance, so in the rewritten table group k is block k
    rewritten = make_groups(table.iloc[order], by=["perturbation", "lane"])
    assert rewritten.table.select("perturbation", "lane").rows() == blocks.select("perturbation", "lane").rows()
    for rows, start, stop in zip(rewritten.rows, blocks["start"], blocks["stop"]):
        assert rows.tolist() == list(range(start, stop))

    # within a block the old rows come in an order unrelated to their original order
    correlations = [np.corrcoef(np.arange(stop - start), order[start:stop])[0, 1]
                    for start, stop in blocks.select("start", "stop").iter_rows()]
    assert np.abs(correlations).max() < 0.2


def test_write_order_keeps_only_the_given_rows():
    """With keep, each group holds exactly its kept rows, shuffled; a group with none kept is dropped."""
    table = pd.DataFrame({"perturbation": ["B", "A", None] * 1000, "lane": [1, 2] * 1500})
    groups = make_groups(table, by=["perturbation", "lane"])
    rng = np.random.default_rng(1)
    dropped_group = groups.table.filter(pl.col("perturbation") == "A", pl.col("lane") == 2)["group"].item()
    keep = rng.permutation(np.setdiff1d(np.arange(3000), groups.rows[dropped_group]))[:1200]
    order, blocks = write_order(groups, seed=0, keep=keep)

    assert sorted(order.tolist()) == sorted(keep.tolist())
    assert blocks.select("perturbation", "lane").rows() == [(None, 1), (None, 2), ("A", 1), ("B", 1), ("B", 2)]
    for group, start, stop in blocks.select("group", "start", "stop").iter_rows():
        assert set(order[start:stop]) == set(groups.rows[group]) & set(keep.tolist())
        assert np.abs(np.corrcoef(np.arange(stop - start), order[start:stop])[0, 1]) < 0.2
    assert blocks["stop"][-1] == len(keep)

    # keeping every row is the same as no keep
    np.testing.assert_array_equal(write_order(groups, seed=0, keep=np.arange(3000))[0], write_order(groups, seed=0)[0])


def run_breaks(rows: list[int], first_row: int, group_size: int) -> int:
    """Places where the next cell is not the next row of its group (wrapping): runs - 1."""
    position = np.asarray(rows) - first_row
    return int(np.sum(np.diff(position) % group_size != 1))


def test_run_draws_give_every_cell_the_same_chance():
    """Runs of 4 from groups of 23 and 24 cells: 6 cells are two runs, and every cell is drawn equally often."""
    table = pd.DataFrame({"perturbation": ["ctrl"] * 23 + ["A"] * 24})
    groups = make_groups(table, by=["perturbation"])
    sampler = Sampler(groups, cross(groups, source=is_ctrl, target=~is_ctrl), n_source=6, run_length=4)
    times_drawn = np.zeros(len(table))
    n_draws = 20000
    for step in range(n_draws):
        ((s, t),) = sampler.sample(step)
        assert len(set(s)) == len(set(t)) == 6
        assert run_breaks(s, 0, 23) <= 1 and run_breaks(t, 23, 24) <= 1
        times_drawn[s + t] += 1
    frequency = times_drawn / n_draws
    assert np.abs(frequency[:23] - 6 / 23).max() < 0.015
    assert np.abs(frequency[23:] - 6 / 24).max() < 0.015


def test_run_epoch_uses_every_cell_once():
    table = pd.DataFrame({"perturbation": ["ctrl"] * 23 + ["A"] * 24})
    groups = make_groups(table, by=["perturbation"])
    sampler = Sampler(groups, cross(groups, source=is_ctrl, target=~is_ctrl), n_source=8, run_length=4)
    for cover, side, first_row, size in (("source", 0, 0, 23), ("target", 1, 23, 24)):
        chunks = [sample[0][side] for sample in sampler.epoch(0, cover=cover)]
        assert sorted(row for chunk in chunks for row in chunk) == list(range(first_row, first_row + size))
        # the group is 6 runs of 4 (one of 3 in the group of 23), so cells change run at most 5 times
        assert sum(run_breaks(chunk, first_row, size) for chunk in chunks) <= 5


def test_run_length_one_matches_cell_by_cell_draws():
    """Seeded outputs recorded from the sampler before run_length existed."""
    table = pd.DataFrame({"perturbation": ["ctrl"] * 10 + ["A"] * 10})
    groups = make_groups(table, by=["perturbation"])
    sampler = Sampler(groups, cross(groups, source=is_ctrl, target=~is_ctrl), n_source=4, run_length=1)
    assert [sampler.sample(step) for step in range(3)] == [
        [([2, 8, 0, 3], [15, 17, 19, 18])], [([8, 4, 2, 6], [17, 12, 10, 19])], [([3, 8, 2, 6], [17, 19, 12, 10])]]
    assert sampler.epoch(0) == [
        [([8, 1], [16, 18, 15, 10])], [([3, 5, 9, 0], [14, 16, 17, 10])], [([4, 6, 2, 7], [17, 18, 13, 12])]]
