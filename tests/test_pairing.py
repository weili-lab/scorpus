import numpy as np
import pandas as pd
import polars as pl
import pytest

from scorpus.pairing import Sampler, combine, cross, links_from, make_groups, src, tgt

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
    parts = lambda e: e.str.split("+")
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
    for s, t in links.iter_rows():
        assert table.iloc[s]["perturbation"] == "ctrl" and table.iloc[t]["perturbation"] != "ctrl"
        assert table.iloc[s]["cell_line"] == table.iloc[t]["cell_line"]

    pairs = combine(
        ctrl_to_pert.with_columns(link=pl.lit("table")),
        cross(groups, on=["cell_line", "perturbation"], source=~is_ctrl).with_columns(link=pl.lit("self")),
    )
    linked = set(links.iter_rows())
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
