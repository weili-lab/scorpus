"""A corpus rewritten in a new row order holds the same cells, moved together, in both layouts."""

from dataclasses import replace

import anndata as ad
import lance
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import sparse

from scorpus import from_h5ad, load_corpus, reorder_corpus
from scorpus.materializers import reorder
from scorpus.pairing import make_groups, write_order


def assert_same_batch(actual, expected):
    np.testing.assert_array_equal(actual.global_row_index, expected.global_row_index)
    np.testing.assert_array_equal(actual.row_offsets, expected.row_offsets)
    np.testing.assert_array_equal(actual.expressed_gene_indices, expected.expressed_gene_indices)
    np.testing.assert_array_equal(actual.expression_counts, expected.expression_counts)
    assert actual.expression_counts.dtype == expected.expression_counts.dtype


def write_source_corpus(tmp_path, float_counts):
    """60 cells with two grouping columns (one with missing values) and a guide UMI extra matrix."""
    rng = np.random.default_rng(0)
    counts = sparse.random(60, 30, density=0.3, random_state=1, format="csr")
    if float_counts:
        counts = (counts * 1000).astype(np.float32)  # fractional float32 values, kept as they are
    else:
        counts.data = np.ceil(counts.data * 100_000)  # whole counts above 65,535
    perturbation = pd.Categorical(rng.choice(["A", "B", "C"], 60))
    perturbation[rng.random(60) < 0.2] = np.nan
    obs = pd.DataFrame({"perturbation": perturbation, "lane": rng.choice(["l1", "l2"], 60)},
                       index=[f"c{i}" for i in range(60)])
    var = pd.DataFrame(index=[f"g{j}" for j in range(30)])
    guide_umi = sparse.csr_matrix(rng.poisson(0.5, (60, 3)).astype(np.int32))
    ad.AnnData(counts, obs=obs, var=var, obsm={"guide_umi": guide_umi}).write_h5ad(tmp_path / "input.h5ad")
    options = {"layers": "X", "no_count": True} if float_counts else {}
    from_h5ad(tmp_path / "input.h5ad", tmp_path / "source", chunk_rows=25,
              extra={"guide_umi": ("obsm/guide_umi", ["sg1", "sg2", "sg3"])}, **options)
    return load_corpus(tmp_path / "source")


@pytest.mark.parametrize("float_counts", [False, True])
def test_reordered_corpus_round_trips_in_both_layouts(tmp_path, monkeypatch, float_counts):
    monkeypatch.setattr(reorder, "BLOCK_ROWS", 7)  # several blocks per table
    source = write_source_corpus(tmp_path, float_counts)
    order, _ = write_order(make_groups(source.obs, ["perturbation", "lane"]), seed=0)
    note = {"by": ["perturbation", "lane"], "seed": 0, "source": "somewhere"}
    reorder_corpus(tmp_path / "source", tmp_path / "two_column", order, note)
    reorder_corpus(tmp_path / "source", tmp_path / "packed", order, note, layout="packed")

    new_rows = np.arange(60)
    expected = replace(source.expression_reader.read_expression_flat(order), global_row_index=new_rows)
    expected_guides = source.read_extra("guide_umi", order)[0].toarray()
    for layout in ["two_column", "packed"]:
        root = tmp_path / layout
        corpus = load_corpus(root)
        assert_same_batch(corpus.expression_reader.read_expression_flat(new_rows), expected)
        pd.testing.assert_frame_equal(corpus.obs, source.obs.iloc[order])
        np.testing.assert_array_equal(corpus.read_extra("guide_umi", new_rows)[0].toarray(), expected_guides)
        if not float_counts:
            pd.testing.assert_frame_equal(pd.read_parquet(root / "statistics.parquet"),
                                          pd.read_parquet(tmp_path / "source" / "statistics.parquet").iloc[order]
                                          .reset_index(drop=True))
        for table in ["expression.lance", "extra/guide_umi.lance"]:
            rows = lance.dataset(root / table).to_table(columns=["global_row_index", "source_row"])
            np.testing.assert_array_equal(rows["global_row_index"].to_numpy(), new_rows)
            np.testing.assert_array_equal(rows["source_row"].to_numpy(), order)
        manifest = yaml.safe_load((root / "corpus.yaml").read_text())
        assert manifest["row_order"] == note
        assert manifest.get("expression_layout", "two_column") == layout
    assert expected.expression_counts.dtype == (np.float32 if float_counts else np.int32)
    assert "packed" in lance.dataset(tmp_path / "packed" / "expression.lance").schema.names


def test_packing_refuses_what_it_cannot_store_exactly(tmp_path):
    genes = np.array([1, 2], dtype=np.int32)
    np.testing.assert_array_equal(reorder._pack(genes, np.array([3.0, 70_000.0])),
                                  [(3 << 16) | 1, (70_000 << 16) | 2])
    for counts in [np.array([1.5, 2.0]), np.array([-1, 2], dtype=np.int32), np.array([2**32, 1], dtype=np.int64)]:
        with pytest.raises(ValueError):
            reorder._pack(genes, counts)

    # 65,536 genes: gene indices no longer fit in 16 bits.
    counts = sparse.csr_matrix(([5.0], ([0], [65_535])), shape=(3, 65_536))
    var = pd.DataFrame(index=[f"g{j}" for j in range(65_536)])
    ad.AnnData(counts, obs=pd.DataFrame(index=["a", "b", "c"]), var=var).write_h5ad(tmp_path / "wide.h5ad")
    from_h5ad(tmp_path / "wide.h5ad", tmp_path / "wide")
    with pytest.raises(ValueError):
        reorder_corpus(tmp_path / "wide", tmp_path / "wide_packed", [2, 0, 1], {}, layout="packed")
    assert not (tmp_path / "wide_packed").exists()
