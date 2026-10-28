"""Results saved next to a corpus (ex_obs / ex_var) come back on the same cells and genes."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scorpus import concat, from_h5ad, load_corpus, reorder_corpus


def convert(tmp_path, name="corpus", cell_ids=("c1", "c2", "c3", "c4")):
    counts = sparse.csr_matrix(np.arange(24, dtype=np.int32).reshape(4, 6) % 5)
    obs = pd.DataFrame({"condition": ["ctrl", "p", "p", "ctrl"]}, index=pd.Index(list(cell_ids), name="barcode"))
    var = pd.DataFrame({"symbol": [f"S{j}" for j in range(6)]}, index=pd.Index([f"g{j}" for j in range(1, 7)], name="gene"))
    ad.AnnData(counts, obs=obs, var=var).write_h5ad(tmp_path / f"{name}.h5ad")
    return from_h5ad(tmp_path / f"{name}.h5ad", tmp_path / name)


def test_default_and_named_ex_files_reload_on_the_same_cells(tmp_path):
    corpus = convert(tmp_path)
    original_obs, original_var = corpus.obs.copy(), corpus.var.copy()
    corpus.add_obs_meta(pd.DataFrame({"leiden": ["0", "1", "1", "0"]}, index=corpus.obs.index))
    corpus.var["dispersion"] = np.arange(6, dtype=np.float64)
    assert corpus.save_ex_obs() == tmp_path / "corpus" / "ex_obs.parquet"
    corpus.save_ex_var("disp")
    first = corpus.obs.copy()
    corpus.add_obs_meta(pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4]}, index=corpus.obs.index))
    corpus.save_ex_obs("scored")

    pd.testing.assert_frame_equal(load_corpus(tmp_path / "corpus").obs, original_obs)
    pd.testing.assert_frame_equal(load_corpus(tmp_path / "corpus").var, original_var)
    pd.testing.assert_frame_equal(load_corpus(tmp_path / "corpus", ex_obs=True).obs, first)
    loaded = load_corpus(tmp_path / "corpus", ex_obs="scored", ex_var="disp")
    pd.testing.assert_frame_equal(loaded.obs, corpus.obs)
    pd.testing.assert_frame_equal(loaded.var, corpus.var)
    # Loaded ex columns reach the metadata index and the AnnData export.
    assert list(loaded.take_metadata([2], columns=["score"])["score"]) == [0.3]
    assert loaded.to_anndata().obs["leiden"].tolist() == ["0", "1", "1", "0"]

    # Saving the default again overwrites it.
    loaded.save_ex_obs()
    pd.testing.assert_frame_equal(load_corpus(tmp_path / "corpus", ex_obs=True).obs, corpus.obs)
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "corpus", ex_var=True)


def test_add_obs_meta_from_parquet_checks_cell_order_and_column_clashes(tmp_path):
    corpus = convert(tmp_path, cell_ids=("c1", "c1", "c2", "c3"))  # repeated barcodes are fine
    meta = pd.DataFrame({"leiden": ["a", "b", "c", "d"]}, index=corpus.obs.index)
    meta.to_parquet(tmp_path / "meta.parquet")
    corpus.add_obs_meta(tmp_path / "meta.parquet")
    assert corpus.obs["leiden"].tolist() == ["a", "b", "c", "d"]
    assert list(corpus.take_metadata([1], columns=["leiden"])["leiden"]) == ["b"]

    with pytest.raises(ValueError, match="corpus order"):
        corpus.add_obs_meta(meta.iloc[[1, 0, 2, 3]].rename(columns={"leiden": "x"}).set_axis(["c1", "c2", "c1", "c3"]))
    with pytest.raises(ValueError, match="corpus order"):
        corpus.add_obs_meta(meta.iloc[:3].rename(columns={"leiden": "x"}))
    with pytest.raises(ValueError, match="overwrite=True"):
        corpus.add_obs_meta(meta)
    with pytest.raises(ValueError, match="Routing"):
        corpus.add_obs_meta(meta.rename(columns={"leiden": "cell_id"}), overwrite=True)
    corpus.add_obs_meta(meta.assign(leiden=["w", "x", "y", "z"]), overwrite=True)
    assert corpus.obs["leiden"].tolist() == ["w", "x", "y", "z"]
    assert list(corpus.take_metadata([1], columns=["leiden"])["leiden"]) == ["x"]


def test_saving_refuses_rows_that_left_corpus_order(tmp_path):
    corpus = convert(tmp_path)
    name = corpus.dataset_ids[0]
    corpus.obs_frames[name] = corpus.obs.iloc[::-1]
    with pytest.raises(ValueError, match="order"):
        corpus.save_ex_obs()
    corpus.var_frames[name] = corpus.var.iloc[:5]
    with pytest.raises(ValueError, match="order"):
        corpus.save_ex_var()
    assert not list((tmp_path / "corpus").glob("ex_*"))


def test_concat_members_load_ex_files_and_views_refuse_single_corpus_operations(tmp_path):
    a = convert(tmp_path, "a")
    convert(tmp_path, "b")
    a.add_obs_meta(pd.DataFrame({"leiden": ["0", "1", "1", "0"]}, index=a.obs.index))
    a.save_ex_obs("clusters")
    view = concat({
        "a": load_corpus(tmp_path / "a", ex_obs="clusters"), "b": load_corpus(tmp_path / "b"),
    })
    assert view.members["a"].obs["leiden"].tolist() == ["0", "1", "1", "0"]
    assert "leiden" not in view.members["b"].obs
    with pytest.raises(ValueError, match="standalone"):
        view.save_ex_obs()
    with pytest.raises(ValueError, match="standalone"):
        view.add_obs_meta(pd.DataFrame({"x": range(8)}))


def test_reorder_carries_every_ex_file(tmp_path):
    corpus = convert(tmp_path)
    corpus.add_obs_meta(pd.DataFrame({"leiden": ["0", "1", "2", "3"]}, index=corpus.obs.index))
    corpus.save_ex_obs()
    corpus.add_obs_meta(pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4]}, index=corpus.obs.index))
    corpus.save_ex_obs("scored")
    corpus.var["dispersion"] = np.arange(6, dtype=np.float64)
    corpus.save_ex_var()
    corpus.save_ex_var("disp")

    order = np.array([3, 0, 2])  # a filtered rewrite, as from write_order(..., keep=...)
    reorder_corpus(tmp_path / "corpus", tmp_path / "new", order, {"by": "test"})
    for ex_obs in [True, "scored"]:
        source = load_corpus(tmp_path / "corpus", ex_obs=ex_obs)
        pd.testing.assert_frame_equal(load_corpus(tmp_path / "new", ex_obs=ex_obs).obs, source.obs.iloc[order])
    for ex_var in [True, "disp"]:
        pd.testing.assert_frame_equal(load_corpus(tmp_path / "new", ex_var=ex_var).var, corpus.var)


def test_hvg_columns_in_ex_var_fill_the_gene_index(tmp_path):
    corpus = convert(tmp_path)
    # Any HVG choice: only highly_variable is read; a rank column, if present, is ignored.
    corpus.var["highly_variable"] = [True, False, True, False, False, True]
    corpus.save_ex_var("hvg")
    corpus.var["highly_variable_rank"] = [5.0, np.nan, 0.0, 3.0, np.nan, 1.0]
    corpus.save_ex_var("hvg_with_rank")
    expected_mask = np.array([True, False, True, False, False, True])

    for name in ["hvg", "hvg_with_rank"]:
        loaded = load_corpus(tmp_path / "corpus", ex_var=name)
        registry = loaded.feature_registry
        local_to_global = registry.local_to_global_map[0, :6]
        np.testing.assert_array_equal(registry.hvg_mask[0, local_to_global], expected_mask)
        var = loaded.to_anndata().var
        np.testing.assert_array_equal(var["highly_variable"], expected_mask)
        assert "hvg_rank" not in var
    assert not load_corpus(tmp_path / "corpus").feature_registry.hvg_mask.any()

    # In a concat, each member keeps its own HVGs.
    convert(tmp_path, "other")
    view = concat({"other": load_corpus(tmp_path / "other"), "hvg": loaded})
    view_map = view.feature_registry.local_to_global_map
    np.testing.assert_array_equal(view.feature_registry.hvg_mask[1, view_map[1, :6]], expected_mask)
    assert not view.feature_registry.hvg_mask[0].any()
