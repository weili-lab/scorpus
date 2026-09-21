import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import sparse

from perturb_data_lab import concat, from_h5ad, load_corpus


def write_data(tmp_path, name="input", values=None, genes=("g1", "g2", "g3")):
    if values is None:
        values = [[1, 0, 2], [0, 0, 0], [3, 1, 0], [0, 0, 0]]
    obs = pd.DataFrame({
        "condition": pd.Categorical(["ctrl", "p", "p", "ctrl"]),
        "numeric": np.arange(4, dtype=np.int16),
        "dataset_id": ["original"] * 4,
    }, index=pd.Index(["c1", "c2", "c3", "c4"], name="barcode"))
    var = pd.DataFrame({"symbol": list(genes)}, index=pd.Index(genes, name="gene"))
    data = ad.AnnData(sparse.csr_matrix(values), obs=obs, var=var)
    path = tmp_path / f"{name}.h5ad"
    data.write_h5ad(path)
    return path, data


def test_original_metadata_and_lazy_roundtrip(tmp_path):
    path, data = write_data(tmp_path)
    corpus = from_h5ad(path, tmp_path / "corpus", chunk_rows=2)
    assert not (tmp_path / "corpus" / "canonical-obs.parquet").exists()
    pd.testing.assert_frame_equal(corpus.obs, data.obs)
    pd.testing.assert_frame_equal(corpus.var, data.var)
    loaded = load_corpus(tmp_path / "corpus")
    eager = loaded.to_anndata()
    lazy = loaded.to_anndata_lazy(chunk_rows=2)
    np.testing.assert_array_equal(eager.X.toarray(), data.X.toarray())
    np.testing.assert_array_equal(lazy.X.compute().toarray(), data.X.toarray())
    pd.testing.assert_frame_equal(lazy.obs, data.obs)
    pd.testing.assert_frame_equal(lazy.var, data.var)
    output = tmp_path / "export.h5ad"
    loaded.write_h5ad(output, chunk_rows=2)
    backed = ad.read_h5ad(output, backed="r")
    try:
        assert backed.isbacked
        np.testing.assert_array_equal(backed.X[:].toarray(), data.X.toarray())
        pd.testing.assert_frame_equal(backed.obs, data.obs)
    finally:
        backed.file.close()
    reordered = loaded.to_anndata(global_row_indices=[3, 0, 1, 0])
    np.testing.assert_array_equal(reordered.X.toarray(), data.X[[3, 0, 1, 0]].toarray())
    with pytest.raises(FileExistsError):
        from_h5ad(path, tmp_path / "corpus")


def test_recovery_is_explicit(tmp_path):
    counts = np.array([[1, 2, 0], [0, 1, 4], [3, 1, 0], [1, 0, 0]])
    values = np.log1p(counts * np.array([2.5, 0.4, 1.7, 2.2])[:, None])
    path, _ = write_data(tmp_path, values=values)
    with pytest.raises(ValueError, match="raw count possible.*attempt_conversion=True"):
        from_h5ad(path, tmp_path / "unapproved")
    assert not (tmp_path / "unapproved").exists()
    corpus = from_h5ad(path, tmp_path / "approved", attempt_conversion=True, chunk_rows=2)
    np.testing.assert_array_equal(corpus.to_anndata().X.toarray(), counts)
    assert set(corpus.expression_kinds.values()) == {"recovered_counts"}


def test_explicit_non_count_preserves_floats(tmp_path):
    values = np.array([[0.1, -0.7, 0], [0, 0, 0], [0.125, 3.14, 0], [0, 0, 0]], dtype=np.float64)
    path, _ = write_data(tmp_path, values=values)
    with pytest.raises(ValueError, match="no raw count found"):
        from_h5ad(path, tmp_path / "bad")
    with pytest.raises(ValueError, match="requires layers"):
        from_h5ad(path, tmp_path / "bad", no_count=True)
    corpus = from_h5ad(path, tmp_path / "floats", layers="X", no_count=True, chunk_rows=2)
    actual = corpus.to_anndata_lazy(chunk_rows=2).X.compute()
    assert actual.dtype == values.dtype
    np.testing.assert_array_equal(actual.toarray(), values)
    assert "size_factor" not in corpus.metadata_index.df.columns


def test_raw_uses_its_own_feature_metadata(tmp_path):
    path, data = write_data(tmp_path)
    data.raw = data.copy()
    subset = data[:, :2].copy()
    subset.X = np.log1p(subset.X * 2.5)
    subset.write_h5ad(path)
    corpus = from_h5ad(path, tmp_path / "raw")
    assert len(corpus.var) == 3
    assert yaml.safe_load((tmp_path / "raw" / "corpus.yaml").read_text())["matrix_source"] == "raw"
    np.testing.assert_array_equal(corpus.to_anndata().X.toarray(), data.X.toarray())


def test_composition_alignment_and_no_copy(tmp_path):
    path_a, data_a = write_data(tmp_path, "a")
    path_b, data_b = write_data(tmp_path, "b", genes=("g3", "g1", "g4"))
    a = from_h5ad(path_a, tmp_path / "a", chunk_rows=2)
    b = from_h5ad(path_b, tmp_path / "b", chunk_rows=2)
    original_a = a.obs.copy(deep=True)
    unaligned = concat({"a": a, "b": b})
    assert unaligned.obs.shape == (8, 0)
    with pytest.raises(ValueError, match="feature_key"):
        unaligned.to_anndata_lazy(var_join="outer")
    view = concat(
        {"a": a, "b": b},
        obs_columns={"perturbation": {"a": "condition", "b": "condition"}},
        feature_key={"a": "symbol", "b": "symbol"},
        var_columns={"symbol": {"a": "symbol", "b": "symbol"}},
    )
    assert view.members["a"] is a
    assert list(view.obs.columns) == ["perturbation"]
    assert view.var.index.tolist() == ["g1", "g2", "g3", "g4"]
    assert view.var["symbol"].tolist() == ["g1", "g2", "g3", "g4"]
    result = view.to_anndata_lazy(var_join="outer", chunk_rows=2)
    expected = np.zeros((8, 4), dtype=np.int32)
    expected[:4, :3] = data_a.X.toarray()
    expected[4:, [2, 0, 3]] = data_b.X.toarray()
    np.testing.assert_array_equal(result.X.compute().toarray(), expected)
    np.testing.assert_array_equal(result.varm["feature_presence"], [[1, 1], [1, 0], [1, 1], [0, 1]])
    inner = view.to_anndata(var_join="inner")
    np.testing.assert_array_equal(inner.X.toarray(), expected[:, [0, 2]])
    batch = view.expression_reader.read_expression_flat([6, 0, 7, 6])
    np.testing.assert_array_equal(batch.global_row_index, [6, 0, 7, 6])
    pd.testing.assert_frame_equal(a.obs, original_a)
    with pytest.raises(ValueError, match="identical ordered"):
        view.to_anndata()
    bad_var = b.var.copy()
    bad_var["symbol"] = ["g1"] * 3
    with pytest.raises(ValueError, match="duplicate mapped"):
        concat({"a": a, "b": b}, var={"a": a.var, "b": bad_var}, feature_key={"a": "symbol", "b": "symbol"})
    with pytest.raises(ValueError, match="identity/order"):
        concat({"a": a, "b": b}, obs={"a": a.obs.iloc[::-1], "b": b.obs})


def test_cli_conversion(tmp_path, monkeypatch):
    from perturb_data_lab.cli import main
    path, _ = write_data(tmp_path)
    monkeypatch.setattr("sys.argv", ["pdl", "convert", "--source", str(path), "--output", str(tmp_path / "cli"), "--chunk-rows", "2"])
    main()
    assert load_corpus(tmp_path / "cli").obs.shape == (4, 3)


def test_explicit_zero_counts_and_runtime_metadata(tmp_path):
    path, _ = write_data(tmp_path, values=np.zeros((4, 3), dtype=np.int32))
    corpus = from_h5ad(path, tmp_path / "zeros", layers="X", chunk_rows=2)
    assert not corpus.to_anndata().X.nnz
    corpus.add_obs_meta(
        pd.DataFrame({"global_row_index": [2, 0, 3, 1], "new_label": ["c", "a", "d", "b"]}),
        on=["global_row_index"],
    )
    assert corpus.obs["new_label"].tolist() == ["a", "b", "c", "d"]
    assert corpus.to_anndata_lazy().obs["new_label"].tolist() == ["a", "b", "c", "d"]
    assert "new_label" not in load_corpus(tmp_path / "zeros").obs


def test_full_scan_rejects_unsampled_bad_counts(tmp_path):
    values = np.ones((100, 2), dtype=np.float64)
    sampled = set(np.linspace(0, 99, 32, dtype=int))
    bad_row = next(i for i in range(100) if i not in sampled)
    values[bad_row, 0] = -1
    path = tmp_path / "bad.h5ad"
    ad.AnnData(sparse.csr_matrix(values)).write_h5ad(path)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        from_h5ad(path, tmp_path / "bad_corpus", chunk_rows=7)
    assert not (tmp_path / "bad_corpus" / "corpus.yaml").exists()
