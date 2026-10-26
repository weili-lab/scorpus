import anndata as ad
import h5py
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from scorpus import from_10x_h5_files, from_h5ad


def write_h5ad(path, X, var=None, obsm=None):
    obs = pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])])
    var = pd.DataFrame(index=[f"g{j}" for j in range(X.shape[1])]) if var is None else var
    ad.AnnData(X, obs=obs, var=var, obsm=obsm).write_h5ad(path)
    return path


def test_round_counts(tmp_path):
    values = np.array([[0.4, 1.5, 0], [2.0, 0, 0.5], [0, 0, 0], [3.25, 0.75, 1]])
    path = write_h5ad(tmp_path / "frac.h5ad", sparse.csr_matrix(values))
    corpus = from_h5ad(path, tmp_path / "rounded", layers="X", round_counts=True, chunk_rows=3)
    np.testing.assert_array_equal(corpus.to_anndata().X.toarray(), np.rint(values))
    manifest = yaml.safe_load((tmp_path / "rounded" / "corpus.yaml").read_text())
    assert manifest["expression_kind"] == "rounded_counts"
    assert manifest["fractional_value_share"] == 5 / 7  # of 7 stored nonzeros


def test_feature_mask_keeps_only_selected_modality(tmp_path):
    counts = np.array([[1, 0, 0.37], [0, 4, -1.2], [2, 3, 0.05]])
    var = pd.DataFrame({"modality": ["RNA", "RNA", "protein"]}, index=["A", "B", "CITE-X"])
    path = write_h5ad(tmp_path / "cite.h5ad", sparse.csr_matrix(counts), var=var)
    corpus = from_h5ad(path, tmp_path / "rna", layers="X", features=("modality", "RNA"), chunk_rows=2)
    assert corpus.var.index.tolist() == ["A", "B"]
    np.testing.assert_array_equal(corpus.to_anndata().X.toarray(), counts[:, :2])


def test_extra_matrices_read_back_by_row(tmp_path):
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.poisson(1, (7, 4)))
    guide_umi = sparse.csr_matrix(rng.poisson(0.5, (7, 3)).astype(np.int32))
    hashing = rng.poisson(2, (7, 2)).astype(np.float32)  # dense obsm
    path = write_h5ad(tmp_path / "g.h5ad", X, obsm={"guide_umi": guide_umi, "hto": hashing})
    corpus = from_h5ad(path, tmp_path / "g", chunk_rows=3, extra={
        "guide_umi": ("obsm/guide_umi", ["sg1", "sg2", "sg3"]),
        "hto": ("obsm/hto", ["h1", "h2"]),
    })
    rows = [6, 0, 3, 3]
    matrix, features = corpus.read_extra("guide_umi", rows)
    np.testing.assert_array_equal(matrix.toarray(), guide_umi[rows].toarray())
    assert features.index.tolist() == ["sg1", "sg2", "sg3"]
    np.testing.assert_array_equal(corpus.read_extra("hto", rows)[0].toarray(), hashing[rows])
    np.testing.assert_array_equal(corpus.to_anndata().X.toarray(), X.toarray())


def test_chunks_are_committed_as_one_lance_write(tmp_path):
    import lance

    X = sparse.csr_matrix(np.random.default_rng(0).poisson(1, (7, 4)))
    path = write_h5ad(tmp_path / "g.h5ad", X, obsm={"guide_umi": X[:, :3].astype(np.int32)})
    from_h5ad(path, tmp_path / "g", chunk_rows=3, extra={"guide_umi": ("obsm/guide_umi", ["a", "b", "c"])})
    for table in ["expression.lance", "extra/guide_umi.lance"]:
        dataset = lance.dataset(tmp_path / "g" / table)
        # 3 chunks, but one commit: a single version and fragment
        assert (dataset.version, len(dataset.get_fragments()), dataset.count_rows()) == (1, 1, 7)


def test_csc_source_is_regrouped_by_rows(tmp_path):
    rng = np.random.default_rng(1)
    values = rng.poisson(0.7, (11, 5)).astype(np.float32)
    path = write_h5ad(tmp_path / "csc.h5ad", sparse.csc_matrix(values))
    corpus = from_h5ad(path, tmp_path / "csc", layers="X", chunk_rows=4, csc_tmp_dir=tmp_path / "tmp")
    np.testing.assert_array_equal(corpus.to_anndata().X.toarray(), values)
    assert not (tmp_path / "tmp" / "csc_row_buckets").exists()


def write_10x_h5(path, cells_by_features, feature_ids, feature_types):
    """A minimal CellRanger h5: features x barcodes stored CSC (one slice per cell)."""
    by_cell = sparse.csr_matrix(cells_by_features)
    with h5py.File(path, "w") as handle:
        m = handle.create_group("matrix")
        m["data"], m["indices"], m["indptr"] = by_cell.data.astype(np.int32), by_cell.indices, by_cell.indptr
        m["shape"] = np.array([by_cell.shape[1], by_cell.shape[0]])
        m["barcodes"] = np.array([f"b{i}".encode() for i in range(by_cell.shape[0])])
        m["features/id"] = np.array([f.encode() for f in feature_ids])
        m["features/feature_type"] = np.array([t.encode() for t in feature_types])


def test_10x_files_map_genes_by_id(tmp_path):
    rng = np.random.default_rng(2)
    gene_ids, guide_ids = ["G1", "G2", "G3"], ["sgA", "sgB"]
    types = ["Gene Expression"] * 3 + ["CRISPR Guide Capture"] * 2
    # Per file: counts keyed by feature id; files list features in different orders.
    counts = {name: {f: rng.poisson(2, n) for f in gene_ids + guide_ids} for name, n in [("a", 5), ("b", 4)]}
    order_a = ["G1", "G2", "G3", "sgA", "sgB"]
    order_b = ["G3", "G1", "G2", "sgB", "sgA"]
    write_10x_h5(tmp_path / "a.h5", np.column_stack([counts["a"][f] for f in order_a]), order_a, types)
    write_10x_h5(tmp_path / "b.h5", np.column_stack([counts["b"][f] for f in order_b]), order_b, types)
    recovered = rng.poisson(2, (3, 3))
    var = pd.DataFrame({"gene_ids": ["G2", "G3", "G1"]}, index=["s2", "s3", "s1"])
    write_h5ad(tmp_path / "r.h5ad", sparse.csr_matrix(recovered.astype(np.int32)), var=var)
    files = {"a": tmp_path / "a.h5", "b": tmp_path / "b.h5", "r": tmp_path / "r.h5ad"}
    cells = pd.DataFrame({
        "file_id": ["a", "a", "b", "b", "a", "r", "r"],
        "local_row": [0, 3, 1, 2, 4, 2, 0],
        "label": list("abcdefg"),
    })
    genes = pd.DataFrame({"symbol": ["s1", "s2", "s3"]}, index=pd.Index(gene_ids, name="gene_id"))
    guides = pd.DataFrame({"target": ["T1", "T2"]}, index=pd.Index(guide_ids, name="guide_id"))
    corpus = from_10x_h5_files(files, cells, tmp_path / "multi", genes=genes, guides=guides, chunk_rows=3)

    expected_genes, expected_guides = [], []
    for file_id, row in zip(cells.file_id, cells.local_row):
        if file_id == "r":
            expected_genes.append(recovered[row][[2, 0, 1]])
            expected_guides.append([0, 0])
        else:
            expected_genes.append([counts[file_id][g][row] for g in gene_ids])
            expected_guides.append([counts[file_id][g][row] for g in guide_ids])
    np.testing.assert_array_equal(corpus.to_anndata().X.toarray(), expected_genes)
    guide_umi, features = corpus.read_extra("guide_umi", range(len(cells)))
    np.testing.assert_array_equal(guide_umi.toarray(), expected_guides)
    pd.testing.assert_frame_equal(features, guides)
    pd.testing.assert_frame_equal(corpus.obs, cells)
    pd.testing.assert_frame_equal(corpus.var, genes)
