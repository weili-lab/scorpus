import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scorpus import from_h5ad  # importing scorpus registers the chunk-wise helpers

sc = pytest.importorskip("scanpy")
from scanpy._compat import DaskArray  # noqa: E402
from scanpy._utils import axis_nnz, check_nonnegative_integers  # noqa: E402

# Six cells in three row chunks of two (chunk_rows=2); empty rows and columns exercise the counts.
VALUES = [[1, 0, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0], [3, 1, 0, 0], [0, 5, 4, 0], [0, 0, 7, 0]]


def lazy_and_in_memory(tmp_path):
    cells = [f"c{i}" for i in range(len(VALUES))]
    genes = ["g1", "g2", "g3", "g4"]
    obs = pd.DataFrame({"condition": ["ctrl", "p"] * 3}, index=pd.Index(cells, name="barcode"))
    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    path = tmp_path / "a.h5ad"
    ad.AnnData(sparse.csr_matrix(np.asarray(VALUES, dtype=np.int32)), obs=obs, var=var).write_h5ad(path)
    lazy = from_h5ad(path, tmp_path / "a", chunk_rows=2).to_anndata_lazy(chunk_rows=2)
    assert lazy.X.numblocks[0] == 3
    in_memory = ad.AnnData(lazy.X.compute(), obs=lazy.obs.copy(), var=lazy.var.copy())
    return lazy, in_memory


def test_import_scorpus_registers_chunkwise_helpers():
    assert axis_nnz.dispatch(DaskArray).__module__ == "scorpus._scanpy_dask"
    assert check_nonnegative_integers.dispatch(DaskArray).__module__ == "scorpus._scanpy_dask"


def test_helpers_match_in_memory(tmp_path):
    lazy, in_memory = lazy_and_in_memory(tmp_path)
    for axis in (0, 1):
        np.testing.assert_array_equal(
            np.asarray(axis_nnz(lazy.X, axis=axis).compute()).ravel(),
            np.asarray(axis_nnz(in_memory.X, axis=axis)).ravel(),
        )
    assert check_nonnegative_integers(lazy.X).compute()
    # A single non-integer value in the last chunk makes the whole matrix fail the check.
    assert not check_nonnegative_integers(lazy.X * 1.5).compute()


def test_qc_and_ttest_match_in_memory(tmp_path):
    lazy, in_memory = lazy_and_in_memory(tmp_path)
    for adata in (lazy, in_memory):
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    pd.testing.assert_frame_equal(lazy.obs, in_memory.obs, check_dtype=False)
    pd.testing.assert_frame_equal(lazy.var, in_memory.var, check_dtype=False)

    for adata in (lazy, in_memory):
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.tl.rank_genes_groups(adata, "condition", reference="ctrl", method="t-test")
    pd.testing.assert_frame_equal(
        sc.get.rank_genes_groups_df(lazy, group="p"),
        sc.get.rank_genes_groups_df(in_memory, group="p"),
        check_dtype=False,
    )
