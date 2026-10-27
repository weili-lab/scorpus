import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scorpus import concat, from_h5ad

cp = pytest.importorskip("cupy")
cupyx_sparse = pytest.importorskip("cupyx.scipy.sparse")
try:
    gpu_count = cp.cuda.runtime.getDeviceCount()
except cp.cuda.runtime.CUDARuntimeError:
    gpu_count = 0
pytestmark = pytest.mark.skipif(gpu_count == 0, reason="needs a CUDA GPU")

# Consecutive empty rows and a trailing empty row exercise the CSR row offsets.
VALUES = [[1, 0, 2], [0, 0, 0], [0, 0, 0], [3, 1, 0], [0, 5, 4], [0, 0, 0]]


def make_corpus(tmp_path, name, genes=("g1", "g2", "g3")):
    cells = [f"{name}{i}" for i in range(len(VALUES))]
    obs = pd.DataFrame({"condition": ["ctrl", "p"] * 3}, index=pd.Index(cells, name="barcode"))
    var = pd.DataFrame({"symbol": list(genes)}, index=pd.Index(genes, name="gene"))
    path = tmp_path / f"{name}.h5ad"
    ad.AnnData(sparse.csr_matrix(np.asarray(VALUES, dtype=np.int32)), obs=obs, var=var).write_h5ad(path)
    return from_h5ad(path, tmp_path / name, chunk_rows=2)


def assert_gpu_matches_cpu(corpus, **kwargs):
    cpu = corpus.to_anndata_lazy(chunk_rows=2, **kwargs)
    gpu = corpus.to_anndata_lazy(chunk_rows=2, device="cuda", **kwargs)
    computed = gpu.X.compute()
    assert isinstance(computed, cupyx_sparse.csr_matrix)
    assert computed.dtype == np.float32
    np.testing.assert_array_equal(computed.get().toarray(), cpu.X.compute().toarray())
    pd.testing.assert_frame_equal(gpu.obs, cpu.obs)
    pd.testing.assert_frame_equal(gpu.var, cpu.var)


def test_standalone_gpu_export_matches_cpu(tmp_path):
    assert_gpu_matches_cpu(make_corpus(tmp_path, "a"))


@pytest.mark.parametrize("var_join", ["inner", "outer"])
def test_concat_gpu_export_matches_cpu(tmp_path, var_join):
    view = concat(
        {"a": make_corpus(tmp_path, "a"), "b": make_corpus(tmp_path, "b", genes=("g3", "g1", "g4"))},
        feature_key={"a": "symbol", "b": "symbol"},
    )
    assert_gpu_matches_cpu(view, var_join=var_join)



def test_gpu_chunks_match_plain_csr(tmp_path):
    """Every CuPy chunk holds the plain CSR of its rows: values (as float32) in stored gene order, int32 indices."""
    a = make_corpus(tmp_path, "a")
    view = concat({"a": a, "b": make_corpus(tmp_path, "b", genes=("g3", "g1", "g4"))}, feature_key={"a": "symbol", "b": "symbol"})
    # Output column of each member gene: outer axis g1 g2 g3 g4, inner axis g1 g3 (-1: off the inner-join axis).
    cases = [(a, {}, [[0, 1, 2]]), (view, {"var_join": "outer"}, [[0, 1, 2], [2, 0, 3]]),
             (view, {"var_join": "inner"}, [[0, -1, 1], [1, 0, -1]])]
    for corpus, kwargs, member_columns in cases:
        x = corpus.to_anndata_lazy(chunk_rows=4, device="cuda", **kwargs).X
        chunks = [x.blocks[i].compute() for i in range(x.numblocks[0])]
        expected = [plain_csr(VALUES[start:start + 4], columns) for columns in member_columns for start in (0, 4)]
        assert len(chunks) == len(expected)
        for chunk, (data, indices, indptr) in zip(chunks, expected):
            assert (chunk.data.dtype, chunk.indices.dtype, chunk.indptr.dtype) == (np.float32, np.int32, np.int32)
            np.testing.assert_array_equal(chunk.data.get(), data)
            np.testing.assert_array_equal(chunk.indices.get(), indices)
            np.testing.assert_array_equal(chunk.indptr.get(), indptr)


def plain_csr(values, columns):
    """CSR built the plain way: each row's nonzero values in stored gene order, at
    their output column. Column -1 marks a gene the inner-join mapping leaves off the axis; it is dropped."""
    data, indices, indptr = [], [], [0]
    for row in values:
        for value, column in zip(row, columns):
            if value and column >= 0:
                data.append(value)
                indices.append(column)
        indptr.append(len(data))
    return data, indices, indptr
