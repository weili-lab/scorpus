"""Chunk-wise Dask versions of two scanpy helpers, registered when scorpus is imported.

scanpy 1.12's Dask versions of axis_nnz(x, axis=0) and check_nonnegative_integers(x) call
map_blocks(..., drop_axis=<the row axis>). Dropping the chunked row axis makes Dask concatenate
every row chunk into one block before the function runs, so calculate_qc_metrics,
rank_genes_groups and highly_variable_genes(flavor="seurat_v3") hold the whole matrix in memory
on a lazy export. Here each chunk returns a small result and the reduction over chunks happens
afterwards. Registering on scanpy's singledispatch functions changes only the running process.
"""

from functools import partial

import numpy as np
from scanpy._compat import DaskArray  # dask.array.Array, the type scanpy dispatches on
from scanpy._utils import axis_nnz, check_nonnegative_integers


@axis_nnz.register(DaskArray)
def _axis_nnz_chunkwise(x, /, axis):
    if axis == 1:
        # One count per cell: rows stay chunked, so this is already chunk-wise (scanpy's own code).
        return x.map_blocks(partial(axis_nnz, axis=1), dtype=np.int64,
                            meta=np.array([], dtype=np.int64), drop_axis=1)
    # One row of per-gene counts per row chunk (n_row_chunks x n_genes), then summed over chunks.
    per_chunk = x.map_blocks(lambda block: np.asarray(axis_nnz(block, axis=0))[np.newaxis, :],
                             dtype=np.int64, meta=np.empty((0, 0), dtype=np.int64),
                             chunks=((1,) * x.numblocks[0], x.chunks[1]))
    return per_chunk.sum(axis=0)


@check_nonnegative_integers.register(DaskArray)
def _check_nonnegative_integers_chunkwise(x, /):
    # One bool per chunk (n_row_chunks x n_col_chunks), then all().
    per_chunk = x.map_blocks(lambda block: np.array([[check_nonnegative_integers(block)]]),
                             dtype=bool, meta=np.empty((0, 0), dtype=bool), chunks=(1, 1))
    return per_chunk.all()
