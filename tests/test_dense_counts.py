import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from scorpus import concat, from_h5ad
from scorpus.loaders import dense_counts


def test_dense_counts_on_the_corpus_gene_axis(tmp_path):
    """Two datasets with different gene panels: counts land on the shared axis, unmeasured genes are masked."""
    members = {}
    for name, genes, counts in (
        ("a", ["g1", "g2", "g3"], [[1, 0, 2], [0, 3, 0]]),
        ("b", ["g3", "g4"], [[5, 0], [0, 7]]),
    ):
        data = ad.AnnData(sparse.csr_matrix(np.array(counts, dtype=np.int32)),
                          obs=pd.DataFrame(index=[f"{name}{i}" for i in range(2)]), var=pd.DataFrame(index=genes))
        data.write_h5ad(tmp_path / f"{name}.h5ad")
        members[name] = from_h5ad(tmp_path / f"{name}.h5ad", tmp_path / name)
    corpus = concat(members, feature_key={"a": "index", "b": "index"})

    gene_ids = list(corpus.feature_registry.global_feature_ids)
    column = [gene_ids.index(g) for g in ("g1", "g2", "g3", "g4")]
    batch = corpus.expression_reader.read_expression_flat([3, 0, 2])   # b1, a0, b0
    counts, measured = dense_counts(corpus, batch)
    np.testing.assert_array_equal(counts[:, column], [[0, 0, 0, 7], [1, 0, 2, 0], [0, 0, 5, 0]])
    np.testing.assert_array_equal(measured[:, column], [[0, 0, 1, 1], [1, 1, 1, 0], [0, 0, 1, 1]])

    # a gene subset, in the given order
    counts, measured = dense_counts(corpus, batch, genes=[gene_ids.index("g3"), gene_ids.index("g1")])
    np.testing.assert_array_equal(counts, [[0, 0], [2, 1], [5, 0]])
    np.testing.assert_array_equal(measured, [[1, 0], [1, 1], [1, 0]])
