"""Dense expression on the corpus gene axis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .expression import ExpressionBatch

if TYPE_CHECKING:
    from .corpus_loader import Corpus


def dense_counts(corpus: Corpus, batch: ExpressionBatch, genes=None) -> tuple[np.ndarray, np.ndarray]:
    """Counts of a sparse batch as a dense matrix on the corpus gene axis.

    genes: global gene IDs (positions in ``corpus.feature_registry.global_feature_ids``) to keep,
    in this order; all genes if None.

    Returns ``counts`` (cells x genes, float32) and ``measured`` (cells x genes, bool): whether
    the cell's dataset measured the gene. An unmeasured gene has count 0, which is not an
    observed zero, so models should ignore it where ``measured`` is False.
    """
    registry = corpus.feature_registry
    genes = np.arange(registry.global_vocab_size) if genes is None else np.asarray(genes, dtype=np.int64)
    dataset = corpus.metadata_index.df["dataset_index"].to_numpy()[batch.global_row_index]

    # Column of every global gene in the output, -1 for genes left out.
    column_of_gene = np.full(registry.global_vocab_size, -1, dtype=np.int64)
    column_of_gene[genes] = np.arange(len(genes))

    cell = np.repeat(np.arange(batch.batch_size), np.diff(batch.row_offsets))
    column = column_of_gene[registry.local_to_global_map[dataset[cell], batch.expressed_gene_indices]]
    kept = column >= 0
    counts = np.zeros((batch.batch_size, len(genes)), dtype=np.float32)
    counts[cell[kept], column[kept]] = batch.expression_counts[kept]
    measured = registry.dataset_has_gene[dataset][:, genes]
    return counts, measured
