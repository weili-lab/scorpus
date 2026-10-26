"""Sparse on-disk corpora with optional, in-memory metadata harmonization."""

from .conversion import from_10x_h5_files, from_h5ad
from .loaders.corpus_loader import Corpus, load_corpus
from .loaders.composition import concat
from .materializers.reorder import reorder_corpus
from .staging import estimate, stage, staged

from .contracts import (
    BLUEPRINT,
    CONTRACT_VERSION,
    REQUIRED_ARTIFACTS,
    build_phase1_blueprint,
)

__all__ = [
    "from_h5ad", "from_10x_h5_files", "load_corpus", "concat", "Corpus",
    "estimate", "stage", "staged", "reorder_corpus",
    "BLUEPRINT",
    "CONTRACT_VERSION",
    "REQUIRED_ARTIFACTS",
    "build_phase1_blueprint",
]
