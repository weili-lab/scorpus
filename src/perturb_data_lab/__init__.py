"""Sparse on-disk corpora with optional, in-memory metadata harmonization."""

from .conversion import from_h5ad
from .loaders.corpus_loader import Corpus, load_corpus
from .loaders.composition import concat

from .contracts import (
    BLUEPRINT,
    CONTRACT_VERSION,
    REQUIRED_ARTIFACTS,
    build_phase1_blueprint,
)

__all__ = [
    "from_h5ad", "load_corpus", "concat", "Corpus",
    "BLUEPRINT",
    "CONTRACT_VERSION",
    "REQUIRED_ARTIFACTS",
    "build_phase1_blueprint",
]
