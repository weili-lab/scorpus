"""Public model-independent corpus loading and sparse processing APIs."""

from .expression import (
    AggregateLanceReader,
    BaseExpressionReader,
    DatasetEntry,
    ExpressionReader,
    FederatedLanceReader,
    LanceDatasetEntry,
    build_expression_reader,
)
from .index import MetadataIndex
from .expression import ExpressionBatch
from .feature_registry import (
    FeatureRegistry,
)
from .corpus_loader import (
    Corpus,
    load_corpus,
)
from .composition import concat
from .dense import dense_counts

__all__ = [
    "concat",
    # Phase 1 — MetadataIndex
    "MetadataIndex",
    # Phase 2 — ExpressionReader (backend-agnostic)
    "ExpressionReader",
    "BaseExpressionReader",
    "DatasetEntry",
    "AggregateLanceReader",
    "FederatedLanceReader",
    "LanceDatasetEntry",
    "build_expression_reader",
    # Phase 3 — Core types
    "ExpressionBatch",
    "dense_counts",
    # Phase 2 — Feature Registry
    "FeatureRegistry",
    # Phase N — Corpus loader factory
    "Corpus",
    "load_corpus",
]
