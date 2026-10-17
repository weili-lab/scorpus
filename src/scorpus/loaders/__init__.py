"""Public model-independent corpus loading and sparse processing APIs."""

from .expression import (
    AggregateLanceReader,
    AggregateZarrReader,
    BaseExpressionReader,
    DatasetEntry,
    ExpressionReader,
    FederatedLanceReader,
    FederatedZarrReader,
    LanceDatasetEntry,
    ZarrDatasetEntry,
    build_expression_reader,
)
from .zarr_reading import ZARR_READ_ENGINES, normalize_zarr_read_engine, open_csr_arrays
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
    "AggregateZarrReader",
    "FederatedZarrReader",
    "LanceDatasetEntry",
    "ZarrDatasetEntry",
    "build_expression_reader",
    "ZARR_READ_ENGINES",
    "normalize_zarr_read_engine",
    "open_csr_arrays",
    # Phase 3 — Core types
    "ExpressionBatch",
    "dense_counts",
    # Phase 2 — Feature Registry
    "FeatureRegistry",
    # Phase N — Corpus loader factory
    "Corpus",
    "load_corpus",
]
