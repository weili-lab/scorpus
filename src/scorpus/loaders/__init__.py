"""Public model-independent corpus loading and sparse processing APIs."""

import importlib

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

# These build model input with torch. They are imported on first use, so the rest of scorpus
# works without torch (install it with ``pip install scorpus[torch]``).
_TORCH_EXPORTS = {
    "ExpressionBatchDataset": ".adapters",
    "CorpusRandomBatchSampler": ".adapters",
    "ContextBatchSampler": ".adapters",
    "build_loader": ".adapters",
    "collate_expression_batch": ".adapters",
    "GeneTokenMapper": ".gene_token_mapper",
    "SparseBatchProcessor": ".sparse_batch",
}


def __getattr__(name):
    if name in _TORCH_EXPORTS:
        return getattr(importlib.import_module(_TORCH_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
    # Phase 3 — Samplers (MetadataIndex-backed)
    "CorpusRandomBatchSampler",
    "ContextBatchSampler",
    # Phase 3 — Data loaders
    "ExpressionBatchDataset",
    "build_loader",
    "collate_expression_batch",
    # Phase 2 — Feature Registry
    "FeatureRegistry",
    "GeneTokenMapper",
    # Phase 3 — sparse batch processing
    "SparseBatchProcessor",
    # Phase N — Corpus loader factory
    "Corpus",
    "load_corpus",
]
