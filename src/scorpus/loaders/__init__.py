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
from .adapters import (
    ExpressionBatchDataset,
    CorpusRandomBatchSampler,
    ContextBatchSampler,
    build_loader,
    collate_expression_batch,
)
from .expression import ExpressionBatch
from .feature_registry import (
    FeatureRegistry,
)
from .gene_token_mapper import (
    GeneTokenMapper,
)
from .corpus_loader import (
    Corpus,
    load_corpus,
)
from .composition import concat
from .sparse_batch import (
    SparseBatchProcessor,
)

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
