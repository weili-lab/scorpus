"""Sparse on-disk corpora with optional, in-memory metadata harmonization."""

import importlib.util

from .conversion import from_10x_h5_files, from_h5ad
from .loaders.corpus_loader import Corpus, load_corpus
from .loaders.composition import concat
from .materializers.reorder import reorder_corpus
from .staging import estimate, stage, staged

from .contracts import CONTRACT_VERSION

# scanpy is optional; when installed, make its Dask QC/DE/seurat_v3 helpers work chunk by chunk
# so a lazy export is never pulled into memory as one block (see _scanpy_dask.py).
if importlib.util.find_spec("scanpy") is not None:
    from . import _scanpy_dask  # noqa: F401

__all__ = [
    "from_h5ad", "from_10x_h5_files", "load_corpus", "concat", "Corpus",
    "estimate", "stage", "staged", "reorder_corpus",
    "CONTRACT_VERSION",
]
