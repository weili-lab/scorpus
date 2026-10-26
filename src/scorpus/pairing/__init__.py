"""Pair source and target cells from any metadata table (pandas or polars).

1. ``make_groups``: cells sharing the values of chosen columns form a group.
2. A pair table: one row per allowed (source group, target group), built with ``cross``,
   ordinary polars filters and ``combine``.
3. ``Sampler``: draws cell rows from the allowed pairs, in step or epoch mode.

``cells`` draws every cell once without pairs, optionally in batches of one group.
``write_order`` gives a row order with every group contiguous and shuffled within, for
rewriting a corpus so that ``Sampler(..., run_length=r)`` reads runs of neighbouring rows.

The sampler returns row positions only; batching, padding and what a pair means for training
are left to the model. This package does not use the rest of scorpus.
"""

from .groups import Groups, combine, cross, describe, links_from, make_groups, src, tgt, write_order
from .sampler import Sampler, cells

__all__ = ["Groups", "Sampler", "cells", "combine", "cross", "describe", "links_from", "make_groups", "src", "tgt", "write_order"]
