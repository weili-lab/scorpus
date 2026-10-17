"""Pair source and target cells from any metadata table (pandas or polars).

1. ``make_groups``: cells sharing the values of chosen columns form a group.
2. A pair table: one row per allowed (source group, target group), built with ``cross``,
   ordinary polars filters and ``combine``.
3. ``Sampler``: draws cell rows from the allowed pairs, in step or epoch mode.

The sampler returns row positions only; batching, padding and what a pair means for training
are left to the model. This package does not use the rest of scorpus.
"""

from .groups import Groups, combine, cross, describe, links_from, make_groups, src, tgt
from .sampler import Sampler

__all__ = ["Groups", "Sampler", "combine", "cross", "describe", "links_from", "make_groups", "src", "tgt"]
