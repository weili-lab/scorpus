"""Deprecated alias for :mod:`scorpus`.

``perturb-data-lab`` was renamed to ``scorpus``. This shim keeps existing
imports working and forwards submodules, so ``import
perturb_data_lab.loaders.corpus_loader`` resolves to the ``scorpus``
module object itself rather than a copy. Update imports to ``scorpus``;
this package will be removed in a future release.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import scorpus as _scorpus

warnings.warn(
    "perturb_data_lab has been renamed to scorpus; import scorpus instead. "
    "This alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Bind the alias to the real package so submodule imports, `from ... import`,
# and isinstance checks all resolve against one set of module objects.
sys.modules[__name__] = _scorpus
for _name in list(sys.modules):
    if _name.startswith("scorpus."):
        sys.modules[__name__ + _name[len("scorpus"):]] = sys.modules[_name]


def __getattr__(name: str):
    return getattr(_scorpus, name)
