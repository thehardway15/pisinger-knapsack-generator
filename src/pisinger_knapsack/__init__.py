"""pisinger-knapsack — a 0/1 knapsack instance generator based on Pisinger's scheme.

The public API exposes the correlation classes (:class:`CorrelationType`), the
instance data model (:class:`KnapsackInstance`), the instance generator
(:func:`generate_instance`) and the capacity rule (:func:`capacity`).
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "CorrelationType",
    "KnapsackInstance",
    "generate_instance",
    "capacity",
    "to_dict",
    "from_dict",
    "save_instance",
    "load_instance",
    "read_metadata",
    "build_manifest",
    "file_checksum",
    "instance_checksum",
    "read_manifest",
    "verify_manifest",
    "write_manifest",
]


from .capacity import capacity  # noqa: F401
from .correlation import CorrelationType  # noqa: F401
from .generator import generate_instance  # noqa: F401
from .instance import KnapsackInstance  # noqa: F401
from .manifest import (
    build_manifest,
    file_checksum,
    instance_checksum,
    read_manifest,
    verify_manifest,
    write_manifest,
)  # noqa: F401
from .serialization import (  # noqa: F401
    from_dict,
    load_instance,
    read_metadata,
    save_instance,
    to_dict,
)
