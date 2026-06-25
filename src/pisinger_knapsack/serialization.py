"""(De)serialization of knapsack instances to and from canonical JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pisinger_knapsack.constants import SCHEMA_VERSION

from .correlation import CorrelationType
from .helper import dict_to_text
from .instance import KnapsackInstance


def to_dict(
    instance: KnapsackInstance, *, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Convert an instance to a JSON-ready mapping.

    Args:
        instance: The instance to encode.
        metadata: Optional provenance (e.g. ``{"seed": ...}``) stored under a
            ``"metadata"`` key. Omitted entirely when ``None``; the library
            assigns no meaning to its contents.

    Returns:
        A mapping with ``schema_version``, ``n``, ``R``, ``correlation_type``
        (the enum's short string), ``capacity`` and the parallel ``values`` and
        ``weights`` lists of native ``int`` -- plus ``metadata`` when given.
    """
    data = {
        "schema_version": SCHEMA_VERSION,
        "n": instance.n,
        "R": instance.R,
        "correlation_type": instance.correlation_type.value,
        "values": instance.values.tolist(),
        "weights": instance.weights.tolist(),
        "capacity": instance.capacity,
    }
    if metadata is not None:
        data["metadata"] = metadata

    return data


def from_dict(data: dict[str, Any]) -> KnapsackInstance:
    """Reconstruct an instance from a :func:`to_dict` mapping.

    ``values`` and ``weights`` are restored as 1-D NumPy ``int64`` arrays. Any
    ``metadata`` is ignored: the returned object is the pure knapsack problem.

    Args:
        data: A decoded instance mapping.

    Returns:
        The reconstructed :class:`KnapsackInstance`.

    Raises:
        ValueError: If ``values`` and ``weights`` differ in length, ``n`` does
            not match the item count, the ``schema_version`` is unsupported, or
            ``correlation_type`` is not a known class.
        KeyError: If a required key is missing.
    """
    if len(data["values"]) != len(data["weights"]):
        raise ValueError(
            "Length of values and weights must be the same"
            f" (got {len(data['values'])} and {len(data['weights'])})"
        )

    if data["schema_version"] != 1:
        raise ValueError(f"Unsupported schema version: {data['schema_version']}")

    if data["n"] != len(data["values"]):
        raise ValueError(
            f"n must match the length of values, "
            f"got n={data['n']} and len(values)={len(data['values'])}"
        )

    return KnapsackInstance(
        n=data["n"],
        R=data["R"],
        correlation_type=CorrelationType(data["correlation_type"]),
        values=np.array(data["values"], dtype=np.int64),
        weights=np.array(data["weights"], dtype=np.int64),
        capacity=data["capacity"],
    )


def save_instance(
    instance: KnapsackInstance, path: str | Path, metadata: dict[str, Any] | None = None
) -> None:
    """Write an instance to ``path`` as canonical UTF-8 JSON.

    The bytes are written without newline translation, so the same instance and
    metadata yield a byte-identical file on every platform -- which keeps
    checksums reproducible.

    Args:
        instance: The instance to serialize.
        path: Destination file path.
        metadata: Optional provenance recorded in the file (see :func:`to_dict`).
    """
    path = Path(path)
    text = dict_to_text(to_dict(instance, metadata=metadata))
    path.write_bytes(text.encode("utf-8"))


def load_instance(path: str | Path) -> KnapsackInstance:
    """Load an instance previously written by :func:`save_instance`.

    Args:
        path: Path to a JSON instance file.

    Returns:
        The reconstructed :class:`KnapsackInstance` (metadata is not part of it).
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return from_dict(data)


def read_metadata(path: str | Path) -> dict[str, Any]:
    """Return the provenance ``metadata`` recorded in an instance file.

    Args:
        path: Path to a JSON instance file.

    Returns:
        The metadata mapping, or an empty dict if the file has none.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata: dict[str, Any] = data.get("metadata", {})
    return metadata
