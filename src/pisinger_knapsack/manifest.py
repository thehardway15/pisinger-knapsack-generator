"""Dataset manifest: checksums and integrity verification over instance files."""

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from .constants import SCHEMA_VERSION
from .helper import dict_to_text
from .instance import KnapsackInstance
from .serialization import load_instance, read_metadata, to_dict


def _compute_sha256(data: bytes) -> str:
    """Return the ``"sha256:<hex>"`` digest of ``data``."""
    return "sha256:" + sha256(data).hexdigest()


def build_manifest(files: Mapping[str, str | Path]) -> dict[str, Any]:
    """Build a manifest indexing a set of instance files.

    Args:
        files: Mapping of ``instance_id`` to the path of each instance file. The
            caller chooses the ids (the CLI uses the file stem).

    Returns:
        A manifest dict ``{"schema_version", "instances": [...]}`` whose entries
        are sorted by ``instance_id``. Each entry carries ``instance_id``, ``n``,
        ``R``, ``correlation_type``, ``capacity`` and a ``checksum`` -- plus the
        file's ``metadata`` when present.

    Raises:
        ValueError: If a path does not point to a file.
    """
    instances = []
    for name, path in files.items():
        if not isinstance(path, Path):
            path = Path(path)
        if not path.is_file():
            raise ValueError(f"Path must be a file, got {path}")

        instance = load_instance(path)
        metadata: dict[str, Any] | None = read_metadata(path)
        if metadata is not None and len(metadata.keys()) == 0:
            metadata = None

        data = {
            "instance_id": name,
            "capacity": instance.capacity,
            "R": instance.R,
            "n": instance.n,
            "correlation_type": instance.correlation_type.value,
            "checksum": instance_checksum(instance, metadata=metadata),
        }

        if metadata is not None:
            data["metadata"] = metadata
        instances.append(data)

    instances.sort(key=lambda x: str(x["instance_id"]))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "instances": instances,
    }

    return manifest


def file_checksum(path: str | Path) -> str:
    """Return the ``"sha256:<hex>"`` checksum of a file's raw bytes.

    Args:
        path: Path to the file to hash.

    Returns:
        The prefixed SHA-256 digest.

    Raises:
        ValueError: If ``path`` does not point to a file.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_file():
        raise ValueError(f"Path must be a file, got {path}")

    return _compute_sha256(path.read_bytes())


def instance_checksum(instance: KnapsackInstance, *, metadata: dict[str, Any] | None = None) -> str:
    """Return the ``"sha256:<hex>"`` checksum of an instance's canonical serialization.

    For a file written by :func:`~pisinger_knapsack.save_instance` with the same
    ``metadata`` this equals :func:`file_checksum` of that file.

    Args:
        instance: The instance to hash.
        metadata: Optional provenance, matching what was (or will be) saved.

    Returns:
        The prefixed SHA-256 digest.
    """
    data = to_dict(instance, metadata=metadata)
    json_str = dict_to_text(data)
    return _compute_sha256(json_str.encode("utf-8"))


def read_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load a manifest written by :func:`write_manifest`.

    Args:
        manifest_path: Path to the manifest JSON file.

    Returns:
        The parsed manifest dict.

    Raises:
        ValueError: If ``manifest_path`` does not point to a file.
    """
    if not isinstance(manifest_path, Path):
        manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise ValueError(f"Path must be a file, got {manifest_path}")

    manifest_json = manifest_path.read_text(encoding="utf-8")
    manifest: dict[str, Any] = json.loads(manifest_json)
    return manifest


def verify_manifest(manifest: dict[str, Any], files: Mapping[str, str | Path]) -> list[str]:
    """Check instance files against the checksums recorded in a manifest.

    Args:
        manifest: A manifest as returned by :func:`build_manifest` or
            :func:`read_manifest`.
        files: Mapping of ``instance_id`` to the path to verify.

    Returns:
        The ids whose file is missing from ``files`` or whose recomputed
        checksum no longer matches the manifest. An empty list means the set is
        intact.

    Raises:
        ValueError: If the manifest's ``schema_version`` is unsupported.
    """
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version: {manifest['schema_version']}")

    instances_id_from_manifest = {instance["instance_id"] for instance in manifest["instances"]}
    instances_id_from_files = set(files.keys())

    missing = instances_id_from_manifest - instances_id_from_files

    for name in instances_id_from_files:
        if name in instances_id_from_manifest:
            instance = next((x for x in manifest["instances"] if x["instance_id"] == name), None)
            if instance is None or file_checksum(files[name]) != instance["checksum"]:
                missing.add(name)

    return list(missing)


def write_manifest(files: Mapping[str, str | Path], manifest_path: str | Path) -> None:
    """Build a manifest for ``files`` and write it to ``manifest_path``.

    The manifest is written as canonical UTF-8 JSON (sorted keys, stable bytes),
    matching the instance serialization format.

    Args:
        files: Mapping of ``instance_id`` to instance file path.
        manifest_path: Destination path for the manifest.
    """
    manifest = build_manifest(files)
    if not isinstance(manifest_path, Path):
        manifest_path = Path(manifest_path)
    manifest_path.write_bytes(dict_to_text(manifest).encode("utf-8"))
