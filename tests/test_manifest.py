"""Tests for the dataset manifest and checksums (R1-T6).

Executable specification of the integrity/packaging contract:

    from pisinger_knapsack import (
        file_checksum,
        instance_checksum,
        build_manifest,
        write_manifest,
        read_manifest,
        verify_manifest,
    )

The manifest is a generic *table of contents* over a set of instance files. R1
owns the FORMAT, the checksum algorithm and verification; it stays agnostic of
thesis specifics: instance ids are supplied by the caller (the CLI uses the file
stem), there is no hard-coded variant list, and provenance such as ``seed`` is
carried verbatim from each instance's optional ``metadata`` block -- R1 never
interprets it.

Contract:

* ``file_checksum(path) -> str`` -- ``"sha256:" + hexdigest`` over the raw file
  bytes. Deterministic; byte-identical files share a checksum.
* ``instance_checksum(instance, *, metadata=None) -> str`` -- the same checksum
  computed over the canonical serialization, so it equals ``file_checksum`` of a
  file written by ``save_instance`` with the same ``metadata``.
* ``build_manifest(files) -> dict`` -- ``files`` maps ``instance_id -> path``;
  returns ``{"schema_version": 1, "instances": [...]}`` with entries sorted by
  ``instance_id``. Each entry: ``instance_id, n, R, correlation_type, capacity,
  checksum`` plus ``metadata`` only when the file carries one.
* ``write_manifest`` / ``read_manifest`` -- canonical JSON file (keys sorted,
  trailing newline) so repeated writes are byte-identical.
* ``verify_manifest(manifest, files) -> list[str]`` -- recomputes each entry's
  checksum from ``files`` and returns the ids that mismatch or are missing;
  ``[]`` means the set is intact.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from pisinger_knapsack import (
    CorrelationType,
    KnapsackInstance,
    build_manifest,
    file_checksum,
    generate_instance,
    instance_checksum,
    read_manifest,
    save_instance,
    verify_manifest,
    write_manifest,
)

SEED = 20260101
DEFAULT_R = 1000
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ENTRY_KEYS = {"instance_id", "n", "R", "correlation_type", "capacity", "checksum"}


def make_instance(
    correlation_type: CorrelationType = CorrelationType.UNCORRELATED,
    n: int = 20,
    seed: int = SEED,
) -> KnapsackInstance:
    """Return a deterministic instance for manifest experiments."""
    return generate_instance(
        np.random.default_rng(seed), n=n, correlation_type=correlation_type, R=DEFAULT_R
    )


def save_tmp(
    instance: KnapsackInstance,
    tmp_path: Path,
    name: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write ``instance`` under ``tmp_path`` and return its path."""
    path = tmp_path / f"{name}.json"
    save_instance(instance, path, metadata=metadata)
    return path


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------


def test_file_checksum_has_sha256_prefix_format(tmp_path: Path) -> None:
    path = save_tmp(make_instance(), tmp_path, "a")

    assert SHA256_RE.match(file_checksum(path))


def test_file_checksum_is_deterministic(tmp_path: Path) -> None:
    path = save_tmp(make_instance(), tmp_path, "a")

    assert file_checksum(path) == file_checksum(path)


def test_byte_identical_files_share_checksum(tmp_path: Path) -> None:
    instance = make_instance()
    a = save_tmp(instance, tmp_path, "a")
    b = save_tmp(instance, tmp_path, "b")

    assert a.read_bytes() == b.read_bytes()
    assert file_checksum(a) == file_checksum(b)


def test_different_content_differs_in_checksum(tmp_path: Path) -> None:
    a = save_tmp(make_instance(seed=1), tmp_path, "a")
    b = save_tmp(make_instance(seed=2), tmp_path, "b")

    assert file_checksum(a) != file_checksum(b)


def test_instance_checksum_matches_file_checksum(tmp_path: Path) -> None:
    instance = make_instance()
    path = save_tmp(instance, tmp_path, "a")

    assert instance_checksum(instance) == file_checksum(path)


def test_instance_checksum_matches_file_with_metadata(tmp_path: Path) -> None:
    instance = make_instance()
    metadata = {"seed": SEED}
    path = save_tmp(instance, tmp_path, "a", metadata=metadata)

    assert instance_checksum(instance, metadata=metadata) == file_checksum(path)


def test_metadata_changes_checksum(tmp_path: Path) -> None:
    instance = make_instance()

    assert instance_checksum(instance) != instance_checksum(instance, metadata={"seed": 1})


# ---------------------------------------------------------------------------
# build_manifest: shape and entries
# ---------------------------------------------------------------------------


def test_build_manifest_top_level_shape(tmp_path: Path) -> None:
    files = {"a": save_tmp(make_instance(), tmp_path, "a")}

    manifest = build_manifest(files)

    assert manifest["schema_version"] == 1
    assert isinstance(manifest["instances"], list)
    assert len(manifest["instances"]) == 1


def test_build_manifest_entry_fields(tmp_path: Path) -> None:
    instance = make_instance(correlation_type=CorrelationType.WEAKLY_CORRELATED, n=30)
    path = save_tmp(instance, tmp_path, "n30_weakly")

    (entry,) = build_manifest({"n30_weakly": path})["instances"]

    assert set(entry.keys()) == ENTRY_KEYS  # no metadata key when file has none
    assert entry["instance_id"] == "n30_weakly"
    assert entry["n"] == 30
    assert entry["R"] == DEFAULT_R
    assert entry["correlation_type"] == "weakly"
    assert entry["capacity"] == instance.capacity
    assert entry["checksum"] == file_checksum(path)


def test_build_manifest_sorts_entries_by_instance_id(tmp_path: Path) -> None:
    files = {
        "c": save_tmp(make_instance(seed=3), tmp_path, "c"),
        "a": save_tmp(make_instance(seed=1), tmp_path, "a"),
        "b": save_tmp(make_instance(seed=2), tmp_path, "b"),
    }

    manifest = build_manifest(files)

    ids = [entry["instance_id"] for entry in manifest["instances"]]
    assert ids == ["a", "b", "c"]


def test_build_manifest_carries_metadata_when_present(tmp_path: Path) -> None:
    path = save_tmp(make_instance(), tmp_path, "a", metadata={"seed": 123, "note": "x"})

    (entry,) = build_manifest({"a": path})["instances"]

    assert entry["metadata"] == {"seed": 123, "note": "x"}


def test_build_manifest_omits_metadata_when_absent(tmp_path: Path) -> None:
    path = save_tmp(make_instance(), tmp_path, "a")

    (entry,) = build_manifest({"a": path})["instances"]

    assert "metadata" not in entry


# ---------------------------------------------------------------------------
# write / read round-trip and canonical form
# ---------------------------------------------------------------------------


def test_manifest_file_round_trip(tmp_path: Path) -> None:
    files = {"a": save_tmp(make_instance(seed=1), tmp_path, "a")}
    manifest_path = tmp_path / "manifest.json"

    write_manifest(files, manifest_path)

    assert read_manifest(manifest_path) == build_manifest(files)


def test_repeated_manifest_writes_are_byte_identical(tmp_path: Path) -> None:
    files = {"a": save_tmp(make_instance(), tmp_path, "a")}
    first = tmp_path / "m1.json"
    second = tmp_path / "m2.json"

    write_manifest(files, first)
    write_manifest(files, second)

    assert first.read_bytes() == second.read_bytes()


def test_manifest_keys_are_sorted_and_newline_terminated(tmp_path: Path) -> None:
    files = {"a": save_tmp(make_instance(), tmp_path, "a")}
    manifest_path = tmp_path / "manifest.json"

    write_manifest(files, manifest_path)
    raw = manifest_path.read_bytes()
    parsed = json.loads(raw.decode("utf-8"))

    assert list(parsed.keys()) == sorted(parsed.keys())
    assert raw.endswith(b"\n")


# ---------------------------------------------------------------------------
# verify_manifest
# ---------------------------------------------------------------------------


def test_verify_manifest_passes_for_intact_set(tmp_path: Path) -> None:
    files = {
        "a": save_tmp(make_instance(seed=1), tmp_path, "a"),
        "b": save_tmp(make_instance(seed=2), tmp_path, "b"),
    }
    manifest = build_manifest(files)

    assert verify_manifest(manifest, files) == []


def test_verify_manifest_detects_tampered_file(tmp_path: Path) -> None:
    files = {
        "a": save_tmp(make_instance(seed=1), tmp_path, "a"),
        "b": save_tmp(make_instance(seed=2), tmp_path, "b"),
    }
    manifest = build_manifest(files)
    # Corrupt one file after the manifest was built.
    files["b"].write_text("{}\n", encoding="utf-8")

    assert verify_manifest(manifest, files) == ["b"]


def test_verify_manifest_detects_missing_file(tmp_path: Path) -> None:
    files = {
        "a": save_tmp(make_instance(seed=1), tmp_path, "a"),
        "b": save_tmp(make_instance(seed=2), tmp_path, "b"),
    }
    manifest = build_manifest(files)
    del files["b"]  # 'b' listed in the manifest but no longer provided

    assert verify_manifest(manifest, files) == ["b"]
