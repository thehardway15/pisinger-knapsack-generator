"""Tests for instance serialization (R1-T5).

Executable specification of the output-format contract:

    from pisinger_knapsack import (
        to_dict,
        from_dict,
        save_instance,
        load_instance,
    )

* ``to_dict(instance) -> dict`` -- a JSON-ready mapping with exactly the keys
  ``schema_version``, ``n``, ``R``, ``correlation_type``, ``capacity``,
  ``values`` and ``weights``. ``correlation_type`` is the enum's short string
  value; ``values``/``weights`` are two parallel lists of native ``int``.
* ``from_dict(data) -> KnapsackInstance`` -- the inverse mapping. ``values`` and
  ``weights`` are restored as 1-D ``numpy`` ``int64`` arrays; an unknown
  correlation type, a missing key, an inconsistent length, or an unsupported
  ``schema_version`` raise ``ValueError`` (a missing key may surface as
  ``KeyError``).
* ``save_instance`` / ``load_instance`` -- thin JSON file wrappers. The written
  file is canonical (keys sorted, trailing newline) so repeated writes are
  byte-identical, supporting stable checksums in R1-T6.
* Round-trip: ``from_dict(to_dict(x)) == x`` and ``load(save(x)) == x``,
  exact down to the integer arrays (relies on ``KnapsackInstance.__eq__``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pisinger_knapsack import (
    CorrelationType,
    KnapsackInstance,
    from_dict,
    generate_instance,
    load_instance,
    read_metadata,
    save_instance,
    to_dict,
)

SEED = 20260101
DEFAULT_R = 1000
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_KEYS = {
    "schema_version",
    "n",
    "R",
    "correlation_type",
    "capacity",
    "values",
    "weights",
}


def make_instance(
    correlation_type: CorrelationType = CorrelationType.UNCORRELATED,
    n: int = 20,
    seed: int = SEED,
) -> KnapsackInstance:
    """Return a deterministic instance to serialize."""
    return generate_instance(
        np.random.default_rng(seed), n=n, correlation_type=correlation_type, R=DEFAULT_R
    )


# ---------------------------------------------------------------------------
# to_dict: shape and field encoding
# ---------------------------------------------------------------------------


def test_to_dict_has_exactly_the_expected_keys() -> None:
    data = to_dict(make_instance())

    assert set(data.keys()) == EXPECTED_KEYS


def test_to_dict_schema_version_is_one() -> None:
    data = to_dict(make_instance())

    assert data["schema_version"] == EXPECTED_SCHEMA_VERSION


@pytest.mark.parametrize("correlation_type", list(CorrelationType))
def test_to_dict_encodes_correlation_type_as_short_string(
    correlation_type: CorrelationType,
) -> None:
    data = to_dict(make_instance(correlation_type=correlation_type))

    assert data["correlation_type"] == correlation_type.value
    assert isinstance(data["correlation_type"], str)


def test_to_dict_scalars_are_native_ints() -> None:
    data = to_dict(make_instance())

    for key in ("schema_version", "n", "R", "capacity"):
        assert type(data[key]) is int  # noqa: E721 -- np.int64 must be rejected


def test_to_dict_items_are_lists_of_native_ints() -> None:
    data = to_dict(make_instance())

    assert isinstance(data["values"], list)
    assert isinstance(data["weights"], list)
    # np.int64 is *not* an int instance; the encoding must use plain ints.
    assert all(type(v) is int for v in data["values"])  # noqa: E721
    assert all(type(w) is int for w in data["weights"])  # noqa: E721


def test_to_dict_items_match_instance_arrays() -> None:
    instance = make_instance(n=30)
    data = to_dict(instance)

    assert len(data["values"]) == instance.n
    assert len(data["weights"]) == instance.n
    assert data["values"] == instance.values.tolist()
    assert data["weights"] == instance.weights.tolist()


def test_to_dict_output_is_json_serializable() -> None:
    # Must not raise: native ints/lists only, no ndarray or np.int64 leaking out.
    json.dumps(to_dict(make_instance()))


# ---------------------------------------------------------------------------
# from_dict: decoding and restored types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("correlation_type", list(CorrelationType))
def test_from_dict_restores_instance(correlation_type: CorrelationType) -> None:
    instance = make_instance(correlation_type=correlation_type)

    restored = from_dict(to_dict(instance))

    assert isinstance(restored, KnapsackInstance)
    assert restored == instance


def test_from_dict_restores_int64_arrays() -> None:
    restored = from_dict(to_dict(make_instance()))

    assert isinstance(restored.values, np.ndarray)
    assert isinstance(restored.weights, np.ndarray)
    assert restored.values.dtype == np.int64
    assert restored.weights.dtype == np.int64
    assert restored.values.ndim == 1
    assert restored.weights.ndim == 1


def test_from_dict_restores_correlation_enum() -> None:
    restored = from_dict(to_dict(make_instance(correlation_type=CorrelationType.WEAKLY_CORRELATED)))

    assert restored.correlation_type is CorrelationType.WEAKLY_CORRELATED


# ---------------------------------------------------------------------------
# from_dict / to_dict round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("correlation_type", list(CorrelationType))
@pytest.mark.parametrize("n", [1, 20, 30, 50])
def test_dict_round_trip_is_exact(correlation_type: CorrelationType, n: int) -> None:
    instance = make_instance(correlation_type=correlation_type, n=n)

    assert from_dict(to_dict(instance)) == instance


def test_round_trip_survives_json_text() -> None:
    instance = make_instance()

    restored = from_dict(json.loads(json.dumps(to_dict(instance))))

    assert restored == instance


# ---------------------------------------------------------------------------
# from_dict: validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing_key", sorted(EXPECTED_KEYS))
def test_from_dict_missing_key_raises(missing_key: str) -> None:
    data = to_dict(make_instance())
    del data[missing_key]

    with pytest.raises((KeyError, ValueError)):
        from_dict(data)


def test_from_dict_unknown_correlation_type_raises() -> None:
    data = to_dict(make_instance())
    data["correlation_type"] = "nonexistent"

    with pytest.raises(ValueError):
        from_dict(data)


def test_from_dict_values_weights_length_mismatch_raises() -> None:
    data = to_dict(make_instance(n=20))
    data["weights"] = data["weights"][:-1]  # 19 weights vs 20 values

    with pytest.raises(ValueError):
        from_dict(data)


def test_from_dict_n_inconsistent_with_items_raises() -> None:
    data = to_dict(make_instance(n=20))
    data["n"] = 19  # declared count no longer matches the 20 listed items

    with pytest.raises(ValueError):
        from_dict(data)


def test_from_dict_unsupported_schema_version_raises() -> None:
    data = to_dict(make_instance())
    data["schema_version"] = EXPECTED_SCHEMA_VERSION + 1

    with pytest.raises(ValueError):
        from_dict(data)


# ---------------------------------------------------------------------------
# save_instance / load_instance: file round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("correlation_type", list(CorrelationType))
def test_file_round_trip_is_exact(correlation_type: CorrelationType, tmp_path: Path) -> None:
    instance = make_instance(correlation_type=correlation_type)
    path = tmp_path / "instance.json"

    save_instance(instance, path)
    restored = load_instance(path)

    assert restored == instance


def test_save_instance_accepts_str_path(tmp_path: Path) -> None:
    instance = make_instance()
    path = tmp_path / "instance.json"

    save_instance(instance, str(path))
    restored = load_instance(str(path))

    assert restored == instance


def test_saved_file_is_valid_json_with_expected_keys(tmp_path: Path) -> None:
    instance = make_instance()
    path = tmp_path / "instance.json"

    save_instance(instance, path)
    parsed = json.loads(path.read_text(encoding="utf-8"))

    assert set(parsed.keys()) == EXPECTED_KEYS


# ---------------------------------------------------------------------------
# Canonical, reproducible on-disk form (supports R1-T6 checksums)
# ---------------------------------------------------------------------------


def test_repeated_saves_are_byte_identical(tmp_path: Path) -> None:
    instance = make_instance()
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"

    save_instance(instance, first)
    save_instance(instance, second)

    assert first.read_bytes() == second.read_bytes()


def test_saved_file_keys_are_sorted(tmp_path: Path) -> None:
    instance = make_instance()
    path = tmp_path / "instance.json"

    save_instance(instance, path)
    parsed = json.loads(path.read_text(encoding="utf-8"))

    assert list(parsed.keys()) == sorted(parsed.keys())


def test_saved_file_ends_with_newline(tmp_path: Path) -> None:
    instance = make_instance()
    path = tmp_path / "instance.json"

    save_instance(instance, path)

    assert path.read_bytes().endswith(b"\n")


# ---------------------------------------------------------------------------
# Optional provenance metadata (R1-T6/T7 support)
# ---------------------------------------------------------------------------
#
# ``metadata`` is an optional, free-form mapping carried verbatim by the
# serialization layer. It is the home for provenance such as ``seed`` recorded
# at the CLI boundary. The typed model stays unaware of it: ``from_dict`` /
# ``load_instance`` ignore it when reconstructing a ``KnapsackInstance``.


def test_to_dict_omits_metadata_by_default() -> None:
    data = to_dict(make_instance())

    assert "metadata" not in data
    assert set(data.keys()) == EXPECTED_KEYS


def test_to_dict_includes_metadata_when_given() -> None:
    data = to_dict(make_instance(), metadata={"seed": 123})

    assert data["metadata"] == {"seed": 123}


def test_save_records_metadata_readable_via_read_metadata(tmp_path: Path) -> None:
    instance = make_instance()
    path = tmp_path / "instance.json"

    save_instance(instance, path, metadata={"seed": 99, "note": "x"})

    assert read_metadata(path) == {"seed": 99, "note": "x"}


def test_read_metadata_is_empty_when_absent(tmp_path: Path) -> None:
    instance = make_instance()
    path = tmp_path / "instance.json"

    save_instance(instance, path)

    assert read_metadata(path) == {}


def test_metadata_values_survive_json_text(tmp_path: Path) -> None:
    instance = make_instance()
    path = tmp_path / "instance.json"

    save_instance(instance, path, metadata={"seed": 20260101})

    # An int stays an int through the JSON round-trip (not coerced to str).
    assert read_metadata(path)["seed"] == 20260101


def test_load_ignores_metadata_when_rebuilding_instance(tmp_path: Path) -> None:
    instance = make_instance()
    with_meta = tmp_path / "with.json"
    without_meta = tmp_path / "without.json"
    save_instance(instance, with_meta, metadata={"seed": 1})
    save_instance(instance, without_meta)

    # The reconstructed instance is the pure knapsack problem either way.
    assert load_instance(with_meta) == load_instance(without_meta) == instance


def test_from_dict_ignores_extra_metadata_key() -> None:
    data = to_dict(make_instance(), metadata={"seed": 1})

    assert from_dict(data) == make_instance()
