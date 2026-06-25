"""Tests for ``KnapsackInstance`` equality (R1-T5 support).

Executable specification for the value equality added to the data model:

* ``KnapsackInstance`` gains a working ``__eq__`` that compares the scalar
  fields (``n``, ``R``, ``correlation_type``, ``capacity``) and performs an
  element-wise comparison of the ``values`` and ``weights`` ndarrays.
* Equality must not raise on the ndarray fields (the dataclass default
  ``__eq__`` does, because an array comparison has an ambiguous truth value).
* Equality is by content, not by array identity, and two instances differing in
  any single field compare unequal.

This contract is what makes the serialization round-trip assertable as
``from_dict(to_dict(x)) == x`` (see ``test_serialization.py``).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from pisinger_knapsack import (
    CorrelationType,
    KnapsackInstance,
    generate_instance,
)

SEED = 20260101
DEFAULT_R = 1000


def make_instance(
    correlation_type: CorrelationType = CorrelationType.UNCORRELATED,
    n: int = 20,
    seed: int = SEED,
) -> KnapsackInstance:
    """Return a deterministic instance for equality experiments."""
    return generate_instance(
        np.random.default_rng(seed), n=n, correlation_type=correlation_type, R=DEFAULT_R
    )


# ---------------------------------------------------------------------------
# Positive equality
# ---------------------------------------------------------------------------


def test_equal_instances_compare_equal() -> None:
    # Same seed -> identical content, but built from separate generators and
    # therefore backed by distinct ndarray objects.
    a = make_instance()
    b = make_instance()

    assert a == b


def test_equality_is_by_content_not_array_identity() -> None:
    a = make_instance()
    # Rebuild with freshly allocated copies of the very same numbers.
    b = dataclasses.replace(
        a, values=np.array(a.values, copy=True), weights=np.array(a.weights, copy=True)
    )

    assert a.values is not b.values
    assert a.weights is not b.weights
    assert a == b


def test_equality_does_not_raise_on_ndarray_fields() -> None:
    # The whole point of the custom __eq__: comparing two instances must yield a
    # plain bool, never raise "truth value of an array is ambiguous".
    a = make_instance()
    b = make_instance()

    result = a == b

    assert isinstance(result, bool)
    assert result is True


# ---------------------------------------------------------------------------
# Inequality: one differing field at a time
# ---------------------------------------------------------------------------


def test_differing_n_makes_unequal() -> None:
    a = make_instance()
    b = dataclasses.replace(a, n=a.n + 1)

    assert a != b


def test_differing_R_makes_unequal() -> None:
    a = make_instance()
    b = dataclasses.replace(a, R=a.R + 1)

    assert a != b


def test_differing_correlation_type_makes_unequal() -> None:
    a = make_instance(correlation_type=CorrelationType.UNCORRELATED)
    b = dataclasses.replace(a, correlation_type=CorrelationType.STRONGLY_CORRELATED)

    assert a != b


def test_differing_capacity_makes_unequal() -> None:
    a = make_instance()
    b = dataclasses.replace(a, capacity=a.capacity + 1)

    assert a != b


def test_differing_values_make_unequal() -> None:
    a = make_instance()
    altered = np.array(a.values, copy=True)
    altered[0] += 1
    b = dataclasses.replace(a, values=altered)

    assert a != b


def test_differing_weights_make_unequal() -> None:
    a = make_instance()
    altered = np.array(a.weights, copy=True)
    altered[0] += 1
    b = dataclasses.replace(a, weights=altered)

    assert a != b


def test_same_length_but_different_content_unequal() -> None:
    a = make_instance(seed=1)
    b = make_instance(seed=2)

    assert a != b


# ---------------------------------------------------------------------------
# Equality against foreign types
# ---------------------------------------------------------------------------


def test_not_equal_to_unrelated_object() -> None:
    a = make_instance()

    assert a != 42
    assert a != "instance"
    assert (a == object()) is False
