"""Tests for the knapsack capacity rule (R1-T3).

Executable specification of the capacity contract:

* ``capacity(weights, fraction=0.5) -> int`` returns
  ``floor(fraction * sum(weights))`` as a native Python ``int``.
* ``KnapsackInstance`` gains an integer ``capacity`` field.
* ``generate_instance`` populates ``capacity`` with the default fraction (0.5).

The "50% knapsack" rule (``W = floor(0.5 * sum(weights))``) is fixed for the
thesis; the standalone function stays generic via the ``fraction`` parameter.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pisinger_knapsack import (
    CorrelationType,
    KnapsackInstance,
    capacity,
    generate_instance,
)

SEED = 20260101
DEFAULT_R = 1000


def make_rng(seed: int = SEED) -> np.random.Generator:
    """Return a fresh, explicitly seeded NumPy generator."""
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Standalone capacity() function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ([2, 4, 6], 6),  # sum=12  -> 0.5*12 = 6
        ([1, 2, 3, 4], 5),  # sum=10  -> 5
        ([1, 1, 1], 1),  # sum=3   -> floor(1.5) = 1
        ([5], 2),  # sum=5   -> floor(2.5) = 2
    ],
)
def test_capacity_is_floor_of_half_sum(weights: list[int], expected: int) -> None:
    assert capacity(np.array(weights)) == expected


def test_capacity_default_fraction_is_one_half() -> None:
    w = np.array([10, 20, 30])  # sum=60
    assert capacity(w) == capacity(w, fraction=0.5) == 30


def test_capacity_respects_custom_fraction() -> None:
    w = np.array([1, 2, 3, 4])  # sum=10
    assert capacity(w, fraction=0.25) == 2  # floor(2.5)
    assert capacity(w, fraction=0.75) == 7  # floor(7.5)


def test_capacity_uses_floor_not_round() -> None:
    # 0.5 * 3 = 1.5 -> floor gives 1 (round would give 2).
    assert capacity(np.array([1, 1, 1])) == 1


def test_capacity_returns_native_python_int() -> None:
    cap = capacity(np.array([2, 4, 6]))

    assert isinstance(cap, int)  # np.int64 is *not* an int instance


def test_capacity_is_nontrivial_for_reasonable_weights() -> None:
    w = np.array([4, 7, 2, 9, 5])  # sum=27
    total = int(w.sum())

    assert 1 <= capacity(w) < total


# ---------------------------------------------------------------------------
# Integration: capacity stored on the generated instance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("correlation_type", list(CorrelationType))
def test_generate_instance_populates_capacity(correlation_type: CorrelationType) -> None:
    instance = generate_instance(make_rng(), n=50, correlation_type=correlation_type, R=DEFAULT_R)

    assert isinstance(instance, KnapsackInstance)
    assert instance.capacity == capacity(instance.weights)


@pytest.mark.parametrize("correlation_type", list(CorrelationType))
def test_generated_capacity_equals_floor_half_sum(correlation_type: CorrelationType) -> None:
    instance = generate_instance(make_rng(), n=50, correlation_type=correlation_type, R=DEFAULT_R)

    assert instance.capacity == math.floor(0.5 * int(instance.weights.sum()))


@pytest.mark.parametrize("correlation_type", list(CorrelationType))
def test_generated_capacity_is_native_int(correlation_type: CorrelationType) -> None:
    instance = generate_instance(make_rng(), n=30, correlation_type=correlation_type, R=DEFAULT_R)

    assert isinstance(instance.capacity, int)


@pytest.mark.parametrize("correlation_type", list(CorrelationType))
def test_generated_capacity_is_nontrivial(correlation_type: CorrelationType) -> None:
    instance = generate_instance(make_rng(), n=50, correlation_type=correlation_type, R=DEFAULT_R)
    total = int(instance.weights.sum())

    assert 1 <= instance.capacity < total
