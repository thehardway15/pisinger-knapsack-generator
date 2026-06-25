"""Tests for the value-weight correlation schemes (R1-T2).

These tests act as an executable specification for the generator's public API:

    from pisinger_knapsack import (
        CorrelationType,
        KnapsackInstance,
        generate_instance,
    )

Expected contract:

* ``CorrelationType`` -- an ``Enum`` with the three Pisinger classes.
* ``KnapsackInstance`` -- a frozen dataclass with fields ``n``, ``R``,
  ``correlation_type``, ``values`` and ``weights`` (the last two are 1-D
  integer ``numpy.ndarray`` -- the instance is an internal data carrier).
* ``generate_instance(rng, n, correlation_type, R=1000, offset=None)`` -- returns
  a ``KnapsackInstance``. ``offset`` defaults to ``R // 10``. Generation is
  driven solely by the injected ``numpy.random.Generator`` (no global RNG).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from pisinger_knapsack import (
    CorrelationType,
    KnapsackInstance,
    generate_instance,
)

# A large sample size makes the Pearson-correlation assertions stable.
N_LARGE = 5000
# Arbitrary fixed seed; reused so the suite is fully deterministic.
SEED = 20260101
DEFAULT_R = 1000


def make_rng(seed: int = SEED) -> np.random.Generator:
    """Return a fresh, explicitly seeded NumPy generator."""
    return np.random.default_rng(seed)


def pearson(instance: KnapsackInstance) -> float:
    """Pearson correlation coefficient between values and weights."""
    return float(np.corrcoef(instance.values, instance.weights)[0, 1])


def instances_equal(a: KnapsackInstance, b: KnapsackInstance) -> bool:
    """Structural equality for instances holding ndarray fields.

    The dataclass default ``__eq__`` cannot be used here: comparing the ndarray
    fields yields an array, whose truth value is ambiguous.
    """
    return (
        a.n == b.n
        and a.R == b.R
        and a.correlation_type is b.correlation_type
        and a.capacity == b.capacity
        and np.array_equal(a.values, b.values)
        and np.array_equal(a.weights, b.weights)
    )


# ---------------------------------------------------------------------------
# Structural contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("correlation_type", list(CorrelationType))
def test_returns_instance_with_expected_fields(correlation_type: CorrelationType) -> None:
    instance = generate_instance(make_rng(), n=42, correlation_type=correlation_type, R=DEFAULT_R)

    assert isinstance(instance, KnapsackInstance)
    assert instance.n == 42
    assert instance.R == DEFAULT_R
    assert instance.correlation_type is correlation_type


@pytest.mark.parametrize("correlation_type", list(CorrelationType))
def test_items_are_one_dimensional_integer_ndarrays(correlation_type: CorrelationType) -> None:
    instance = generate_instance(make_rng(), n=20, correlation_type=correlation_type, R=DEFAULT_R)

    assert isinstance(instance.values, np.ndarray)
    assert isinstance(instance.weights, np.ndarray)
    assert instance.values.ndim == 1
    assert instance.weights.ndim == 1
    assert np.issubdtype(instance.values.dtype, np.integer)
    assert np.issubdtype(instance.weights.dtype, np.integer)


@pytest.mark.parametrize("correlation_type", list(CorrelationType))
@pytest.mark.parametrize("n", [1, 20, 30, 50, 137])
def test_item_count_matches_n(correlation_type: CorrelationType, n: int) -> None:
    instance = generate_instance(make_rng(), n=n, correlation_type=correlation_type, R=DEFAULT_R)

    assert len(instance.values) == n
    assert len(instance.weights) == n


# ---------------------------------------------------------------------------
# Value / weight ranges per correlation type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("correlation_type", list(CorrelationType))
def test_weights_within_unit_to_R(correlation_type: CorrelationType) -> None:
    instance = generate_instance(
        make_rng(), n=N_LARGE, correlation_type=correlation_type, R=DEFAULT_R
    )

    assert all(1 <= w <= DEFAULT_R for w in instance.weights)


@pytest.mark.parametrize("correlation_type", list(CorrelationType))
def test_values_at_least_one(correlation_type: CorrelationType) -> None:
    instance = generate_instance(
        make_rng(), n=N_LARGE, correlation_type=correlation_type, R=DEFAULT_R
    )

    assert all(v >= 1 for v in instance.values)


def test_uncorrelated_values_within_unit_to_R() -> None:
    instance = generate_instance(
        make_rng(), n=N_LARGE, correlation_type=CorrelationType.UNCORRELATED, R=DEFAULT_R
    )

    assert all(1 <= v <= DEFAULT_R for v in instance.values)


def test_strongly_correlated_uses_fixed_offset() -> None:
    offset = DEFAULT_R // 10
    instance = generate_instance(
        make_rng(), n=N_LARGE, correlation_type=CorrelationType.STRONGLY_CORRELATED, R=DEFAULT_R
    )

    assert all(v == w + offset for v, w in zip(instance.values, instance.weights, strict=True))


def test_strongly_correlated_values_may_exceed_R() -> None:
    # By definition v = w + R/10, so for large weights v can exceed R; this is
    # an accepted property of the strongly correlated class.
    instance = generate_instance(
        make_rng(), n=N_LARGE, correlation_type=CorrelationType.STRONGLY_CORRELATED, R=DEFAULT_R
    )

    assert max(instance.values) > DEFAULT_R


def test_weakly_correlated_values_within_noise_band() -> None:
    offset = DEFAULT_R // 10
    instance = generate_instance(
        make_rng(), n=N_LARGE, correlation_type=CorrelationType.WEAKLY_CORRELATED, R=DEFAULT_R
    )

    for v, w in zip(instance.values, instance.weights, strict=True):
        assert max(1, w - offset) <= v <= w + offset


def test_weakly_correlated_values_may_exceed_R() -> None:
    # The noise band is only clipped at the bottom (to >= 1); the upper end is
    # left untouched, so weights near R plus positive noise yield values above
    # R -- exactly as for the strongly correlated class.
    instance = generate_instance(
        make_rng(), n=N_LARGE, correlation_type=CorrelationType.WEAKLY_CORRELATED, R=DEFAULT_R
    )

    assert max(instance.values) > DEFAULT_R


# ---------------------------------------------------------------------------
# Correlation strength: uncorrelated < weakly < strongly (== 1.0)
# ---------------------------------------------------------------------------


def test_uncorrelated_correlation_near_zero() -> None:
    instance = generate_instance(
        make_rng(), n=N_LARGE, correlation_type=CorrelationType.UNCORRELATED, R=DEFAULT_R
    )

    assert abs(pearson(instance)) < 0.1


def test_weakly_correlated_correlation_high() -> None:
    instance = generate_instance(
        make_rng(), n=N_LARGE, correlation_type=CorrelationType.WEAKLY_CORRELATED, R=DEFAULT_R
    )

    assert pearson(instance) > 0.9


def test_strongly_correlated_correlation_is_one() -> None:
    instance = generate_instance(
        make_rng(), n=N_LARGE, correlation_type=CorrelationType.STRONGLY_CORRELATED, R=DEFAULT_R
    )

    assert pearson(instance) == pytest.approx(1.0, abs=1e-9)


def test_correlation_increases_monotonically() -> None:
    r_uncorr = pearson(
        generate_instance(
            make_rng(), n=N_LARGE, correlation_type=CorrelationType.UNCORRELATED, R=DEFAULT_R
        )
    )
    r_weak = pearson(
        generate_instance(
            make_rng(), n=N_LARGE, correlation_type=CorrelationType.WEAKLY_CORRELATED, R=DEFAULT_R
        )
    )
    r_strong = pearson(
        generate_instance(
            make_rng(), n=N_LARGE, correlation_type=CorrelationType.STRONGLY_CORRELATED, R=DEFAULT_R
        )
    )

    assert r_uncorr < r_weak < r_strong


# ---------------------------------------------------------------------------
# Determinism and RNG isolation (overlaps with R1-T4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("correlation_type", list(CorrelationType))
def test_same_seed_yields_identical_instances(correlation_type: CorrelationType) -> None:
    first = generate_instance(make_rng(123), n=200, correlation_type=correlation_type, R=DEFAULT_R)
    second = generate_instance(make_rng(123), n=200, correlation_type=correlation_type, R=DEFAULT_R)

    assert instances_equal(first, second)


@pytest.mark.parametrize("correlation_type", list(CorrelationType))
def test_different_seed_yields_different_instances(correlation_type: CorrelationType) -> None:
    first = generate_instance(make_rng(1), n=200, correlation_type=correlation_type, R=DEFAULT_R)
    second = generate_instance(make_rng(2), n=200, correlation_type=correlation_type, R=DEFAULT_R)

    assert not instances_equal(first, second)


def test_generation_is_independent_of_call_order() -> None:
    # A reference result produced from a freshly seeded generator...
    reference = generate_instance(
        make_rng(777), n=200, correlation_type=CorrelationType.WEAKLY_CORRELATED, R=DEFAULT_R
    )

    # ...must be reproduced even if an unrelated generation happened in between,
    # proving there is no shared/global RNG state.
    generate_instance(
        make_rng(999), n=200, correlation_type=CorrelationType.STRONGLY_CORRELATED, R=DEFAULT_R
    )
    again = generate_instance(
        make_rng(777), n=200, correlation_type=CorrelationType.WEAKLY_CORRELATED, R=DEFAULT_R
    )

    assert instances_equal(reference, again)


def test_does_not_touch_global_numpy_rng() -> None:
    before = cast("tuple[Any, ...]", np.random.get_state())
    generate_instance(
        make_rng(), n=N_LARGE, correlation_type=CorrelationType.UNCORRELATED, R=DEFAULT_R
    )
    after = cast("tuple[Any, ...]", np.random.get_state())

    # before/after are (str, ndarray, int, int, float) tuples.
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_passing_generator_advances_its_state() -> None:
    # The injected generator must actually be consumed, so two sequential calls
    # on the *same* generator object produce different draws.
    rng = make_rng(555)
    first = generate_instance(
        rng, n=200, correlation_type=CorrelationType.UNCORRELATED, R=DEFAULT_R
    )
    second = generate_instance(
        rng, n=200, correlation_type=CorrelationType.UNCORRELATED, R=DEFAULT_R
    )

    assert not instances_equal(first, second)


# ---------------------------------------------------------------------------
# Custom offset
# ---------------------------------------------------------------------------


def test_offset_defaults_to_R_over_ten() -> None:
    instance = generate_instance(
        make_rng(), n=N_LARGE, correlation_type=CorrelationType.STRONGLY_CORRELATED, R=2000
    )

    assert all(v == w + 200 for v, w in zip(instance.values, instance.weights, strict=True))


def test_explicit_offset_overrides_default_for_strongly() -> None:
    instance = generate_instance(
        make_rng(),
        n=N_LARGE,
        correlation_type=CorrelationType.STRONGLY_CORRELATED,
        R=DEFAULT_R,
        offset=50,
    )

    assert all(v == w + 50 for v, w in zip(instance.values, instance.weights, strict=True))


def test_explicit_offset_overrides_default_for_weakly() -> None:
    offset = 250
    instance = generate_instance(
        make_rng(),
        n=N_LARGE,
        correlation_type=CorrelationType.WEAKLY_CORRELATED,
        R=DEFAULT_R,
        offset=offset,
    )

    for v, w in zip(instance.values, instance.weights, strict=True):
        assert max(1, w - offset) <= v <= w + offset


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_n", [0, -1, -100])
def test_non_positive_n_raises(bad_n: int) -> None:
    with pytest.raises(ValueError):
        generate_instance(
            make_rng(), n=bad_n, correlation_type=CorrelationType.UNCORRELATED, R=DEFAULT_R
        )


@pytest.mark.parametrize("bad_R", [0, -1, -1000])
def test_non_positive_R_raises(bad_R: int) -> None:
    with pytest.raises(ValueError):
        generate_instance(make_rng(), n=20, correlation_type=CorrelationType.UNCORRELATED, R=bad_R)


def test_invalid_correlation_type_raises() -> None:
    with pytest.raises((TypeError, ValueError)):
        generate_instance(
            make_rng(),
            n=20,
            correlation_type="uncorrelated",  # type: ignore[arg-type]
            R=DEFAULT_R,
        )
