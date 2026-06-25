"""Generation of 0/1 knapsack instances under Pisinger's correlation schemes."""

import numpy as np

from .capacity import capacity
from .constants import DEFAULT_R
from .correlation import CorrelationType
from .instance import KnapsackInstance


def generate_instance(
    rng: np.random.Generator,
    n: int,
    correlation_type: CorrelationType,
    R: int = DEFAULT_R,
    offset: int | None = None,
) -> KnapsackInstance:
    """Generate one knapsack instance for the given correlation class.

    All draws come from the supplied ``rng``; no global NumPy RNG state is read
    or written, so a given generator state yields a reproducible instance.
    Weights and values are sampled as integers directly (no rounding step).

    Args:
        rng: Explicit NumPy generator driving every random draw.
        n: Number of items to generate; must be ``>= 1``.
        correlation_type: Value-weight correlation class to apply.
        R: Data-range coefficient; weights are drawn from ``U(1, R)``. Must be
            ``>= 1``.
        offset: Value-weight offset. Defaults to ``R // 10`` (Pisinger's
            convention) when ``None``; must resolve to ``>= 1``.

    Returns:
        The generated :class:`KnapsackInstance`.

    Raises:
        ValueError: If ``n``, ``R`` or the resolved ``offset`` is below 1, or
            if ``correlation_type`` is an unsupported member.
        TypeError: If ``correlation_type`` is not a :class:`CorrelationType`.

    Note:
        Formulas follow pisinger2005hard ("Where are the hard knapsack
        problems?"). With ``w ~ U(1, R)`` and the offset above:

        * uncorrelated: ``v ~ U(1, R)``, independent of ``w``;
        * weakly correlated: ``v ~ U(w - offset, w + offset)`` clipped to
          ``>= 1`` (lower bound only);
        * strongly correlated: ``v = w + offset``.

        Weights always lie in ``[1, R]``. Values are always ``>= 1`` but may
        exceed ``R`` for the weakly and strongly correlated classes (by design).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if R <= 0:
        raise ValueError(f"R must be positive, got {R}")
    if not isinstance(correlation_type, CorrelationType):
        raise TypeError(f"correlation_type must be a CorrelationType, got {type(correlation_type)}")

    if offset is None:
        offset = R // 10

    if offset < 1:
        raise ValueError(f"offset must be at least 1, got {offset}")

    match correlation_type:
        case CorrelationType.UNCORRELATED:
            values = rng.integers(1, R + 1, size=n)
            weights = rng.integers(1, R + 1, size=n)
        case CorrelationType.WEAKLY_CORRELATED:
            weights = rng.integers(1, R + 1, size=n)
            noise = rng.integers(-offset, offset + 1, size=n)
            values = np.maximum(weights + noise, 1)
        case CorrelationType.STRONGLY_CORRELATED:
            weights = rng.integers(1, R + 1, size=n)
            values = weights + offset
        case _:
            raise ValueError(f"Invalid correlation type: {correlation_type}")

    kp = KnapsackInstance(
        n=n,
        R=R,
        correlation_type=correlation_type,
        values=values,
        weights=weights,
        capacity=capacity(weights),
    )

    return kp
