"""Knapsack capacity rule for generated instances."""

import math

import numpy as np
from numpy.typing import NDArray


def capacity(weights: NDArray[np.int_], fraction: float = 0.5) -> int:
    """Knapsack capacity as a fraction of the total weight.

    Returns ``floor(fraction * sum(weights))``. The default ``fraction = 0.5``
    is the "50% knapsack" rule used throughout the thesis: a capacity near half
    the total weight forces genuine selection (neither "everything fits" nor
    "nothing fits"), which maximises the difficulty spread between the
    correlation classes. The ``fraction`` parameter keeps the function generic
    for other capacity regimes.

    Args:
        weights: Item weights (non-negative integers).
        fraction: Capacity as a share of the total weight; must be in the open
            interval ``(0, 1)``. Restricting it below 1 guarantees a
            non-trivial knapsack (``capacity < sum(weights)``).

    Returns:
        The capacity as a native Python ``int``.

    Raises:
        ValueError: If ``fraction`` is not in ``(0, 1)``.
    """
    if not (0 < fraction < 1):
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    return math.floor(fraction * int(np.sum(weights)))
