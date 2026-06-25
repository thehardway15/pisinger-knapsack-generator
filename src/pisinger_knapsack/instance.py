"""Data model for a single 0/1 knapsack instance."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .correlation import CorrelationType


@dataclass(frozen=True)
class KnapsackInstance:
    """A generated 0/1 knapsack instance.

    The ``values`` and ``weights`` arrays are positionally aligned: item ``i``
    has value ``values[i]`` and weight ``weights[i]``.

    Attributes:
        n: Number of items (length of ``values`` and ``weights``).
        R: Data-range coefficient used during generation.
        correlation_type: Value-weight correlation class of the instance.
        values: Item values, a 1-D integer array with all entries ``>= 1``.
        weights: Item weights, a 1-D integer array with entries in ``[1, R]``.
        capacity: Knapsack capacity, ``floor(0.5 * sum(weights))`` by default.

    Note:
        The dataclass is frozen, but its ndarray fields are not deep-copied and
        remain mutable in place; callers should treat them as read-only.
        Equality is value-based (see :meth:`__eq__`): instances compare equal
        when their scalar fields match and their arrays are element-wise equal.
    """

    n: int
    R: int
    correlation_type: CorrelationType
    values: NDArray[np.int_]
    weights: NDArray[np.int_]
    capacity: int

    def __eq__(self, other: object) -> bool:
        """Compare two instances by value.

        Scalar fields must be equal and the ``values`` / ``weights`` arrays must
        match element-wise. A custom implementation is required because the
        default dataclass ``__eq__`` would raise on the ambiguous truth value of
        an array comparison.

        Args:
            other: Object to compare against.

        Returns:
            ``True`` if ``other`` is an equal :class:`KnapsackInstance`, ``False``
            if it is an unequal one, and ``NotImplemented`` for any other type
            (so Python falls back to the reflected comparison).
        """
        if not isinstance(other, KnapsackInstance):
            return NotImplemented
        a, b = self, other
        return (
            a.n == b.n
            and a.R == b.R
            and a.correlation_type == b.correlation_type
            and np.array_equal(a.values, b.values)
            and np.array_equal(a.weights, b.weights)
            and a.capacity == b.capacity
        )
