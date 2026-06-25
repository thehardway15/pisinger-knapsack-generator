"""Value-weight correlation classes for generated knapsack instances."""

from enum import Enum


class CorrelationType(Enum):
    """Correlation between an item's value and its weight.

    Follows Pisinger's instance taxonomy (pisinger2005hard, "Where are the hard
    knapsack problems?"). Given a weight ``w ~ U(1, R)`` and an offset ``d``
    (``R // 10`` by convention), the value ``v`` per class is described below.
    The enum value is the lowercase short name written to instance files.

    Attributes:
        UNCORRELATED: ``v ~ U(1, R)``, drawn independently of ``w``. Wide spread
            of value/weight ratios; the easiest class to solve.
        WEAKLY_CORRELATED: ``v ~ U(w - d, w + d)`` clipped to ``>= 1``. Values
            track weights with bounded noise.
        STRONGLY_CORRELATED: ``v = w + d`` exactly. All items share a constant
            value/weight gap; the classic hard class.
    """

    UNCORRELATED = "uncorrelated"
    WEAKLY_CORRELATED = "weakly"
    STRONGLY_CORRELATED = "strongly"
