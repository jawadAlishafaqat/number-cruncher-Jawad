"""Descriptive statistics functions using only the Python standard library."""

import math
import statistics
from typing import Sequence


def mean(data: Sequence[float]) -> float:
    """Return the arithmetic mean of *data*.

    Args:
        data: A non-empty sequence of real numbers.

    Returns:
        The arithmetic mean as a float.

    Raises:
        ValueError: If *data* is empty.

    Example:
        >>> mean([1, 2, 3, 4, 5])
        3.0
        >>> mean([10.5, 20.5])
        15.5
    """
    if not data:
        raise ValueError("mean() requires a non-empty sequence")
    return statistics.mean(data)


def median(data: Sequence[float]) -> float:
    """Return the median (middle value) of *data*.

    For sequences with an even number of elements the two central values
    are averaged, so the result may not itself appear in *data*.

    Args:
        data: A non-empty sequence of real numbers.

    Returns:
        The median value as a float.

    Raises:
        ValueError: If *data* is empty.

    Example:
        >>> median([3, 1, 2])
        2
        >>> median([1, 2, 3, 4])
        2.5
    """
    if not data:
        raise ValueError("median() requires a non-empty sequence")
    return statistics.median(data)


def variance(data: Sequence[float], ddof: int = 1) -> float:
    """Return the variance of *data*.

    Args:
        data: A non-empty sequence of real numbers.  When *ddof* is 1
            (the default) at least two elements are required.
        ddof: Delta degrees of freedom.  Use ``0`` for the population
            variance and ``1`` (default) for the sample variance.

    Returns:
        The variance as a float.

    Raises:
        ValueError: If *data* is empty, or if ``len(data) < 2`` when
            ``ddof=1``.

    Example:
        >>> variance([2, 4, 4, 4, 5, 5, 7, 9])          # sample variance
        4.571428571428571
        >>> variance([2, 4, 4, 4, 5, 5, 7, 9], ddof=0)  # population variance
        4.0
    """
    if not data:
        raise ValueError("variance() requires a non-empty sequence")
    if ddof == 1:
        if len(data) < 2:
            raise ValueError(
                "variance() requires at least 2 data points when ddof=1"
            )
        return statistics.variance(data)
    if ddof == 0:
        return statistics.pvariance(data)
    # General path for arbitrary ddof values
    n = len(data)
    if n <= ddof:
        raise ValueError(
            f"variance() requires more than {ddof} data point(s) for ddof={ddof}"
        )
    mu = sum(data) / n
    return sum((x - mu) ** 2 for x in data) / (n - ddof)


def std_dev(data: Sequence[float], ddof: int = 1) -> float:
    """Return the standard deviation of *data*.

    This is the square root of :func:`variance` and shares the same
    *ddof* convention.

    Args:
        data: A non-empty sequence of real numbers.  When *ddof* is 1
            (the default) at least two elements are required.
        ddof: Delta degrees of freedom.  Use ``0`` for the population
            standard deviation and ``1`` (default) for the sample
            standard deviation.

    Returns:
        The standard deviation as a float (always ≥ 0).

    Raises:
        ValueError: If *data* is empty, or if ``len(data) < 2`` when
            ``ddof=1``.

    Example:
        >>> std_dev([2, 4, 4, 4, 5, 5, 7, 9])          # sample std dev
        2.138089935299395
        >>> std_dev([2, 4, 4, 4, 5, 5, 7, 9], ddof=0)  # population std dev
        2.0
    """
    return math.sqrt(variance(data, ddof=ddof))


def percentile(data: Sequence[float], p: float) -> float:
    """Return the *p*-th percentile of *data* using linear interpolation.

    The percentile is computed by sorting *data*, locating the fractional
    index ``i = p / 100 * (n - 1)``, and linearly interpolating between
    the two surrounding values when *i* is not a whole number.

    Args:
        data: A non-empty sequence of real numbers.
        p: Percentile to compute, in the closed interval ``[0, 100]``.

    Returns:
        The interpolated *p*-th percentile as a float.

    Raises:
        ValueError: If *data* is empty.
        ValueError: If *p* is outside the range ``[0, 100]``.

    Example:
        >>> percentile([1, 2, 3, 4, 5], 50)   # median
        3.0
        >>> percentile([1, 2, 3, 4, 5], 25)   # first quartile
        2.0
        >>> percentile([1, 2, 3, 4, 5], 75)   # third quartile
        4.0
    """
    if not data:
        raise ValueError("percentile() requires a non-empty sequence")
    if not (0.0 <= p <= 100.0):
        raise ValueError(f"percentile p must be in [0, 100], got {p!r}")

    sorted_data = sorted(data)
    n = len(sorted_data)

    if n == 1:
        return float(sorted_data[0])

    index = p / 100 * (n - 1)
    lower = int(index)
    upper = lower + 1

    if upper >= n:
        return float(sorted_data[-1])

    fraction = index - lower
    return sorted_data[lower] + fraction * (sorted_data[upper] - sorted_data[lower])
