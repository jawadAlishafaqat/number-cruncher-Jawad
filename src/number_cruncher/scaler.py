"""Numeric scaling and clamping functions."""

from typing import List, Sequence

from .stats import mean, std_dev


def min_max_scale(
    data: Sequence[float],
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> List[float]:
    """Scale *data* to *feature_range* using min-max normalisation.

    Each value is transformed by::

        x_scaled = lo + (x - data_min) / (data_max - data_min) * (hi - lo)

    When all values in *data* are identical (zero range) every element is
    mapped to the lower bound of *feature_range*.

    Args:
        data: A non-empty sequence of real numbers.
        feature_range: A ``(min, max)`` tuple defining the target output
            range.  Defaults to ``(0.0, 1.0)``.

    Returns:
        A new list of floats scaled to *feature_range*, preserving the
        original order of *data*.

    Raises:
        ValueError: If *data* is empty.
        ValueError: If ``feature_range[0] >= feature_range[1]``.

    Example:
        >>> min_max_scale([0, 5, 10])
        [0.0, 0.5, 1.0]
        >>> min_max_scale([0, 10], feature_range=(-1.0, 1.0))
        [-1.0, 1.0]
        >>> min_max_scale([7, 7, 7])   # constant data → lower bound
        [0.0, 0.0, 0.0]
    """
    if not data:
        raise ValueError("min_max_scale() requires a non-empty sequence")

    lo, hi = feature_range
    if lo >= hi:
        raise ValueError(
            f"feature_range min must be strictly less than max, got {feature_range!r}"
        )

    data_min = min(data)
    data_max = max(data)
    span = data_max - data_min

    if span == 0:
        return [lo] * len(data)

    scale = (hi - lo) / span
    return [lo + (x - data_min) * scale for x in data]


def z_score_scale(data: Sequence[float]) -> List[float]:
    """Standardise *data* to zero mean and unit variance (z-score normalisation).

    Each value is transformed by::

        x_scaled = (x - mean) / std_dev

    When the standard deviation is zero (all values identical) every element
    is returned as ``0.0`` rather than raising a ``ZeroDivisionError``.

    Args:
        data: A non-empty sequence of real numbers containing at least two
            elements so that sample standard deviation is well-defined.

    Returns:
        A new list of floats with mean ≈ 0 and standard deviation ≈ 1,
        preserving the original order of *data*.  Returns a list of zeros
        when all input values are identical.

    Raises:
        ValueError: If *data* is empty.
        ValueError: If *data* contains fewer than 2 elements (sample std dev
            requires at least 2 data points).

    Example:
        >>> z_score_scale([2, 4, 4, 4, 5, 5, 7, 9])
        [-1.5, -0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 2.0]  # approximate
        >>> z_score_scale([3, 3, 3])   # constant data → all zeros
        [0.0, 0.0, 0.0]
    """
    if not data:
        raise ValueError("z_score_scale() requires a non-empty sequence")
    if len(data) < 2:
        raise ValueError(
            "z_score_scale() requires at least 2 data points for sample std dev"
        )

    mu = mean(data)
    sigma = std_dev(data, ddof=1)

    if sigma == 0:
        return [0.0] * len(data)

    return [(x - mu) / sigma for x in data]


def clamp(
    value: float,
    lower: float,
    upper: float,
) -> float:
    """Clamp *value* to the closed interval ``[lower, upper]``.

    * If *value* is below *lower*, returns *lower*.
    * If *value* is above *upper*, returns *upper*.
    * Otherwise returns *value* unchanged.

    Args:
        value: The number to clamp.
        lower: The inclusive lower bound of the allowed range.
        upper: The inclusive upper bound of the allowed range.

    Returns:
        *value* constrained to ``[lower, upper]`` as a float.

    Raises:
        ValueError: If ``lower > upper``.

    Example:
        >>> clamp(5.0, 0.0, 10.0)
        5.0
        >>> clamp(-3.0, 0.0, 10.0)   # below lower bound
        0.0
        >>> clamp(15.0, 0.0, 10.0)   # above upper bound
        10.0
        >>> clamp(0.0, 0.0, 0.0)     # degenerate single-point range
        0.0
    """
    if lower > upper:
        raise ValueError(
            f"lower bound must be <= upper bound, got lower={lower!r}, upper={upper!r}"
        )
    return float(max(lower, min(value, upper)))
