"""Unit conversion and normalisation helpers.

No imports are required — all functions use only primitive arithmetic.
"""


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit.

    Formula::

        °F = °C × 9/5 + 32

    Args:
        celsius: Temperature in degrees Celsius.  Accepts any real number,
            including values below absolute zero (no physical validation is
            performed).

    Returns:
        The equivalent temperature in degrees Fahrenheit as a float.

    Raises:
        TypeError: If *celsius* is not a numeric type.

    Examples:
        >>> celsius_to_fahrenheit(0)      # freezing point of water
        32.0
        >>> celsius_to_fahrenheit(100)    # boiling point of water
        212.0
        >>> celsius_to_fahrenheit(37)     # normal human body temperature
        98.60000000000001
        >>> celsius_to_fahrenheit(-40)    # point where °C and °F coincide
        -40.0
    """
    if not isinstance(celsius, (int, float)):
        raise TypeError(f"celsius must be a numeric type, got {type(celsius).__name__!r}")
    return celsius * 9 / 5 + 32


def km_to_miles(km: float) -> float:
    """Convert a distance from kilometres to miles.

    Formula::

        miles = km / 1.609344

    The conversion factor ``1.609344`` is the exact international definition
    (one international mile = 1 609.344 metres).

    Args:
        km: Distance in kilometres.  Must be a non-negative number.

    Returns:
        The equivalent distance in miles as a float.

    Raises:
        TypeError: If *km* is not a numeric type.
        ValueError: If *km* is negative.

    Examples:
        >>> km_to_miles(1.609344)     # exactly one mile
        1.0
        >>> km_to_miles(0)            # zero kilometres
        0.0
        >>> km_to_miles(42.195)       # marathon distance
        26.21875594...                # approximately 26.22 miles
        >>> km_to_miles(100)
        62.13711922...
    """
    if not isinstance(km, (int, float)):
        raise TypeError(f"km must be a numeric type, got {type(km).__name__!r}")
    if km < 0:
        raise ValueError(f"km must be non-negative, got {km!r}")
    return km / 1.609344


def normalize_0_100(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """Normalise *value* to the 0–100 scale given a known *minimum* and *maximum*.

    Formula::

        result = (value - minimum) / (maximum - minimum) × 100

    The result is **not** clamped — values outside ``[minimum, maximum]``
    will produce results outside ``[0, 100]``, which is useful for detecting
    out-of-range inputs.

    Args:
        value: The number to normalise.
        minimum: The lower bound of the original scale (maps to ``0``).
        maximum: The upper bound of the original scale (maps to ``100``).

    Returns:
        The normalised value on the 0–100 scale as a float.

    Raises:
        TypeError: If any argument is not a numeric type.
        ValueError: If ``minimum == maximum`` (zero-range scale is undefined).

    Examples:
        >>> normalize_0_100(50, 0, 100)     # midpoint → 50
        50.0
        >>> normalize_0_100(0, 0, 100)      # at minimum → 0
        0.0
        >>> normalize_0_100(100, 0, 100)    # at maximum → 100
        100.0
        >>> normalize_0_100(0, -10, 10)     # midpoint of -10..10 → 50
        50.0
        >>> normalize_0_100(150, 0, 100)    # out-of-range → > 100
        150.0
    """
    for name, val in (("value", value), ("minimum", minimum), ("maximum", maximum)):
        if not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be a numeric type, got {type(val).__name__!r}")
    if minimum == maximum:
        raise ValueError(
            f"minimum and maximum must differ; both are {minimum!r}"
        )
    return (value - minimum) / (maximum - minimum) * 100
