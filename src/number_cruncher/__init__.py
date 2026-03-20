"""number_cruncher — statistical utilities, scaling, and unit conversion."""

__version__ = "0.3.0"

from .stats import mean, median, variance, std_dev, percentile
from .scaler import min_max_scale, z_score_scale, clamp
from .converter import celsius_to_fahrenheit, km_to_miles, normalize_0_100

__all__ = [
    # stats
    "mean",
    "median",
    "variance",
    "std_dev",
    "percentile",
    # scaler
    "min_max_scale",
    "z_score_scale",
    "clamp",
    # converter
    "celsius_to_fahrenheit",
    "km_to_miles",
    "normalize_0_100",
]
