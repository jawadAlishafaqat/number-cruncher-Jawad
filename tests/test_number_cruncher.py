"""Unit tests for the number_cruncher package."""

import math
import pytest

from number_cruncher.stats import mean, median, std_dev, percentile
from number_cruncher.scaler import min_max_scale, z_score_scale, clamp
from number_cruncher.converter import celsius_to_fahrenheit, km_to_miles, normalize_0_100


# ---------------------------------------------------------------------------
# 1. mean
# ---------------------------------------------------------------------------

class TestMean:
    def test_integers(self):
        assert mean([1, 2, 3, 4, 5]) == 3.0

    def test_floats(self):
        assert math.isclose(mean([1.5, 2.5, 3.0]), 7.0 / 3)

    def test_single_element(self):
        assert mean([42]) == 42.0

    def test_negative_values(self):
        assert mean([-3, -1, -2]) == -2.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            mean([])


# ---------------------------------------------------------------------------
# 2. median
# ---------------------------------------------------------------------------

class TestMedian:
    def test_odd_length_unsorted(self):
        assert median([3, 1, 2]) == 2

    def test_even_length_averages_middle(self):
        assert median([1, 2, 3, 4]) == 2.5

    def test_single_element(self):
        assert median([7]) == 7

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            median([])


# ---------------------------------------------------------------------------
# 3. std_dev
# ---------------------------------------------------------------------------

class TestStdDev:
    def test_population_ddof0(self):
        # classic example: population std dev of [2,4,4,4,5,5,7,9] == 2.0
        assert math.isclose(std_dev([2, 4, 4, 4, 5, 5, 7, 9], ddof=0), 2.0)

    def test_sample_ddof1(self):
        assert math.isclose(
            std_dev([2, 4, 4, 4, 5, 5, 7, 9], ddof=1),
            2.138089935299395,
            rel_tol=1e-9,
        )

    def test_constant_data_returns_zero(self):
        assert std_dev([5, 5, 5, 5], ddof=0) == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            std_dev([])

    def test_single_element_ddof1_raises(self):
        with pytest.raises(ValueError):
            std_dev([99], ddof=1)


# ---------------------------------------------------------------------------
# 4. percentile
# ---------------------------------------------------------------------------

class TestPercentile:
    DATA = [1, 2, 3, 4, 5]

    def test_p0_returns_minimum(self):
        assert percentile(self.DATA, 0) == 1.0

    def test_p100_returns_maximum(self):
        assert percentile(self.DATA, 100) == 5.0

    def test_p50_equals_median(self):
        assert percentile(self.DATA, 50) == 3.0

    def test_p25_first_quartile(self):
        assert percentile(self.DATA, 25) == 2.0

    def test_p75_third_quartile(self):
        assert percentile(self.DATA, 75) == 4.0

    def test_single_element_any_p(self):
        assert percentile([9], 42) == 9.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            percentile([], 50)

    def test_p_below_0_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            percentile(self.DATA, -1)

    def test_p_above_100_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            percentile(self.DATA, 101)


# ---------------------------------------------------------------------------
# 5. min_max_scale
# ---------------------------------------------------------------------------

class TestMinMaxScale:
    def test_default_range(self):
        assert min_max_scale([0, 5, 10]) == [0.0, 0.5, 1.0]

    def test_custom_range(self):
        result = min_max_scale([0, 10], feature_range=(-1.0, 1.0))
        assert result == [-1.0, 1.0]

    def test_preserves_order(self):
        result = min_max_scale([10, 0, 5])
        assert math.isclose(result[0], 1.0)
        assert math.isclose(result[1], 0.0)
        assert math.isclose(result[2], 0.5)

    def test_constant_data_maps_to_lower_bound(self):
        assert min_max_scale([7, 7, 7]) == [0.0, 0.0, 0.0]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            min_max_scale([])

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="strictly less than max"):
            min_max_scale([1, 2, 3], feature_range=(5.0, 1.0))

    def test_equal_range_bounds_raises(self):
        with pytest.raises(ValueError):
            min_max_scale([1, 2, 3], feature_range=(1.0, 1.0))


# ---------------------------------------------------------------------------
# 6. z_score_scale
# ---------------------------------------------------------------------------

class TestZScoreScale:
    def test_mean_approximately_zero(self):
        result = z_score_scale([2, 4, 4, 4, 5, 5, 7, 9])
        assert math.isclose(sum(result), 0.0, abs_tol=1e-9)

    def test_std_approximately_one(self):
        result = z_score_scale([2, 4, 4, 4, 5, 5, 7, 9])
        n = len(result)
        mu = sum(result) / n
        sample_var = sum((x - mu) ** 2 for x in result) / (n - 1)
        assert math.isclose(sample_var, 1.0, rel_tol=1e-9)

    def test_constant_data_returns_zeros(self):
        assert z_score_scale([3, 3, 3]) == [0.0, 0.0, 0.0]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            z_score_scale([])

    def test_single_element_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            z_score_scale([5])


# ---------------------------------------------------------------------------
# 7. clamp
# ---------------------------------------------------------------------------

class TestClamp:
    def test_value_within_range(self):
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_value_below_lower(self):
        assert clamp(-3.0, 0.0, 10.0) == 0.0

    def test_value_above_upper(self):
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_value_at_lower_boundary(self):
        assert clamp(0.0, 0.0, 10.0) == 0.0

    def test_value_at_upper_boundary(self):
        assert clamp(10.0, 0.0, 10.0) == 10.0

    def test_degenerate_single_point_range(self):
        assert clamp(99.0, 4.0, 4.0) == 4.0

    def test_inverted_bounds_raises(self):
        with pytest.raises(ValueError, match="lower bound must be <= upper bound"):
            clamp(5.0, 10.0, 0.0)

    def test_returns_float(self):
        assert isinstance(clamp(3, 0, 10), float)


# ---------------------------------------------------------------------------
# 8. celsius_to_fahrenheit
# ---------------------------------------------------------------------------

class TestCelsiusToFahrenheit:
    def test_freezing_point(self):
        assert celsius_to_fahrenheit(0) == 32.0

    def test_boiling_point(self):
        assert celsius_to_fahrenheit(100) == 212.0

    def test_convergence_point(self):
        # −40 is the one temperature where °C == °F
        assert celsius_to_fahrenheit(-40) == -40.0

    def test_body_temperature(self):
        assert math.isclose(celsius_to_fahrenheit(37), 98.6, rel_tol=1e-5)

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError, match="numeric type"):
            celsius_to_fahrenheit("hot")


# ---------------------------------------------------------------------------
# 9. km_to_miles
# ---------------------------------------------------------------------------

class TestKmToMiles:
    def test_one_mile_definition(self):
        assert math.isclose(km_to_miles(1.609344), 1.0)

    def test_zero(self):
        assert km_to_miles(0) == 0.0

    def test_marathon(self):
        assert math.isclose(km_to_miles(42.195), 26.2188, rel_tol=1e-4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            km_to_miles(-1)

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError, match="numeric type"):
            km_to_miles("far")


# ---------------------------------------------------------------------------
# 10. normalize_0_100
# ---------------------------------------------------------------------------

class TestNormalize0100:
    def test_minimum_maps_to_zero(self):
        assert normalize_0_100(0, 0, 100) == 0.0

    def test_maximum_maps_to_hundred(self):
        assert normalize_0_100(100, 0, 100) == 100.0

    def test_midpoint(self):
        assert normalize_0_100(50, 0, 100) == 50.0

    def test_negative_range_midpoint(self):
        assert normalize_0_100(0, -10, 10) == 50.0

    def test_out_of_range_exceeds_100(self):
        assert normalize_0_100(150, 0, 100) == 150.0

    def test_equal_min_max_raises(self):
        with pytest.raises(ValueError, match="must differ"):
            normalize_0_100(5, 5, 5)

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError, match="numeric type"):
            normalize_0_100("x", 0, 100)
