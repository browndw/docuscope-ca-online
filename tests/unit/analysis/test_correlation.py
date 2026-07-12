"""Tests for lightweight correlation helpers."""

import math

import pytest

from webapp.utilities.analysis.correlation import pearson_correlation


def test_pearson_correlation_computes_r_without_scipy():
    result = pearson_correlation([1, 2, 3, 4], [2, 4, 6, 8])

    assert result.statistic == pytest.approx(1.0)
    assert result.pvalue == 0.0


def test_pearson_correlation_uses_pairwise_finite_values():
    result = pearson_correlation([1, 2, float("nan"), 4], [1, 2, 3, 4])

    assert result.statistic == pytest.approx(1.0)
    assert result.pvalue == 0.0


def test_pearson_correlation_returns_nan_for_constant_input():
    result = pearson_correlation([1, 1, 1, 1], [1, 2, 3, 4])

    assert math.isnan(result.statistic)
    assert math.isnan(result.pvalue)


@pytest.mark.parametrize(
    ("x_values", "y_values", "expected_r", "expected_p"),
    [
        (
            [1, 2, 3, 4, 5],
            [1, 2, 1.3, 3.75, 2.25],
            0.6268327489789578,
            0.25777730285388845,
        ),
        (
            [10, 20, 30, 40, 50, 60],
            [11, 19, 33, 39, 52, 61],
            0.9965529841647024,
            1.780239867314677e-05,
        ),
        (
            [2, 4, 6, 8, 10, 12],
            [1, 3, 2, 5, 7, 8],
            0.9528852579245707,
            0.0032774057544441185,
        ),
    ],
)
def test_pearson_correlation_matches_reference_values(
    x_values, y_values, expected_r, expected_p
):
    result = pearson_correlation(x_values, y_values)

    assert result.statistic == pytest.approx(expected_r)
    assert result.pvalue == pytest.approx(expected_p)
