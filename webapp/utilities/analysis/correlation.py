"""Small correlation helpers without a SciPy runtime dependency."""

from dataclasses import dataclass
from math import exp, lgamma, log, log1p, sqrt
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PearsonResult:
    """Pearson correlation result with a two-sided p-value."""

    statistic: float
    pvalue: float


def _regularized_incomplete_beta(value: float, a: float, b: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0

    def continued_fraction(x_value: float, a_value: float, b_value: float) -> float:
        max_iterations = 200
        epsilon = 3e-14
        fp_min = 1e-300

        qab = a_value + b_value
        qap = a_value + 1.0
        qam = a_value - 1.0
        c_value = 1.0
        d_value = 1.0 - qab * x_value / qap
        if abs(d_value) < fp_min:
            d_value = fp_min
        d_value = 1.0 / d_value
        h_value = d_value

        for iteration in range(1, max_iterations + 1):
            m2 = 2 * iteration
            aa = iteration * (b_value - iteration) * x_value / (
                (qam + m2) * (a_value + m2)
            )
            d_value = 1.0 + aa * d_value
            if abs(d_value) < fp_min:
                d_value = fp_min
            c_value = 1.0 + aa / c_value
            if abs(c_value) < fp_min:
                c_value = fp_min
            d_value = 1.0 / d_value
            h_value *= d_value * c_value

            aa = -(
                (a_value + iteration)
                * (qab + iteration)
                * x_value
                / ((a_value + m2) * (qap + m2))
            )
            d_value = 1.0 + aa * d_value
            if abs(d_value) < fp_min:
                d_value = fp_min
            c_value = 1.0 + aa / c_value
            if abs(c_value) < fp_min:
                c_value = fp_min
            d_value = 1.0 / d_value
            delta = d_value * c_value
            h_value *= delta
            if abs(delta - 1.0) < epsilon:
                break

        return h_value

    log_beta = lgamma(a) + lgamma(b) - lgamma(a + b)
    front = exp(a * log(value) + b * log1p(-value) - log_beta)
    if value < (a + 1.0) / (a + b + 2.0):
        return front * continued_fraction(value, a, b) / a
    return 1.0 - front * continued_fraction(1.0 - value, b, a) / b


def _pearson_pvalue(r: float, n: int) -> float:
    if abs(r) == 1.0:
        return 0.0
    degrees_of_freedom = n - 2
    t_squared = (r * r) * degrees_of_freedom / (1.0 - r * r)
    beta_value = degrees_of_freedom / (degrees_of_freedom + t_squared)
    return _regularized_incomplete_beta(beta_value, degrees_of_freedom / 2.0, 0.5)


def pearson_correlation(x: Iterable[float], y: Iterable[float]) -> PearsonResult:
    """Compute Pearson's r and a two-sided p-value."""
    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)
    valid_mask = np.isfinite(x_array) & np.isfinite(y_array)
    x_valid = x_array[valid_mask]
    y_valid = y_array[valid_mask]
    n = len(x_valid)
    if n < 3:
        return PearsonResult(statistic=float("nan"), pvalue=float("nan"))

    x_centered = x_valid - x_valid.mean()
    y_centered = y_valid - y_valid.mean()
    denominator = sqrt(float(np.sum(x_centered ** 2) * np.sum(y_centered ** 2)))
    if denominator == 0:
        return PearsonResult(statistic=float("nan"), pvalue=float("nan"))

    r = float(np.sum(x_centered * y_centered) / denominator)
    r = max(min(r, 1.0), -1.0)
    if abs(r) == 1.0:
        return PearsonResult(statistic=r, pvalue=0.0)
    return PearsonResult(statistic=r, pvalue=_pearson_pvalue(r, n))
