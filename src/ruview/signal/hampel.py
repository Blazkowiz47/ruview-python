"""Hampel filtering for robust 1D outlier replacement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


MAD_SCALE = 1.4826
_EPSILON = 1e-15


@dataclass(frozen=True)
class HampelResult:
    """Result arrays from a Hampel filter pass."""

    filtered: NDArray[np.float64]
    outlier_indices: NDArray[np.int64]
    medians: NDArray[np.float64]
    sigma_estimates: NDArray[np.float64]


def hampel_filter(
    signal: ArrayLike,
    *,
    half_window: int = 3,
    threshold: float = 3.0,
) -> HampelResult:
    """Replace 1D outliers with local medians using the Hampel rule."""

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"hampel_filter expects a 1D array, got shape {values.shape}")
    if values.size == 0:
        raise ValueError("hampel_filter requires a non-empty signal")
    if half_window <= 0:
        raise ValueError("half_window must be positive")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("hampel_filter requires finite signal values")

    filtered = values.copy()
    medians = np.empty_like(values)
    sigma_estimates = np.empty_like(values)
    outliers: list[int] = []

    for index, value in enumerate(values):
        start = max(0, index - half_window)
        end = min(values.size, index + half_window + 1)
        window = values[start:end]

        median = float(np.median(window))
        mad = float(np.median(np.abs(window - median)))
        sigma = MAD_SCALE * mad

        medians[index] = median
        sigma_estimates[index] = sigma

        deviation = abs(float(value) - median)
        is_outlier = deviation > threshold * sigma if sigma > _EPSILON else deviation > _EPSILON
        if is_outlier:
            filtered[index] = median
            outliers.append(index)

    return HampelResult(
        filtered=filtered,
        outlier_indices=np.asarray(outliers, dtype=np.int64),
        medians=medians,
        sigma_estimates=sigma_estimates,
    )
