"""Attention-weighted body velocity profile aggregation."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def attention_weighted_bvp(
    stft_rows: ArrayLike,
    sensitivity: ArrayLike | None,
    n_velocity_bins: int | None = None,
    *,
    eps: float = 1e-12,
) -> FloatArray:
    """Aggregate per-subcarrier STFT rows into one body velocity profile."""

    rows, n_bins = _coerce_stft_rows(stft_rows, n_velocity_bins)
    if n_bins == 0 or rows.shape[0] == 0:
        return np.zeros(n_bins, dtype=np.float64)

    weights = bvp_attention_weights(rows, sensitivity, n_velocity_bins=n_bins, eps=eps)
    return (weights @ rows[:, :n_bins]).astype(np.float64, copy=False)


def bvp_attention_weights(
    stft_rows: ArrayLike,
    sensitivity: ArrayLike | None,
    n_velocity_bins: int | None = None,
    *,
    eps: float = 1e-12,
) -> FloatArray:
    """Return scaled dot-product attention weights for STFT/BVP rows."""

    if eps <= 0.0 or not np.isfinite(float(eps)):
        raise ValueError("eps must be positive and finite")

    rows, n_bins = _coerce_stft_rows(stft_rows, n_velocity_bins)
    n_rows = int(rows.shape[0])
    if n_bins == 0 or n_rows == 0:
        return np.zeros(n_rows, dtype=np.float64)

    rows = rows[:, :n_bins]
    priors = _coerce_sensitivity_prior(sensitivity, n_rows, eps=float(eps))
    query = priors @ rows
    logits = (rows @ query) / math.sqrt(float(n_bins))
    logits = logits + np.log(priors + float(eps))
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(n_rows, 1.0 / n_rows, dtype=np.float64)
    return (weights / total).astype(np.float64, copy=False)


def _coerce_stft_rows(
    stft_rows: ArrayLike,
    n_velocity_bins: int | None,
) -> tuple[FloatArray, int]:
    rows = np.asarray(stft_rows, dtype=np.float64)
    if not np.all(np.isfinite(rows)):
        raise ValueError("stft_rows values must be finite")

    if rows.ndim == 1:
        if rows.size == 0:
            n_bins = 0 if n_velocity_bins is None else _validate_n_bins(n_velocity_bins)
            return np.zeros((0, n_bins), dtype=np.float64), n_bins
        rows = rows.reshape(1, -1)
    elif rows.ndim != 2:
        raise ValueError(f"stft_rows must be a 1D or 2D array, got shape {rows.shape}")

    if n_velocity_bins is None:
        n_bins = int(rows.shape[1])
    else:
        n_bins = _validate_n_bins(n_velocity_bins)
    if n_bins > rows.shape[1]:
        raise ValueError(
            "n_velocity_bins must not exceed the STFT row width; "
            f"got n_velocity_bins={n_bins}, row width={rows.shape[1]}"
        )
    return rows.astype(np.float64, copy=False), n_bins


def _coerce_sensitivity_prior(
    sensitivity: ArrayLike | None,
    n_rows: int,
    *,
    eps: float,
) -> FloatArray:
    if sensitivity is None:
        return np.full(n_rows, 1.0 / n_rows, dtype=np.float64)

    values = np.asarray(sensitivity, dtype=np.float64).reshape(-1)
    if values.size != n_rows:
        raise ValueError("sensitivity length must match the number of STFT rows")
    if not np.all(np.isfinite(values)):
        raise ValueError("sensitivity values must be finite")

    clipped = np.clip(values, 0.0, None)
    total = float(np.sum(clipped))
    if total <= eps:
        return np.full(n_rows, 1.0 / n_rows, dtype=np.float64)
    return (clipped / total).astype(np.float64, copy=False)


def _validate_n_bins(n_velocity_bins: int) -> int:
    n_bins = int(n_velocity_bins)
    if n_bins < 0:
        raise ValueError("n_velocity_bins must be non-negative")
    return n_bins
