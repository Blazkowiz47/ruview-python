"""Attention-gated spectrogram helpers for RuVector-style signal features."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def gate_spectrogram(
    spectrogram: ArrayLike,
    n_freq: int | None = None,
    n_time: int | None = None,
    lambda_: float = 0.1,
    *,
    tau: int = 2,
    eps: float = 1e-7,
    time_axis: int = 0,
    **kwargs: object,
) -> FloatArray:
    """Suppress low-energy spectrogram frames and amplify motion frames.

    Flat inputs follow the Rust reference layout: ``n_time`` row-major frames,
    each with ``n_freq`` frequency bins. Two-dimensional inputs are returned in
    their original shape; use ``time_axis`` when time is stored on columns.
    """

    if "lambda" in kwargs:
        lambda_ = float(kwargs.pop("lambda"))
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected keyword argument(s): {unknown}")

    lambda_value = _validate_lambda(lambda_)
    if tau < 0:
        raise ValueError("tau must be non-negative")
    if eps <= 0.0 or not np.isfinite(float(eps)):
        raise ValueError("eps must be positive and finite")

    values = np.asarray(spectrogram, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("spectrogram values must be finite")

    if values.ndim == 1:
        _validate_flat_shape(values.size, n_freq, n_time)
        if values.size == 0:
            return values.copy()
        frames = values.reshape((int(n_time), int(n_freq)))
        gated = _gate_frames(frames, lambda_value=lambda_value, tau=int(tau), eps=float(eps))
        return gated.reshape(values.shape)

    if values.ndim == 2:
        if values.size == 0:
            return values.copy()
        axis = _normalize_axis(time_axis, values.ndim)
        frames = np.moveaxis(values, axis, 0)
        gated = _gate_frames(frames, lambda_value=lambda_value, tau=int(tau), eps=float(eps))
        return np.moveaxis(gated, 0, axis).astype(np.float64, copy=False)

    raise ValueError(f"gate_spectrogram expects a flat or 2D array, got shape {values.shape}")


def _gate_frames(
    frames: FloatArray,
    *,
    lambda_value: float,
    tau: int,
    eps: float,
) -> FloatArray:
    if frames.shape[0] == 0 or frames.shape[1] == 0:
        return frames.copy()

    energy = np.sqrt(np.mean(np.abs(frames) ** 2, axis=1))
    energy_min = float(np.min(energy))
    energy_max = float(np.max(energy))
    span = energy_max - energy_min
    if span <= eps:
        return frames.copy() if energy_max > eps else np.zeros_like(frames, dtype=np.float64)

    normalized_energy = (energy - energy_min) / span
    strong = normalized_energy >= max(lambda_value, eps)
    if not np.any(strong) and energy_max > eps:
        strong[int(np.argmax(normalized_energy))] = True

    support = np.where(normalized_energy >= lambda_value * 0.5, normalized_energy, 0.0)
    strong_indices = np.flatnonzero(strong)
    if strong_indices.size:
        for index in strong_indices:
            start = max(0, int(index) - tau)
            stop = min(frames.shape[0], int(index) + tau + 1)
            for neighbor in range(start, stop):
                distance = abs(neighbor - int(index))
                decay = 1.0 - (distance / float(tau + 1))
                support[neighbor] = max(support[neighbor], decay)

    low_scale = max(eps, 1.0 - lambda_value)
    high_scale = 1.0 + lambda_value
    scale = low_scale + (high_scale - low_scale) * np.clip(support, 0.0, 1.0)
    return (frames * scale[:, np.newaxis]).astype(np.float64, copy=False)


def _validate_lambda(value: float) -> float:
    lambda_value = float(value)
    if not np.isfinite(lambda_value) or not 0.0 <= lambda_value <= 1.0:
        raise ValueError("lambda_ must be finite and between 0 and 1")
    return lambda_value


def _validate_flat_shape(size: int, n_freq: int | None, n_time: int | None) -> None:
    if n_freq is None or n_time is None:
        raise ValueError("flat spectrograms require n_freq and n_time")
    if int(n_freq) < 0 or int(n_time) < 0:
        raise ValueError("n_freq and n_time must be non-negative")
    if int(n_freq) * int(n_time) != size:
        raise ValueError(
            "flat spectrogram length must equal n_freq * n_time; "
            f"got {size} values for n_freq={n_freq}, n_time={n_time}"
        )


def _normalize_axis(axis: int, ndim: int) -> int:
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis % ndim
