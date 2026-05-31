"""RuVector-style subcarrier partitioning and importance weights."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def mincut_subcarrier_partition(sensitivity: ArrayLike) -> tuple[list[int], list[int]]:
    """Partition subcarrier indices into sensitive and insensitive groups.

    The Rust reference builds a graph with high pairwise weights between
    similarly sensitive subcarriers. This NumPy port preserves that behavior by
    cutting the largest gap in sorted sensitivity scores, then naming the group
    with the higher mean score ``sensitive``.
    """

    scores = _as_sensitivity_vector(sensitivity)
    n_subcarriers = int(scores.size)
    if n_subcarriers == 0:
        return ([], [])
    if n_subcarriers == 1:
        return ([0], [])

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    gaps = np.diff(sorted_scores)

    if gaps.size == 0 or float(np.max(gaps)) <= 1e-12:
        midpoint = n_subcarriers // 2
        side_a = list(range(midpoint))
        side_b = list(range(midpoint, n_subcarriers))
    else:
        split = int(np.argmax(gaps)) + 1
        side_a = [int(index) for index in order[split:]]
        side_b = [int(index) for index in order[:split]]

    if not side_a or not side_b:
        midpoint = n_subcarriers // 2
        side_a = list(range(midpoint))
        side_b = list(range(midpoint, n_subcarriers))

    return _higher_mean_first(side_a, side_b, scores)


def subcarrier_partition(sensitivity: ArrayLike) -> tuple[list[int], list[int]]:
    """Compatibility alias for :func:`mincut_subcarrier_partition`."""

    return mincut_subcarrier_partition(sensitivity)


def subcarrier_importance_weights(sensitivity: ArrayLike) -> FloatArray:
    """Return per-subcarrier weights that emphasize the sensitive partition."""

    scores = _as_sensitivity_vector(sensitivity)
    if scores.size == 0:
        return np.zeros(0, dtype=np.float64)

    sensitive, _insensitive = mincut_subcarrier_partition(scores)
    weights = np.full(scores.shape, 0.5, dtype=np.float64)
    if not sensitive:
        return weights

    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    span = score_max - score_min
    if span <= 1e-12:
        scaled = np.ones(scores.shape, dtype=np.float64)
    else:
        scaled = (scores - score_min) / span

    sensitive_indices = np.asarray(sensitive, dtype=np.int64)
    weights[sensitive_indices] = 1.0 + np.clip(scaled[sensitive_indices], 0.0, 1.0)
    return weights


def _higher_mean_first(
    side_a: Sequence[int],
    side_b: Sequence[int],
    scores: FloatArray,
) -> tuple[list[int], list[int]]:
    mean_a = _mean_for_indices(side_a, scores)
    mean_b = _mean_for_indices(side_b, scores)
    if mean_a >= mean_b:
        return (list(side_a), list(side_b))
    return (list(side_b), list(side_a))


def _mean_for_indices(indices: Sequence[int], scores: FloatArray) -> float:
    if not indices:
        return 0.0
    return float(np.mean(scores[np.asarray(indices, dtype=np.int64)]))


def _as_sensitivity_vector(sensitivity: ArrayLike) -> FloatArray:
    scores = np.asarray(sensitivity, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(scores)):
        raise ValueError("sensitivity scores must be finite")
    return scores
