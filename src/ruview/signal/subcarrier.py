"""Subcarrier-level summary primitives."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def subcarrier_variance(
    data: ArrayLike,
    *,
    time_axis: int = 0,
    subcarrier_axis: int = -1,
    ddof: int = 1,
    aggregate_other_axes: bool = True,
) -> NDArray[np.float64]:
    """Return variance across time for each subcarrier.

    For a ``[time, stream, subcarrier]`` window, the default returns one
    variance score per subcarrier by averaging per-stream variances.
    """

    values = np.asarray(data, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError(f"subcarrier_variance expects at least 2 dimensions, got {values.shape}")
    if ddof < 0:
        raise ValueError("ddof must be non-negative")

    time_axis = _normalize_axis(time_axis, values.ndim)
    subcarrier_axis = _normalize_axis(subcarrier_axis, values.ndim)
    if time_axis == subcarrier_axis:
        raise ValueError("time_axis and subcarrier_axis must be different")

    output_shape = tuple(dim for axis, dim in enumerate(values.shape) if axis != time_axis)
    if values.shape[time_axis] <= ddof:
        variances = np.zeros(output_shape, dtype=np.float64)
    else:
        variances = np.var(values, axis=time_axis, ddof=ddof).astype(np.float64, copy=False)

    if not aggregate_other_axes or variances.ndim == 1:
        return variances

    result_subcarrier_axis = subcarrier_axis - 1 if subcarrier_axis > time_axis else subcarrier_axis
    axes_to_average = tuple(
        axis for axis in range(variances.ndim) if axis != result_subcarrier_axis
    )
    return np.mean(variances, axis=axes_to_average).astype(np.float64, copy=False)


def _normalize_axis(axis: int, ndim: int) -> int:
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis % ndim
