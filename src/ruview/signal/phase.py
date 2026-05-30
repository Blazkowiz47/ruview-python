"""Phase-domain signal helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def unwrap_phase(
    phase: ArrayLike,
    *,
    axis: int = -1,
    discontinuity: float | None = None,
    period: float = 2.0 * np.pi,
) -> NDArray[np.float64]:
    """Unwrap phase values along one axis."""

    phase_array = np.asarray(phase, dtype=np.float64)
    if phase_array.size == 0:
        return phase_array.copy()

    kwargs: dict[str, float | int] = {"axis": axis, "period": period}
    if discontinuity is not None:
        kwargs["discont"] = discontinuity
    return np.unwrap(phase_array, **kwargs).astype(np.float64, copy=False)
