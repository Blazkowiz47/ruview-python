"""DTW gesture template matching for short CSI feature sequences."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class GestureError(ValueError):
    """Base error for gesture operations."""


class GestureType(str, Enum):
    """Built-in gesture labels used by the template classifier."""

    WAVE = "wave"
    POINT = "point"
    BECKON = "beckon"
    PUSH = "push"
    CIRCLE = "circle"
    CUSTOM = "custom"

    @classmethod
    def from_value(cls, value: "GestureType | str") -> "GestureType":
        if isinstance(value, GestureType):
            return value
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            valid = ", ".join(item.value for item in cls)
            raise GestureError(f"unknown gesture type {value!r}; expected one of {valid}") from exc


@dataclass(frozen=True)
class GestureConfig:
    """Configuration for DTW template matching."""

    feature_dim: int = 8
    min_sequence_len: int = 10
    max_distance: float = 50.0
    band_width: int | None = 5
    normalize: bool = False

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise GestureError("feature_dim must be positive")
        if self.min_sequence_len <= 0:
            raise GestureError("min_sequence_len must be positive")
        if self.max_distance <= 0.0 or not math.isfinite(self.max_distance):
            raise GestureError("max_distance must be a finite positive value")
        if self.band_width is not None and self.band_width < 0:
            raise GestureError("band_width must be non-negative or None")


@dataclass(frozen=True)
class GestureTemplate:
    """Reference sequence for a named gesture."""

    name: str
    gesture_type: GestureType | str
    sequence: ArrayLike
    feature_dim: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise GestureError("template name cannot be empty")
        sequence = _as_sequence(self.sequence)
        gesture_type = GestureType.from_value(self.gesture_type)
        feature_dim = sequence.shape[1] if self.feature_dim is None else int(self.feature_dim)
        if feature_dim <= 0:
            raise GestureError("template feature_dim must be positive")
        if sequence.shape[1] != feature_dim:
            raise GestureError(
                f"feature dimension mismatch: expected {feature_dim}, got {sequence.shape[1]}"
            )
        object.__setattr__(self, "gesture_type", gesture_type)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "feature_dim", feature_dim)


@dataclass(frozen=True)
class GestureResult:
    """Nearest-template classification result."""

    recognized: bool
    gesture_type: GestureType | None
    template_name: str | None
    distance: float
    confidence: float
    person_id: int = 0
    timestamp_us: int = 0


class GestureClassifier:
    """Classifies feature trajectories by nearest constrained DTW template."""

    def __init__(self, config: GestureConfig | None = None) -> None:
        self.config = config if config is not None else GestureConfig()
        self._templates: list[GestureTemplate] = []

    @property
    def templates(self) -> tuple[GestureTemplate, ...]:
        return tuple(self._templates)

    @property
    def template_count(self) -> int:
        return len(self._templates)

    def add_template(self, template: GestureTemplate) -> None:
        """Register a validated template for later classification."""

        if template.feature_dim != self.config.feature_dim:
            raise GestureError(
                f"feature dimension mismatch: expected {self.config.feature_dim}, "
                f"got {template.feature_dim}"
            )
        if len(template.sequence) < self.config.min_sequence_len:
            raise GestureError(
                f"sequence too short: need >= {self.config.min_sequence_len}, "
                f"got {len(template.sequence)}"
            )
        self._templates.append(template)

    def add_templates(self, templates: Iterable[GestureTemplate]) -> None:
        for template in templates:
            self.add_template(template)

    def classify(
        self,
        sequence: ArrayLike,
        *,
        person_id: int = 0,
        timestamp_us: int = 0,
    ) -> GestureResult:
        """Return the nearest gesture template for a CSI feature sequence."""

        if not self._templates:
            raise GestureError("no gesture templates registered")

        query = _as_sequence(sequence, expected_dim=self.config.feature_dim)
        if len(query) < self.config.min_sequence_len:
            raise GestureError(
                f"sequence too short: need >= {self.config.min_sequence_len}, got {len(query)}"
            )

        query_for_match = _z_normalize(query) if self.config.normalize else query
        distances: list[tuple[float, GestureTemplate]] = []
        for template in self._templates:
            candidate = _z_normalize(template.sequence) if self.config.normalize else template.sequence
            dist = dtw_distance(query_for_match, candidate, band_width=self.config.band_width)
            distances.append((dist, template))

        distances.sort(key=lambda item: item[0])
        best_distance, best_template = distances[0]
        second_best = distances[1][0] if len(distances) > 1 else math.inf
        recognized = math.isfinite(best_distance) and best_distance <= self.config.max_distance

        if not recognized:
            return GestureResult(
                recognized=False,
                gesture_type=None,
                template_name=None,
                distance=float(best_distance),
                confidence=0.0,
                person_id=int(person_id),
                timestamp_us=int(timestamp_us),
            )

        confidence = _classification_confidence(best_distance, second_best, self.config.max_distance)
        return GestureResult(
            recognized=True,
            gesture_type=best_template.gesture_type,
            template_name=best_template.name,
            distance=float(best_distance),
            confidence=confidence,
            person_id=int(person_id),
            timestamp_us=int(timestamp_us),
        )


def dtw_distance(
    seq_a: ArrayLike,
    seq_b: ArrayLike,
    *,
    band_width: int | None = None,
) -> float:
    """Compute constrained Dynamic Time Warping distance between two sequences."""

    a = _as_sequence(seq_a)
    b = _as_sequence(seq_b)
    if a.shape[1] != b.shape[1]:
        raise GestureError(f"feature dimension mismatch: expected {a.shape[1]}, got {b.shape[1]}")
    if len(a) == 0 or len(b) == 0:
        return math.inf

    n = len(a)
    m = len(b)
    if band_width is None:
        band = max(n, m)
    else:
        if band_width < 0:
            raise GestureError("band_width must be non-negative or None")
        band = max(int(band_width), abs(n - m))

    previous = np.full(m + 1, np.inf, dtype=np.float64)
    current = np.full(m + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0

    for i in range(1, n + 1):
        current.fill(np.inf)
        start = max(1, i - band)
        end = min(m, i + band)
        for j in range(start, end + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous, current = current, previous

    return float(previous[m])


def _as_sequence(sequence: ArrayLike, expected_dim: int | None = None) -> FloatArray:
    arr = np.asarray(sequence, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise GestureError("gesture sequence must be a 2-D array of frames by features")
    if expected_dim is not None and arr.shape[1] != expected_dim:
        raise GestureError(f"feature dimension mismatch: expected {expected_dim}, got {arr.shape[1]}")
    if not np.all(np.isfinite(arr)):
        raise GestureError("gesture sequence contains non-finite values")
    return np.array(arr, dtype=np.float64, copy=True)


def _z_normalize(sequence: FloatArray) -> FloatArray:
    mean = np.mean(sequence, axis=0, keepdims=True)
    std = np.std(sequence, axis=0, keepdims=True)
    return (sequence - mean) / np.maximum(std, 1e-12)


def _classification_confidence(best: float, second: float, max_distance: float) -> float:
    if math.isfinite(second) and second > 1e-12:
        return float(np.clip(1.0 - best / second, 0.0, 1.0))
    return float(np.clip(1.0 - best / max_distance, 0.0, 1.0))


__all__ = [
    "GestureClassifier",
    "GestureConfig",
    "GestureError",
    "GestureResult",
    "GestureTemplate",
    "GestureType",
    "dtw_distance",
]
