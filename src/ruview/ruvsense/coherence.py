"""Coherence scoring for consecutive CSI or CIR observations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ruview.core import CsiFrame
from ruview.signal.csi_processor import CsiWindow


ComplexArray = NDArray[np.complex128]


class CoherenceError(ValueError):
    """Raised when coherence inputs are invalid."""


@dataclass(frozen=True)
class CoherenceConfig:
    """Weights and numeric guards for combined coherence scoring."""

    correlation_weight: float = 0.45
    magnitude_weight: float = 0.35
    phase_weight: float = 0.20
    epsilon: float = 1e-12
    accept_threshold: float = 0.85

    def __post_init__(self) -> None:
        weights = (self.correlation_weight, self.magnitude_weight, self.phase_weight)
        if any(weight < 0.0 or not math.isfinite(weight) for weight in weights):
            raise CoherenceError("coherence weights must be finite and non-negative")
        if sum(weights) <= 0.0:
            raise CoherenceError("at least one coherence weight must be positive")
        if self.epsilon <= 0.0 or not math.isfinite(self.epsilon):
            raise CoherenceError("epsilon must be positive and finite")
        if not 0.0 <= self.accept_threshold <= 1.0:
            raise CoherenceError("accept_threshold must be in [0, 1]")

    @property
    def weight_sum(self) -> float:
        return float(self.correlation_weight + self.magnitude_weight + self.phase_weight)


@dataclass(frozen=True)
class CoherenceComponents:
    """Individual coherence terms and their weighted combined score."""

    correlation: float
    magnitude_similarity: float
    phase_stability: float
    score: float
    sample_count: int

    @property
    def combined_score(self) -> float:
        return self.score


@dataclass
class CoherenceTracker:
    """Small EMA tracker for scoring observations against an accepted reference."""

    config: CoherenceConfig = CoherenceConfig()
    reference: ComplexArray | None = None
    ema_decay: float = 0.95
    stale_count: int = 0
    current_score: float = 1.0
    last_components: CoherenceComponents | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.ema_decay < 1.0:
            raise CoherenceError("ema_decay must be in (0, 1)")
        if self.reference is not None:
            self.reference = _coerce_complex_vector(self.reference)

    @property
    def initialized(self) -> bool:
        return self.reference is not None

    def initialize(self, observation: CsiWindow | CsiFrame | ArrayLike) -> CoherenceComponents:
        self.reference = _coerce_complex_vector(observation)
        self.stale_count = 0
        self.current_score = 1.0
        self.last_components = CoherenceComponents(1.0, 1.0, 1.0, 1.0, int(self.reference.size))
        return self.last_components

    def update(self, observation: CsiWindow | CsiFrame | ArrayLike) -> CoherenceComponents:
        current = _coerce_complex_vector(observation)
        if self.reference is None:
            return self.initialize(current)
        components = coherence_components(current, self.reference, self.config)
        self.current_score = components.score
        self.last_components = components
        if components.score >= self.config.accept_threshold:
            alpha = 1.0 - self.ema_decay
            self.reference = (self.ema_decay * self.reference + alpha * current).astype(
                np.complex128,
                copy=False,
            )
            self.stale_count = 0
        else:
            self.stale_count += 1
        return components

    def reset_stale(self) -> None:
        self.stale_count = 0


def normalized_correlation(
    current: CsiWindow | CsiFrame | ArrayLike,
    reference: CsiWindow | CsiFrame | ArrayLike,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return absolute complex normalized correlation in ``[0, 1]``."""

    a, b = _paired_vectors(current, reference)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= epsilon and norm_b <= epsilon:
        return 1.0
    if norm_a <= epsilon or norm_b <= epsilon:
        return 0.0
    score = abs(np.vdot(b, a)) / (norm_a * norm_b)
    return _clip01(score)


def magnitude_similarity(
    current: CsiWindow | CsiFrame | ArrayLike,
    reference: CsiWindow | CsiFrame | ArrayLike,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return L2 magnitude agreement in ``[0, 1]``."""

    a, b = _paired_vectors(current, reference)
    mag_a = np.abs(a)
    mag_b = np.abs(b)
    denom = float(np.linalg.norm(mag_a) + np.linalg.norm(mag_b))
    if denom <= epsilon:
        return 1.0
    distance = float(np.linalg.norm(mag_a - mag_b))
    return _clip01(1.0 - distance / denom)


def phase_stability(
    current: CsiWindow | CsiFrame | ArrayLike,
    reference: CsiWindow | CsiFrame | ArrayLike,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return weighted circular phase consistency in ``[0, 1]``."""

    a, b = _paired_vectors(current, reference)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= epsilon and norm_b <= epsilon:
        return 1.0
    if norm_a <= epsilon or norm_b <= epsilon:
        return 0.0

    weights = np.abs(a) * np.abs(b)
    weight_sum = float(np.sum(weights))
    if weight_sum <= epsilon:
        return 0.0
    phase_delta = np.angle(a * np.conjugate(b))
    resultant = np.sum(weights * np.exp(1j * phase_delta)) / weight_sum
    return _clip01(abs(resultant))


def coherence_components(
    current: CsiWindow | CsiFrame | ArrayLike,
    reference: CsiWindow | CsiFrame | ArrayLike,
    config: CoherenceConfig | None = None,
) -> CoherenceComponents:
    """Compute correlation, magnitude, phase, and combined coherence."""

    cfg = config or CoherenceConfig()
    a, b = _paired_vectors(current, reference)
    corr = normalized_correlation(a, b, epsilon=cfg.epsilon)
    mag = magnitude_similarity(a, b, epsilon=cfg.epsilon)
    phase = phase_stability(a, b, epsilon=cfg.epsilon)
    combined = (
        cfg.correlation_weight * corr
        + cfg.magnitude_weight * mag
        + cfg.phase_weight * phase
    ) / cfg.weight_sum
    return CoherenceComponents(
        correlation=corr,
        magnitude_similarity=mag,
        phase_stability=phase,
        score=_clip01(combined),
        sample_count=int(a.size),
    )


def coherence_score(
    current: CsiWindow | CsiFrame | ArrayLike,
    reference: CsiWindow | CsiFrame | ArrayLike,
    config: CoherenceConfig | None = None,
) -> float:
    """Return the combined coherence score in ``[0, 1]``."""

    return coherence_components(current, reference, config).score


def consecutive_coherence(
    observations: list[CsiWindow | CsiFrame | ArrayLike] | tuple[CsiWindow | CsiFrame | ArrayLike, ...],
    config: CoherenceConfig | None = None,
) -> tuple[CoherenceComponents, ...]:
    """Score each consecutive pair in a sequence of CSI/CIR observations."""

    if len(observations) < 2:
        return ()
    return tuple(
        coherence_components(current, previous, config)
        for previous, current in zip(observations[:-1], observations[1:], strict=True)
    )


def _paired_vectors(
    current: CsiWindow | CsiFrame | ArrayLike,
    reference: CsiWindow | CsiFrame | ArrayLike,
) -> tuple[ComplexArray, ComplexArray]:
    a = _coerce_complex_vector(current)
    b = _coerce_complex_vector(reference)
    if a.shape != b.shape:
        raise CoherenceError(f"coherence inputs must have matching shapes, got {a.shape} and {b.shape}")
    return a, b


def _coerce_complex_vector(source: CsiWindow | CsiFrame | ArrayLike) -> ComplexArray:
    if isinstance(source, CsiFrame):
        array = np.asarray(source.data, dtype=np.complex128)
    elif isinstance(source, CsiWindow):
        array = np.asarray(source.data, dtype=np.complex128)
    else:
        array = np.asarray(source, dtype=np.complex128)
    if array.size == 0:
        raise CoherenceError("coherence input must not be empty")
    if not np.all(np.isfinite(array.real) & np.isfinite(array.imag)):
        raise CoherenceError("coherence input contains non-finite values")
    return array.ravel().astype(np.complex128, copy=False)


def _clip01(value: float | np.floating) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


__all__ = [
    "CoherenceComponents",
    "CoherenceConfig",
    "CoherenceError",
    "CoherenceTracker",
    "coherence_components",
    "coherence_score",
    "consecutive_coherence",
    "magnitude_similarity",
    "normalized_correlation",
    "phase_stability",
]
