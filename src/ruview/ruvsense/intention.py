"""Pre-movement intention scoring from CSI lead features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Deque
from collections import deque

import numpy as np


class IntentionError(ValueError):
    """Base error for intention detection operations."""


class IntentLabel(str, Enum):
    """Coarse labels for pre-movement lead scoring."""

    NONE = "none"
    PREPARING = "preparing"
    LIKELY = "likely"
    MOTION = "motion"


@dataclass(frozen=True)
class IntentionConfig:
    """Thresholds for the slope/energy/phase lead detector."""

    window_size: int = 12
    slope_threshold: float = 0.04
    energy_threshold: float = 0.08
    phase_lead_threshold: float = 0.12
    motion_onset_energy: float = 0.45
    score_threshold: float = 0.62
    preparing_threshold: float = 0.38
    min_sustained_frames: int = 3
    max_lead_time_s: float = 0.5

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise IntentionError("window_size must be positive")
        if self.min_sustained_frames <= 0:
            raise IntentionError("min_sustained_frames must be positive")
        for name in (
            "slope_threshold",
            "energy_threshold",
            "phase_lead_threshold",
            "motion_onset_energy",
            "score_threshold",
            "preparing_threshold",
            "max_lead_time_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise IntentionError(f"{name} must be a finite positive value")
        if self.energy_threshold >= self.motion_onset_energy:
            raise IntentionError("energy_threshold must be lower than motion_onset_energy")
        if self.preparing_threshold > self.score_threshold:
            raise IntentionError("preparing_threshold must be <= score_threshold")


@dataclass(frozen=True)
class IntentionFeatures:
    """Lead features extracted from a short CSI window."""

    slope: float
    energy: float
    phase_lead: float
    timestamp_us: int = 0

    def __post_init__(self) -> None:
        values = (self.slope, self.energy, self.phase_lead)
        if not all(math.isfinite(float(value)) for value in values):
            raise IntentionError("intention features must be finite")
        if self.energy < 0.0:
            raise IntentionError("energy must be non-negative")
        object.__setattr__(self, "slope", float(self.slope))
        object.__setattr__(self, "energy", float(self.energy))
        object.__setattr__(self, "phase_lead", float(self.phase_lead))
        object.__setattr__(self, "timestamp_us", int(self.timestamp_us))


@dataclass(frozen=True)
class IntentSignal:
    """Intent score and label for one update."""

    detected: bool
    label: IntentLabel
    score: float
    slope_score: float
    energy_score: float
    phase_lead_score: float
    sustained_frames: int
    estimated_lead_time_s: float
    timestamp_us: int


class IntentionDetector:
    """Tracks sustained pre-movement evidence before overt motion energy."""

    def __init__(self, config: IntentionConfig | None = None) -> None:
        self.config = config if config is not None else IntentionConfig()
        self._history: Deque[IntentionFeatures] = deque(maxlen=self.config.window_size)
        self._sustained_frames = 0
        self._total_frames = 0

    @property
    def history_len(self) -> int:
        return len(self._history)

    @property
    def sustained_frames(self) -> int:
        return self._sustained_frames

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def reset(self) -> None:
        self._history.clear()
        self._sustained_frames = 0
        self._total_frames = 0

    def update(
        self,
        features: IntentionFeatures | None = None,
        *,
        slope: float | None = None,
        energy: float | None = None,
        phase_lead: float | None = None,
        timestamp_us: int = 0,
    ) -> IntentSignal:
        """Feed one lead-feature frame and return the current intent signal."""

        if features is None:
            if slope is None or energy is None or phase_lead is None:
                raise IntentionError("slope, energy, and phase_lead are required")
            features = IntentionFeatures(
                slope=slope,
                energy=energy,
                phase_lead=phase_lead,
                timestamp_us=timestamp_us,
            )

        self._history.append(features)
        self._total_frames += 1

        signal = score_intention(features, self.config, sustained_frames=self._sustained_frames)
        if signal.label is IntentLabel.MOTION:
            self._sustained_frames = 0
            return signal

        if signal.score >= self.config.score_threshold and features.energy < self.config.motion_onset_energy:
            self._sustained_frames += 1
        else:
            self._sustained_frames = 0

        return score_intention(features, self.config, sustained_frames=self._sustained_frames)


def score_intention(
    features: IntentionFeatures,
    config: IntentionConfig | None = None,
    *,
    sustained_frames: int = 0,
) -> IntentSignal:
    """Score one lead-feature vector without mutating detector state."""

    cfg = config if config is not None else IntentionConfig()
    positive_slope = max(features.slope, 0.0)
    slope_score = float(np.clip(positive_slope / cfg.slope_threshold, 0.0, 1.0))
    energy_score = float(np.clip(features.energy / cfg.energy_threshold, 0.0, 1.0))
    phase_score = float(np.clip(abs(features.phase_lead) / cfg.phase_lead_threshold, 0.0, 1.0))
    score = float(np.clip(0.42 * slope_score + 0.33 * energy_score + 0.25 * phase_score, 0.0, 1.0))

    if features.energy >= cfg.motion_onset_energy:
        return IntentSignal(
            detected=False,
            label=IntentLabel.MOTION,
            score=score,
            slope_score=slope_score,
            energy_score=energy_score,
            phase_lead_score=phase_score,
            sustained_frames=0,
            estimated_lead_time_s=0.0,
            timestamp_us=features.timestamp_us,
        )

    detected = sustained_frames >= cfg.min_sustained_frames and score >= cfg.score_threshold
    if detected:
        label = IntentLabel.LIKELY
    elif score >= cfg.preparing_threshold:
        label = IntentLabel.PREPARING
    else:
        label = IntentLabel.NONE

    lead_time = 0.0
    if detected:
        remaining_energy = max(cfg.motion_onset_energy - features.energy, 0.0)
        lead_time = remaining_energy / max(positive_slope, 1e-9)
        lead_time = float(np.clip(lead_time, 0.0, cfg.max_lead_time_s))

    return IntentSignal(
        detected=detected,
        label=label,
        score=score,
        slope_score=slope_score,
        energy_score=energy_score,
        phase_lead_score=phase_score,
        sustained_frames=int(sustained_frames),
        estimated_lead_time_s=lead_time,
        timestamp_us=features.timestamp_us,
    )


__all__ = [
    "IntentLabel",
    "IntentSignal",
    "IntentionConfig",
    "IntentionDetector",
    "IntentionError",
    "IntentionFeatures",
    "score_intention",
]
