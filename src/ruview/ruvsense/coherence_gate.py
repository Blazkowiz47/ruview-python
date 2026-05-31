"""Coherence gate decisions for downstream pose or tracking updates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from ruview.ruvsense.coherence import CoherenceComponents


class CoherenceGateError(ValueError):
    """Raised when coherence-gate inputs or configuration are invalid."""


class GateDecisionType(str, Enum):
    """Actions emitted by the coherence gate."""

    ACCEPT = "accept"
    PREDICT_ONLY = "predict_only"
    REJECT = "reject"
    RECALIBRATE = "recalibrate"


@dataclass(frozen=True)
class CoherenceGateConfig:
    """Thresholds for mapping coherence, novelty, and drift to gate decisions."""

    accept_threshold: float = 0.85
    reject_threshold: float = 0.50
    novelty_accept_threshold: float = 0.35
    novelty_reject_threshold: float = 0.70
    novelty_recalibrate_threshold: float = 0.95
    drift_predict_threshold: float = 0.60
    drift_recalibrate_threshold: float = 0.90
    max_stale_frames: int = 200
    predict_only_noise: float = 3.0
    max_noise_multiplier: float = 6.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.reject_threshold <= self.accept_threshold <= 1.0:
            raise CoherenceGateError("thresholds must satisfy 0 <= reject <= accept <= 1")
        for name in (
            "novelty_accept_threshold",
            "novelty_reject_threshold",
            "novelty_recalibrate_threshold",
            "drift_predict_threshold",
            "drift_recalibrate_threshold",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or not math.isfinite(value):
                raise CoherenceGateError(f"{name} must be finite and non-negative")
        if not (
            self.novelty_accept_threshold
            <= self.novelty_reject_threshold
            <= self.novelty_recalibrate_threshold
        ):
            raise CoherenceGateError("novelty thresholds must be ordered accept <= reject <= recalibrate")
        if self.drift_predict_threshold > self.drift_recalibrate_threshold:
            raise CoherenceGateError("drift thresholds must be ordered predict <= recalibrate")
        if self.max_stale_frames <= 0:
            raise CoherenceGateError("max_stale_frames must be positive")
        if self.predict_only_noise < 1.0:
            raise CoherenceGateError("predict_only_noise must be at least 1")
        if self.max_noise_multiplier < self.predict_only_noise:
            raise CoherenceGateError("max_noise_multiplier must be at least predict_only_noise")


@dataclass(frozen=True)
class GateDecision:
    """Decision emitted by the coherence gate."""

    action: GateDecisionType
    coherence_score: float
    novelty_score: float
    drift_score: float
    noise_multiplier: float
    stale_frames: int
    reason: str

    @property
    def allows_update(self) -> bool:
        return self.action == GateDecisionType.ACCEPT

    @property
    def is_rejected(self) -> bool:
        return self.action in {GateDecisionType.REJECT, GateDecisionType.RECALIBRATE}

    @property
    def requires_recalibration(self) -> bool:
        return self.action == GateDecisionType.RECALIBRATE


@dataclass
class CoherenceGate:
    """Stateful gate policy with a consecutive low-coherence counter."""

    config: CoherenceGateConfig = CoherenceGateConfig()
    consecutive_low: int = 0
    last_decision: GateDecision | None = None

    def evaluate(
        self,
        coherence: float | CoherenceComponents,
        *,
        novelty_score: float = 0.0,
        deviation_score: float = 0.0,
        calibration_drift: float | Any = 0.0,
        stale_frames: int | None = None,
    ) -> GateDecision:
        """Evaluate one observation and update the low-coherence counter."""

        score = _coherence_value(coherence)
        novelty = max(_nonnegative_metric(novelty_score), _nonnegative_metric(deviation_score))
        drift = _drift_metric(calibration_drift)
        external_stale = int(stale_frames) if stale_frames is not None else self.consecutive_low
        if external_stale < 0:
            raise CoherenceGateError("stale_frames must be non-negative")

        cfg = self.config
        if drift >= cfg.drift_recalibrate_threshold:
            decision = _decision(
                GateDecisionType.RECALIBRATE,
                score,
                novelty,
                drift,
                cfg.max_noise_multiplier,
                max(external_stale, self.consecutive_low),
                "calibration drift threshold exceeded",
            )
        elif novelty >= cfg.novelty_recalibrate_threshold:
            decision = _decision(
                GateDecisionType.RECALIBRATE,
                score,
                novelty,
                drift,
                cfg.max_noise_multiplier,
                max(external_stale, self.consecutive_low),
                "novelty threshold requires recalibration",
            )
        elif max(external_stale, self.consecutive_low) >= cfg.max_stale_frames:
            decision = _decision(
                GateDecisionType.RECALIBRATE,
                score,
                novelty,
                drift,
                cfg.max_noise_multiplier,
                max(external_stale, self.consecutive_low),
                "stale frame budget exceeded",
            )
        elif (
            score >= cfg.accept_threshold
            and novelty <= cfg.novelty_accept_threshold
            and drift < cfg.drift_predict_threshold
        ):
            self.consecutive_low = 0
            decision = _decision(
                GateDecisionType.ACCEPT,
                score,
                novelty,
                drift,
                1.0,
                0,
                "coherence accepted",
            )
        elif score >= cfg.reject_threshold and novelty < cfg.novelty_reject_threshold:
            self.consecutive_low = max(self.consecutive_low, external_stale) + 1
            if self.consecutive_low >= cfg.max_stale_frames:
                decision = _decision(
                    GateDecisionType.RECALIBRATE,
                    score,
                    novelty,
                    drift,
                    cfg.max_noise_multiplier,
                    self.consecutive_low,
                    "stale frame budget exceeded",
                )
            else:
                decision = _decision(
                    GateDecisionType.PREDICT_ONLY,
                    score,
                    novelty,
                    drift,
                    max(
                        cfg.predict_only_noise,
                        adaptive_noise_multiplier(
                            score,
                            accept=cfg.accept_threshold,
                            reject=cfg.reject_threshold,
                            max_inflation=cfg.predict_only_noise,
                        ),
                    ),
                    self.consecutive_low,
                    "moderate coherence or elevated drift",
                )
        else:
            self.consecutive_low = max(self.consecutive_low, external_stale) + 1
            if self.consecutive_low >= cfg.max_stale_frames:
                decision = _decision(
                    GateDecisionType.RECALIBRATE,
                    score,
                    novelty,
                    drift,
                    cfg.max_noise_multiplier,
                    self.consecutive_low,
                    "stale frame budget exceeded",
                )
            else:
                decision = _decision(
                    GateDecisionType.REJECT,
                    score,
                    novelty,
                    drift,
                    cfg.max_noise_multiplier,
                    self.consecutive_low,
                    "low coherence or excessive novelty",
                )

        if decision.action == GateDecisionType.RECALIBRATE:
            self.consecutive_low = max(self.consecutive_low, decision.stale_frames)
        self.last_decision = decision
        return decision

    def reset(self) -> None:
        self.consecutive_low = 0
        self.last_decision = None


def decide_gate(
    coherence: float | CoherenceComponents,
    *,
    config: CoherenceGateConfig | None = None,
    novelty_score: float = 0.0,
    deviation_score: float = 0.0,
    calibration_drift: float | Any = 0.0,
    stale_frames: int = 0,
) -> GateDecision:
    """Stateless helper for a single gate decision."""

    gate = CoherenceGate(config or CoherenceGateConfig())
    return gate.evaluate(
        coherence,
        novelty_score=novelty_score,
        deviation_score=deviation_score,
        calibration_drift=calibration_drift,
        stale_frames=stale_frames,
    )


def adaptive_noise_multiplier(
    coherence: float,
    *,
    accept: float = 0.85,
    reject: float = 0.50,
    max_inflation: float = 3.0,
) -> float:
    """Scale noise from 1 at accept threshold to ``max_inflation`` at reject."""

    score = _coherence_value(coherence)
    if max_inflation < 1.0:
        raise CoherenceGateError("max_inflation must be at least 1")
    if score >= accept:
        return 1.0
    if score <= reject:
        return float(max_inflation)
    span = accept - reject
    if span <= 1e-12:
        return float(max_inflation)
    t = (accept - score) / span
    return float(1.0 + t * (max_inflation - 1.0))


def _decision(
    action: GateDecisionType,
    coherence_score: float,
    novelty_score: float,
    drift_score: float,
    noise_multiplier: float,
    stale_frames: int,
    reason: str,
) -> GateDecision:
    return GateDecision(
        action=action,
        coherence_score=float(coherence_score),
        novelty_score=float(novelty_score),
        drift_score=float(drift_score),
        noise_multiplier=float(noise_multiplier),
        stale_frames=int(stale_frames),
        reason=reason,
    )


def _coherence_value(coherence: float | CoherenceComponents) -> float:
    if isinstance(coherence, CoherenceComponents):
        value = coherence.score
    else:
        value = float(coherence)
    if not math.isfinite(value):
        raise CoherenceGateError("coherence must be finite")
    return float(np.clip(value, 0.0, 1.0))


def _nonnegative_metric(value: float) -> float:
    metric = float(value)
    if metric < 0.0 or not math.isfinite(metric):
        raise CoherenceGateError("novelty/deviation metrics must be finite and non-negative")
    return metric


def _drift_metric(value: float | Any) -> float:
    if hasattr(value, "drift_flagged") and bool(value.drift_flagged):
        return 1.0
    if hasattr(value, "recalibration_recommended") and bool(value.recalibration_recommended):
        return 1.0
    if hasattr(value, "amplitude_z_median"):
        amplitude = float(getattr(value, "amplitude_z_median"))
        phase = float(getattr(value, "phase_drift_median", 0.0))
        return max(amplitude / 6.0, phase / math.pi)
    metric = float(value)
    if metric < 0.0 or not math.isfinite(metric):
        raise CoherenceGateError("calibration_drift must be finite and non-negative")
    return metric


__all__ = [
    "CoherenceGate",
    "CoherenceGateConfig",
    "CoherenceGateError",
    "GateDecision",
    "GateDecisionType",
    "adaptive_noise_multiplier",
    "decide_gate",
]
