"""Identity-risk score and gate-action mapping for BFLD privacy."""

from __future__ import annotations

import math
from enum import Enum

PREDICT_ONLY_THRESHOLD = 0.5
REJECT_THRESHOLD = 0.7
RECALIBRATE_THRESHOLD = 0.9


def identity_risk_score(sep: float, stab: float, consist: float, conf: float) -> float:
    """Compute the bounded multiplicative identity-risk score."""

    score = _clamp01(sep) * _clamp01(stab) * _clamp01(consist) * _clamp01(conf)
    return _clamp01(score)


score = identity_risk_score


class GateAction(str, Enum):
    """Coherence-gate decision derived from an identity-risk score."""

    Accept = "accept"
    PredictOnly = "predict_only"
    Reject = "reject"
    Recalibrate = "recalibrate"

    @classmethod
    def from_score(cls, risk_score: float) -> "GateAction":
        value = float(risk_score)
        if math.isnan(value):
            return cls.Accept
        if value < PREDICT_ONLY_THRESHOLD:
            return cls.Accept
        if value < REJECT_THRESHOLD:
            return cls.PredictOnly
        if value < RECALIBRATE_THRESHOLD:
            return cls.Reject
        return cls.Recalibrate

    def allows_publish(self) -> bool:
        return self in {GateAction.Accept, GateAction.PredictOnly}

    def drops_event(self) -> bool:
        return self in {GateAction.Reject, GateAction.Recalibrate}

    def requires_recalibrate(self) -> bool:
        return self is GateAction.Recalibrate


def _clamp01(value: float) -> float:
    v = float(value)
    if math.isnan(v):
        return 0.0
    return min(1.0, max(0.0, v))
