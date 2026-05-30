"""Confidence scores used by sensing and pose contracts."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

from ruview.core.errors import ValidationError

DEFAULT_CONFIDENCE_THRESHOLD = 0.5


@dataclass(frozen=True, order=True)
class Confidence:
    """Confidence score constrained to the inclusive range ``[0.0, 1.0]``."""

    score: float

    MIN: ClassVar["Confidence"]
    MAX: ClassVar["Confidence"]

    def __post_init__(self) -> None:
        value = float(self.score)
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValidationError(f"Confidence must be in [0.0, 1.0], got {self.score!r}")
        object.__setattr__(self, "score", value)

    @property
    def value(self) -> float:
        """Return the raw floating-point value."""

        return self.score

    def is_high(self) -> bool:
        """Return ``True`` when this score meets the default visibility threshold."""

        return self.score >= DEFAULT_CONFIDENCE_THRESHOLD

    def exceeds(self, threshold: float) -> bool:
        """Return ``True`` when this score meets ``threshold``."""

        return self.score >= threshold


Confidence.MIN = Confidence(0.0)
Confidence.MAX = Confidence(1.0)

