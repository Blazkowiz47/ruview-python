"""CSI vital-frame contracts and EMA static suppression."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray


ESP32_VITAL_SUBCARRIERS = 56


@dataclass(frozen=True)
class CsiVitalFrame:
    """Frame-like input for the lightweight vital-sign pipeline."""

    amplitudes: ArrayLike
    phases: ArrayLike | None = None
    sample_index: int = 0
    sample_rate_hz: float = 100.0

    def __post_init__(self) -> None:
        amplitudes = np.asarray(self.amplitudes, dtype=np.float64).reshape(-1)
        phases = (
            np.zeros_like(amplitudes)
            if self.phases is None
            else np.asarray(self.phases, dtype=np.float64).reshape(-1)
        )
        if phases.size != amplitudes.size:
            raise ValueError(
                "CsiVitalFrame amplitudes and phases must have matching lengths, "
                f"got {amplitudes.size} and {phases.size}"
            )
        if self.sample_rate_hz <= 0.0 or not isfinite(float(self.sample_rate_hz)):
            raise ValueError("sample_rate_hz must be positive and finite")
        object.__setattr__(self, "amplitudes", amplitudes)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "sample_index", int(self.sample_index))
        object.__setattr__(self, "sample_rate_hz", float(self.sample_rate_hz))

    @property
    def n_subcarriers(self) -> int:
        return int(self.amplitudes.size)


@dataclass
class CsiVitalPreprocessor:
    """EMA baseline remover that returns per-subcarrier residual amplitudes."""

    n_subcarriers: int = ESP32_VITAL_SUBCARRIERS
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.n_subcarriers <= 0:
            raise ValueError("n_subcarriers must be positive")
        self.alpha = _clamp_alpha(self.alpha)
        self._predictions = np.zeros(int(self.n_subcarriers), dtype=np.float64)
        self._initialized = np.zeros(int(self.n_subcarriers), dtype=bool)

    @classmethod
    def esp32_default(cls) -> "CsiVitalPreprocessor":
        """Return the ESP32 default preprocessor: 56 subcarriers, alpha 0.05."""

        return cls(n_subcarriers=ESP32_VITAL_SUBCARRIERS, alpha=0.05)

    def process(self, frame: CsiVitalFrame | ArrayLike) -> NDArray[np.float64] | None:
        """Process one frame and return amplitude residuals, or ``None`` when empty."""

        amplitudes = _amplitudes_from_frame(frame)
        n = min(amplitudes.size, int(self.n_subcarriers))
        if n == 0:
            return None

        observed = amplitudes[:n]
        residuals = np.zeros(n, dtype=np.float64)
        ready = self._initialized[:n]
        residuals[ready] = observed[ready] - self._predictions[:n][ready]

        seeded = ~ready
        if np.any(seeded):
            self._predictions[:n][seeded] = observed[seeded]
            self._initialized[:n][seeded] = True

        if np.any(ready):
            self._predictions[:n][ready] = (
                self.alpha * observed[ready] + (1.0 - self.alpha) * self._predictions[:n][ready]
            )

        return residuals

    def reset(self) -> None:
        """Clear EMA predictions and initialization state."""

        self._predictions.fill(0.0)
        self._initialized.fill(False)

    def set_alpha(self, alpha: float) -> None:
        """Update the EMA alpha after clamping to the Rust reference range."""

        self.alpha = _clamp_alpha(alpha)

    @property
    def predictions(self) -> NDArray[np.float64]:
        """Return a copy of current EMA predictions for diagnostics."""

        return self._predictions.copy()


def _amplitudes_from_frame(frame: CsiVitalFrame | ArrayLike) -> NDArray[np.float64]:
    if isinstance(frame, CsiVitalFrame):
        return np.asarray(frame.amplitudes, dtype=np.float64).reshape(-1)
    return np.asarray(frame, dtype=np.float64).reshape(-1)


def _clamp_alpha(alpha: float) -> float:
    if not isfinite(float(alpha)):
        return 0.05
    return float(np.clip(alpha, 0.001, 0.999))
