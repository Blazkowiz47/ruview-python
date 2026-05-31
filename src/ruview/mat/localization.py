"""MAT localization primitives for range-based survivor positioning.

The module mirrors the research behavior of the Rust MAT localization helpers:
RSSI/ToA distance conversion, 2D least-squares trilateration, depth estimation,
uncertainty-aware fusion, and UWB-style range constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
SPEED_OF_LIGHT_M_S = 299_792_458.0


class LocalizationError(ValueError):
    """Base error for MAT localization operations."""


class EstimateSource(str, Enum):
    """Source label used to weight fused position estimates."""

    RSSI_TRIANGULATION = "rssi_triangulation"
    TIME_OF_ARRIVAL = "time_of_arrival"
    CSI_FINGERPRINT = "csi_fingerprint"
    ANGLE_OF_ARRIVAL = "angle_of_arrival"
    DEPTH_ESTIMATION = "depth_estimation"
    RANGE_CONSTRAINT = "range_constraint"
    FUSED = "fused"


@dataclass(frozen=True)
class LocationUncertainty:
    """Horizontal/vertical position uncertainty in metres."""

    horizontal_error: float = 1.0
    vertical_error: float = 1.5
    confidence: float = 0.95
    gdop: float = 1.0

    def __post_init__(self) -> None:
        horizontal = _finite_float(self.horizontal_error, "horizontal_error")
        vertical = _finite_float(self.vertical_error, "vertical_error")
        confidence = _finite_float(self.confidence, "confidence")
        gdop = _finite_float(self.gdop, "gdop")
        if horizontal < 0.0 or vertical < 0.0:
            raise LocalizationError("uncertainty errors must be non-negative")
        if not 0.0 <= confidence <= 1.0:
            raise LocalizationError("confidence must be in [0, 1]")
        if gdop < 1.0:
            raise LocalizationError("gdop must be at least 1.0")
        object.__setattr__(self, "horizontal_error", horizontal)
        object.__setattr__(self, "vertical_error", vertical)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "gdop", gdop)


@dataclass(frozen=True)
class SensorPosition:
    """Surveyed MAT sensor position in metres."""

    id: str
    x: float
    y: float
    z: float = 0.0
    sensor_type: str = "transceiver"
    is_operational: bool = True

    def __post_init__(self) -> None:
        if not str(self.id):
            raise LocalizationError("sensor id must not be empty")
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "x", _finite_float(self.x, "x"))
        object.__setattr__(self, "y", _finite_float(self.y, "y"))
        object.__setattr__(self, "z", _finite_float(self.z, "z"))
        object.__setattr__(self, "sensor_type", str(self.sensor_type))
        object.__setattr__(self, "is_operational", bool(self.is_operational))

    @property
    def xy(self) -> tuple[float, float]:
        return self.x, self.y

    @property
    def xyz(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


@dataclass(frozen=True)
class DistanceEstimate:
    """Estimated range from one sensor to the target."""

    sensor_id: str
    distance_m: float
    confidence: float = 1.0
    uncertainty_m: float | None = None

    def __post_init__(self) -> None:
        distance = _finite_float(self.distance_m, "distance_m")
        confidence = _finite_float(self.confidence, "confidence")
        if distance < 0.0:
            raise LocalizationError("distance_m must be non-negative")
        if not 0.0 <= confidence <= 1.0:
            raise LocalizationError("confidence must be in [0, 1]")
        uncertainty = None
        if self.uncertainty_m is not None:
            uncertainty = _finite_float(self.uncertainty_m, "uncertainty_m")
            if uncertainty <= 0.0:
                raise LocalizationError("uncertainty_m must be positive")
        object.__setattr__(self, "sensor_id", str(self.sensor_id))
        object.__setattr__(self, "distance_m", distance)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "uncertainty_m", uncertainty)

    @property
    def distance(self) -> float:
        """Rust-style alias for ``distance_m``."""

        return self.distance_m


@dataclass(frozen=True)
class PositionEstimate:
    """A 3D position estimate with uncertainty and source metadata."""

    x: float
    y: float
    z: float = 0.0
    uncertainty: LocationUncertainty = field(default_factory=LocationUncertainty)
    source: EstimateSource | str = EstimateSource.FUSED
    weight: float = 1.0
    timestamp_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite_float(self.x, "x"))
        object.__setattr__(self, "y", _finite_float(self.y, "y"))
        object.__setattr__(self, "z", _finite_float(self.z, "z"))
        weight = _finite_float(self.weight, "weight")
        timestamp_s = _finite_float(self.timestamp_s, "timestamp_s")
        if weight < 0.0:
            raise LocalizationError("weight must be non-negative")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "timestamp_s", timestamp_s)
        if not isinstance(self.uncertainty, LocationUncertainty):
            raise LocalizationError("uncertainty must be a LocationUncertainty")

    @property
    def position(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    def as_array(self) -> FloatArray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def distance_to(self, other: Sequence[float] | "PositionEstimate") -> float:
        point = other.as_array() if isinstance(other, PositionEstimate) else _as_point3(other, "other")
        return float(np.linalg.norm(self.as_array() - point))


@dataclass(frozen=True)
class TriangulationConfig:
    """Configuration for RSSI/ToA trilateration."""

    min_sensors: int = 3
    max_uncertainty_m: float = 5.0
    path_loss_exponent: float = 3.0
    reference_distance_m: float = 1.0
    reference_rssi_dbm: float = -30.0
    weighted: bool = True

    def __post_init__(self) -> None:
        if self.min_sensors < 3:
            raise LocalizationError("min_sensors must be at least 3")
        if self.max_uncertainty_m <= 0.0 or not math.isfinite(float(self.max_uncertainty_m)):
            raise LocalizationError("max_uncertainty_m must be positive and finite")
        if self.path_loss_exponent <= 0.0 or not math.isfinite(float(self.path_loss_exponent)):
            raise LocalizationError("path_loss_exponent must be positive and finite")
        if self.reference_distance_m <= 0.0 or not math.isfinite(float(self.reference_distance_m)):
            raise LocalizationError("reference_distance_m must be positive and finite")
        if not math.isfinite(float(self.reference_rssi_dbm)):
            raise LocalizationError("reference_rssi_dbm must be finite")


def rssi_to_distance(
    rssi_dbm: float,
    *,
    reference_rssi_dbm: float = -30.0,
    reference_distance_m: float = 1.0,
    path_loss_exponent: float = 3.0,
) -> float:
    """Convert RSSI to distance with the log-distance path-loss model."""

    rssi = _finite_float(rssi_dbm, "rssi_dbm")
    reference_rssi = _finite_float(reference_rssi_dbm, "reference_rssi_dbm")
    reference_distance = _finite_float(reference_distance_m, "reference_distance_m")
    exponent = _finite_float(path_loss_exponent, "path_loss_exponent")
    if reference_distance <= 0.0:
        raise LocalizationError("reference_distance_m must be positive")
    if exponent <= 0.0:
        raise LocalizationError("path_loss_exponent must be positive")
    power = (reference_rssi - rssi) / (10.0 * exponent)
    return float(reference_distance * 10.0**power)


def toa_to_distance(
    toa_ns: float,
    *,
    round_trip: bool = True,
    speed_m_s: float = SPEED_OF_LIGHT_M_S,
) -> float:
    """Convert time-of-arrival nanoseconds to metres."""

    toa = _finite_float(toa_ns, "toa_ns")
    speed = _finite_float(speed_m_s, "speed_m_s")
    if toa < 0.0:
        raise LocalizationError("toa_ns must be non-negative")
    if speed <= 0.0:
        raise LocalizationError("speed_m_s must be positive")
    divisor = 2.0 if round_trip else 1.0
    return float(toa * 1e-9 * speed / divisor)


class Triangulator:
    """Range-based 2D least-squares triangulator."""

    def __init__(self, config: TriangulationConfig | None = None) -> None:
        self.config = config or TriangulationConfig()

    def estimate_position(
        self,
        sensors: Sequence[SensorPosition],
        rssi_values: Mapping[str, float] | Sequence[tuple[str, float]],
    ) -> PositionEstimate | None:
        """Estimate a 2D position from RSSI readings."""

        distances = [
            DistanceEstimate(
                sensor_id=sensor_id,
                distance_m=rssi_to_distance(
                    rssi,
                    reference_rssi_dbm=self.config.reference_rssi_dbm,
                    reference_distance_m=self.config.reference_distance_m,
                    path_loss_exponent=self.config.path_loss_exponent,
                ),
            )
            for sensor_id, rssi in _iter_measurements(rssi_values)
        ]
        return self.trilaterate(sensors, distances, source=EstimateSource.RSSI_TRIANGULATION)

    def estimate_from_toa(
        self,
        sensors: Sequence[SensorPosition],
        toa_values: Mapping[str, float] | Sequence[tuple[str, float]],
        *,
        round_trip: bool = True,
    ) -> PositionEstimate | None:
        """Estimate a 2D position from ToA readings in nanoseconds."""

        distances = [
            DistanceEstimate(
                sensor_id=sensor_id,
                distance_m=toa_to_distance(toa_ns, round_trip=round_trip),
            )
            for sensor_id, toa_ns in _iter_measurements(toa_values)
        ]
        return self.trilaterate(sensors, distances, source=EstimateSource.TIME_OF_ARRIVAL)

    def trilaterate(
        self,
        sensors: Sequence[SensorPosition],
        distances: Mapping[str, float] | Sequence[DistanceEstimate | tuple[str, float]],
        *,
        source: EstimateSource | str = EstimateSource.RANGE_CONSTRAINT,
        z: float = 0.0,
    ) -> PositionEstimate | None:
        """Solve a 2D trilateration problem with linearized least squares."""

        sensor_by_id = {sensor.id: sensor for sensor in sensors if sensor.is_operational}
        distance_estimates = _coerce_distances(distances)
        pairs: list[tuple[SensorPosition, DistanceEstimate]] = []
        seen: set[str] = set()
        for estimate in distance_estimates:
            sensor = sensor_by_id.get(estimate.sensor_id)
            if sensor is None or estimate.sensor_id in seen:
                continue
            seen.add(estimate.sensor_id)
            pairs.append((sensor, estimate))

        if len(pairs) < self.config.min_sensors:
            return None
        result = _solve_trilateration(pairs, weighted=self.config.weighted)
        if result is None:
            return None
        xy, residuals, rank = result
        if rank < 2 or not np.all(np.isfinite(xy)):
            return None

        uncertainty = _calculate_uncertainty(xy, pairs, residuals)
        if uncertainty.horizontal_error > self.config.max_uncertainty_m:
            return None
        return PositionEstimate(
            x=float(xy[0]),
            y=float(xy[1]),
            z=_finite_float(z, "z"),
            uncertainty=uncertainty,
            source=source,
        )


def trilaterate_2d(
    sensors: Sequence[SensorPosition],
    distances: Mapping[str, float] | Sequence[DistanceEstimate | tuple[str, float]],
    *,
    config: TriangulationConfig | None = None,
) -> PositionEstimate | None:
    """Convenience wrapper for ``Triangulator.trilaterate``."""

    return Triangulator(config).trilaterate(sensors, distances)


@dataclass(frozen=True)
class DepthEstimatorConfig:
    """Configuration for attenuation-based depth estimation."""

    max_depth_m: float = 10.0
    min_attenuation_db: float = 3.0
    attenuation_per_meter_db: float = 3.0
    free_space_loss_1m_db: float = 47.0

    def __post_init__(self) -> None:
        for name in (
            "max_depth_m",
            "min_attenuation_db",
            "attenuation_per_meter_db",
            "free_space_loss_1m_db",
        ):
            value = _finite_float(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if self.max_depth_m <= 0.0:
            raise LocalizationError("max_depth_m must be positive")
        if self.min_attenuation_db < 0.0:
            raise LocalizationError("min_attenuation_db must be non-negative")
        if self.attenuation_per_meter_db <= 0.0:
            raise LocalizationError("attenuation_per_meter_db must be positive")


@dataclass(frozen=True)
class DepthEstimate:
    """Estimated depth below the scan surface."""

    depth_m: float
    uncertainty_m: float
    confidence: float

    def __post_init__(self) -> None:
        depth = _finite_float(self.depth_m, "depth_m")
        uncertainty = _finite_float(self.uncertainty_m, "uncertainty_m")
        confidence = _finite_float(self.confidence, "confidence")
        if depth < 0.0:
            raise LocalizationError("depth_m must be non-negative")
        if uncertainty < 0.0:
            raise LocalizationError("uncertainty_m must be non-negative")
        if not 0.0 <= confidence <= 1.0:
            raise LocalizationError("confidence must be in [0, 1]")
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "uncertainty_m", uncertainty)
        object.__setattr__(self, "confidence", confidence)

    @property
    def depth(self) -> float:
        return self.depth_m


class DepthEstimator:
    """Simple attenuation-through-debris depth estimator."""

    def __init__(self, config: DepthEstimatorConfig | None = None) -> None:
        self.config = config or DepthEstimatorConfig()

    def estimate_depth(self, signal_attenuation_db: float, horizontal_distance_m: float = 0.0) -> DepthEstimate | None:
        attenuation = _finite_float(signal_attenuation_db, "signal_attenuation_db")
        horizontal = max(_finite_float(horizontal_distance_m, "horizontal_distance_m"), 0.0)
        if attenuation < self.config.min_attenuation_db:
            return DepthEstimate(depth_m=0.0, uncertainty_m=0.5, confidence=0.9)

        free_space = 0.0 if horizontal <= 0.0 else self.config.free_space_loss_1m_db + 20.0 * math.log10(horizontal)
        debris_attenuation = max(0.0, attenuation - free_space)
        depth = debris_attenuation / self.config.attenuation_per_meter_db
        if depth > self.config.max_depth_m:
            return None
        uncertainty = 0.3 + 0.15 * depth
        confidence = max(0.3, 1.0 - depth / self.config.max_depth_m)
        return DepthEstimate(depth_m=depth, uncertainty_m=uncertainty, confidence=confidence)

    def apply_depth(self, position: PositionEstimate, depth: DepthEstimate) -> PositionEstimate:
        """Return a position with negative z for below-surface depth."""

        confidence = math.sqrt(position.uncertainty.confidence * depth.confidence)
        uncertainty = LocationUncertainty(
            horizontal_error=position.uncertainty.horizontal_error,
            vertical_error=depth.uncertainty_m,
            confidence=confidence,
            gdop=position.uncertainty.gdop,
        )
        return PositionEstimate(
            x=position.x,
            y=position.y,
            z=-depth.depth_m,
            uncertainty=uncertainty,
            source=position.source,
            weight=position.weight,
            timestamp_s=position.timestamp_s,
        )


@dataclass(frozen=True)
class PositionFusionConfig:
    """Configuration for uncertainty-weighted position fusion."""

    min_variance_m2: float = 1e-6

    def __post_init__(self) -> None:
        if self.min_variance_m2 <= 0.0 or not math.isfinite(float(self.min_variance_m2)):
            raise LocalizationError("min_variance_m2 must be positive and finite")


class PositionFuser:
    """Fuse multiple position estimates with uncertainty/source weights."""

    SOURCE_WEIGHTS: dict[EstimateSource, float] = {
        EstimateSource.TIME_OF_ARRIVAL: 1.0,
        EstimateSource.ANGLE_OF_ARRIVAL: 0.9,
        EstimateSource.CSI_FINGERPRINT: 0.8,
        EstimateSource.RSSI_TRIANGULATION: 0.7,
        EstimateSource.RANGE_CONSTRAINT: 0.9,
        EstimateSource.DEPTH_ESTIMATION: 0.6,
        EstimateSource.FUSED: 1.0,
    }

    def __init__(self, config: PositionFusionConfig | None = None, *, max_history: int = 20) -> None:
        self.config = config or PositionFusionConfig()
        self.max_history = int(max_history)
        if self.max_history <= 0:
            raise LocalizationError("max_history must be positive")
        self.history: list[PositionEstimate] = []

    def fuse(self, estimates: Sequence[PositionEstimate]) -> PositionEstimate | None:
        if not estimates:
            return None
        if len(estimates) == 1:
            return estimates[0]

        weights = np.array([self.calculate_weight(estimate) for estimate in estimates], dtype=np.float64)
        if not np.any(weights > 0.0):
            return None
        points = np.vstack([estimate.as_array() for estimate in estimates])
        fused = np.average(points, axis=0, weights=weights)
        uncertainty = self._fused_uncertainty(estimates, weights)
        timestamp = max(estimate.timestamp_s for estimate in estimates)
        return PositionEstimate(
            x=float(fused[0]),
            y=float(fused[1]),
            z=float(fused[2]),
            uncertainty=uncertainty,
            source=EstimateSource.FUSED,
            weight=float(np.sum(weights)),
            timestamp_s=float(timestamp),
        )

    def fuse_with_history(self, current: PositionEstimate, *, alpha: float = 0.3) -> PositionEstimate:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        self.history.append(current)
        if len(self.history) > self.max_history:
            del self.history[: len(self.history) - self.max_history]
        smoothed = current.as_array()
        for index, estimate in enumerate(reversed(self.history[:-1]), start=1):
            weight = alpha * (1.0 - alpha) ** index
            smoothed = smoothed * (1.0 - weight) + estimate.as_array() * weight
        return PositionEstimate(
            x=float(smoothed[0]),
            y=float(smoothed[1]),
            z=float(smoothed[2]),
            uncertainty=current.uncertainty,
            source=EstimateSource.FUSED,
            weight=current.weight,
            timestamp_s=current.timestamp_s,
        )

    def calculate_weight(self, estimate: PositionEstimate) -> float:
        source = _coerce_source(estimate.source)
        source_weight = self.SOURCE_WEIGHTS.get(source, 0.7)
        horizontal_var = max(estimate.uncertainty.horizontal_error**2, self.config.min_variance_m2)
        vertical_var = max(estimate.uncertainty.vertical_error**2, self.config.min_variance_m2)
        variance = (2.0 * horizontal_var + vertical_var) / 3.0
        return float(source_weight * estimate.weight * estimate.uncertainty.confidence / variance)

    def clear_history(self) -> None:
        self.history.clear()

    def _fused_uncertainty(self, estimates: Sequence[PositionEstimate], weights: FloatArray) -> LocationUncertainty:
        total_weight = max(float(np.sum(weights)), self.config.min_variance_m2)
        horizontal = math.sqrt(1.0 / total_weight)
        vertical_weights = np.array(
            [
                estimate.weight
                * estimate.uncertainty.confidence
                / max(estimate.uncertainty.vertical_error**2, self.config.min_variance_m2)
                for estimate in estimates
            ],
            dtype=np.float64,
        )
        vertical_total = max(float(np.sum(vertical_weights)), self.config.min_variance_m2)
        vertical = math.sqrt(1.0 / vertical_total)
        confidence = min(0.99, float(np.average([e.uncertainty.confidence for e in estimates], weights=weights)) + 0.02)
        gdop = float(np.average([e.uncertainty.gdop for e in estimates], weights=weights))
        return LocationUncertainty(horizontal_error=horizontal, vertical_error=vertical, confidence=confidence, gdop=max(1.0, gdop))


def fuse_positions(estimates: Sequence[PositionEstimate]) -> PositionEstimate | None:
    """Convenience wrapper for ``PositionFuser.fuse``."""

    return PositionFuser().fuse(estimates)


@dataclass(frozen=True)
class RangeConstraint:
    """One range measurement from an anchor to a tag."""

    anchor_id: int | str
    anchor_pos: Sequence[float]
    measured_range_m: float
    uncertainty_m: float
    signal_quality: float = 1.0
    at_ns: int = 0

    def __post_init__(self) -> None:
        anchor_pos = _as_point3(self.anchor_pos, "anchor_pos")
        measured = _finite_float(self.measured_range_m, "measured_range_m")
        uncertainty = _finite_float(self.uncertainty_m, "uncertainty_m")
        quality = _finite_float(self.signal_quality, "signal_quality")
        if measured < 0.0:
            raise LocalizationError("measured_range_m must be non-negative")
        if uncertainty <= 0.0:
            raise LocalizationError("uncertainty_m must be positive")
        if not 0.0 <= quality <= 1.0:
            raise LocalizationError("signal_quality must be in [0, 1]")
        object.__setattr__(self, "anchor_pos", tuple(float(v) for v in anchor_pos))
        object.__setattr__(self, "measured_range_m", measured)
        object.__setattr__(self, "uncertainty_m", uncertainty)
        object.__setattr__(self, "signal_quality", quality)
        object.__setattr__(self, "at_ns", int(self.at_ns))

    def predicted_range(self, position: Sequence[float] | PositionEstimate) -> float:
        point = position.as_array() if isinstance(position, PositionEstimate) else _as_point3(position, "position")
        return float(np.linalg.norm(point - np.asarray(self.anchor_pos, dtype=np.float64)))

    def residual(self, position: Sequence[float] | PositionEstimate) -> float:
        return self.predicted_range(position) - self.measured_range_m

    def mahalanobis(self, position: Sequence[float] | PositionEstimate) -> float:
        return abs(self.residual(position)) / max(self.uncertainty_m, 1e-6)

    def is_consistent(self, position: Sequence[float] | PositionEstimate, gate_sigma: float = 3.0) -> bool:
        gate = _finite_float(gate_sigma, "gate_sigma")
        if gate < 0.0:
            raise LocalizationError("gate_sigma must be non-negative")
        return self.mahalanobis(position) <= gate


@dataclass(frozen=True)
class RangeValidationResult:
    """Range-constraint validation summary for a candidate position."""

    consistent: bool
    admitted_anchor_ids: tuple[int | str, ...]
    rejected_anchor_ids: tuple[int | str, ...]
    rms_residual_sigma: float
    max_residual_sigma: float


@dataclass(frozen=True)
class RangeRefineResult:
    """Output of constraint-aware range refinement."""

    position: tuple[float, float, float]
    rms_residual_sigma: float
    rejected_anchors: tuple[int | str, ...]
    iterations: int

    def as_estimate(self, uncertainty: LocationUncertainty | None = None) -> PositionEstimate:
        return PositionEstimate(
            *self.position,
            uncertainty=uncertainty or LocationUncertainty(horizontal_error=self.rms_residual_sigma),
            source=EstimateSource.RANGE_CONSTRAINT,
        )


@dataclass(frozen=True)
class RangeConstraintFusionConfig:
    """Gradient-descent configuration for range-constraint refinement."""

    gate_sigma: float = 3.0
    step: float = 1.0
    max_iters: int = 200
    tol_m: float = 1e-4

    def __post_init__(self) -> None:
        if self.gate_sigma < 0.0 or not math.isfinite(float(self.gate_sigma)):
            raise LocalizationError("gate_sigma must be non-negative and finite")
        if self.step <= 0.0 or not math.isfinite(float(self.step)):
            raise LocalizationError("step must be positive and finite")
        if self.max_iters < 0:
            raise LocalizationError("max_iters must be non-negative")
        if self.tol_m <= 0.0 or not math.isfinite(float(self.tol_m)):
            raise LocalizationError("tol_m must be positive and finite")


class RangeConstraintFuser:
    """Refine a position against admitted range constraints."""

    def __init__(self, config: RangeConstraintFusionConfig | None = None) -> None:
        self.config = config or RangeConstraintFusionConfig()

    def refine(
        self,
        prior: Sequence[float] | PositionEstimate,
        constraints: Sequence[RangeConstraint],
    ) -> RangeRefineResult:
        point = prior.as_array() if isinstance(prior, PositionEstimate) else _as_point3(prior, "prior")
        admitted: list[RangeConstraint] = []
        rejected: list[int | str] = []
        for constraint in constraints:
            if constraint.is_consistent(point, self.config.gate_sigma):
                admitted.append(constraint)
            else:
                rejected.append(constraint.anchor_id)

        iterations = 0
        if admitted:
            for iterations in range(1, self.config.max_iters + 1):
                grad = np.zeros(3, dtype=np.float64)
                sum_weight = 0.0
                for constraint in admitted:
                    anchor = np.asarray(constraint.anchor_pos, dtype=np.float64)
                    delta = point - anchor
                    distance = max(float(np.linalg.norm(delta)), 1e-9)
                    weight = 1.0 / max(constraint.uncertainty_m, 1e-6) ** 2
                    sum_weight += weight
                    coeff = 2.0 * weight * (distance - constraint.measured_range_m) / distance
                    grad += coeff * delta
                scale = self.config.step / (2.0 * max(sum_weight, 1e-12))
                update = scale * grad
                point = point - update
                if float(np.linalg.norm(update)) < self.config.tol_m:
                    break

        if admitted:
            residuals = np.array([constraint.mahalanobis(point) for constraint in admitted], dtype=np.float64)
            rms = float(np.sqrt(np.mean(residuals * residuals)))
        else:
            rms = math.inf
        return RangeRefineResult(
            position=(float(point[0]), float(point[1]), float(point[2])),
            rms_residual_sigma=rms,
            rejected_anchors=tuple(rejected),
            iterations=iterations,
        )

    def associate(self, tracks: Sequence[Sequence[float] | PositionEstimate], constraint: RangeConstraint) -> int | None:
        best_index: int | None = None
        best_mahalanobis = math.inf
        for index, track in enumerate(tracks):
            distance = constraint.mahalanobis(track)
            if distance <= self.config.gate_sigma and distance < best_mahalanobis:
                best_index = index
                best_mahalanobis = distance
        return best_index


def validate_range_constraints(
    position: Sequence[float] | PositionEstimate,
    constraints: Sequence[RangeConstraint],
    *,
    gate_sigma: float = 3.0,
    min_signal_quality: float = 0.0,
) -> RangeValidationResult:
    """Validate a candidate position against range constraints."""

    if not constraints:
        return RangeValidationResult(False, (), (), math.inf, math.inf)
    quality = _finite_float(min_signal_quality, "min_signal_quality")
    if not 0.0 <= quality <= 1.0:
        raise LocalizationError("min_signal_quality must be in [0, 1]")
    admitted: list[int | str] = []
    rejected: list[int | str] = []
    residuals: list[float] = []
    for constraint in constraints:
        residual = constraint.mahalanobis(position)
        residuals.append(residual)
        if constraint.signal_quality >= quality and residual <= gate_sigma:
            admitted.append(constraint.anchor_id)
        else:
            rejected.append(constraint.anchor_id)
    values = np.asarray(residuals, dtype=np.float64)
    rms = float(np.sqrt(np.mean(values * values))) if values.size else math.inf
    max_residual = float(np.max(values)) if values.size else math.inf
    return RangeValidationResult(
        consistent=not rejected,
        admitted_anchor_ids=tuple(admitted),
        rejected_anchor_ids=tuple(rejected),
        rms_residual_sigma=rms,
        max_residual_sigma=max_residual,
    )


def _solve_trilateration(
    pairs: Sequence[tuple[SensorPosition, DistanceEstimate]],
    *,
    weighted: bool,
) -> tuple[FloatArray, FloatArray, int] | None:
    ref_sensor, ref_distance = pairs[0]
    x1, y1 = ref_sensor.x, ref_sensor.y
    r1 = ref_distance.distance_m

    rows: list[list[float]] = []
    rhs: list[float] = []
    weights: list[float] = []
    for sensor, distance in pairs[1:]:
        rows.append([2.0 * (sensor.x - x1), 2.0 * (sensor.y - y1)])
        rhs.append(
            r1 * r1
            - distance.distance_m * distance.distance_m
            - x1 * x1
            + sensor.x * sensor.x
            - y1 * y1
            + sensor.y * sensor.y
        )
        weights.append(_distance_weight(ref_distance, distance) if weighted else 1.0)

    a = np.asarray(rows, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    if a.shape[0] < 2 or a.shape[1] != 2:
        return None
    if weighted:
        w = np.sqrt(np.asarray(weights, dtype=np.float64))
        a = a * w[:, None]
        b = b * w
    try:
        solution, _residual_sum, rank, _singular_values = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    residuals = _range_residuals(solution, pairs)
    return np.asarray(solution, dtype=np.float64), residuals, int(rank)


def _distance_weight(reference: DistanceEstimate, estimate: DistanceEstimate) -> float:
    uncertainty = 0.0
    for item in (reference, estimate):
        if item.uncertainty_m is not None:
            uncertainty += item.uncertainty_m**2
        else:
            uncertainty += max(0.5, 0.10 * item.distance_m) ** 2
    confidence = max(reference.confidence * estimate.confidence, 1e-6)
    return float(confidence / max(uncertainty, 1e-6))


def _range_residuals(position_xy: FloatArray, pairs: Sequence[tuple[SensorPosition, DistanceEstimate]]) -> FloatArray:
    residuals = []
    for sensor, distance in pairs:
        predicted = math.hypot(float(position_xy[0]) - sensor.x, float(position_xy[1]) - sensor.y)
        residuals.append(distance.distance_m - predicted)
    return np.asarray(residuals, dtype=np.float64)


def _calculate_uncertainty(
    position_xy: FloatArray,
    pairs: Sequence[tuple[SensorPosition, DistanceEstimate]],
    residuals: FloatArray,
) -> LocationUncertainty:
    rmse = float(np.sqrt(np.mean(residuals * residuals))) if residuals.size else math.inf
    gdop = estimate_gdop(position_xy, [sensor for sensor, _distance in pairs])
    horizontal = rmse * gdop
    confidence = float(np.clip(0.95 / (1.0 + horizontal), 0.05, 0.95))
    if horizontal < 1e-12:
        confidence = 0.95
    return LocationUncertainty(
        horizontal_error=horizontal,
        vertical_error=horizontal * 1.5,
        confidence=confidence,
        gdop=gdop,
    )


def estimate_gdop(position_xy: Sequence[float], sensors: Sequence[SensorPosition]) -> float:
    """Estimate a simple 2D geometric dilution of precision factor."""

    position = np.asarray(position_xy, dtype=np.float64)
    if position.shape[0] < 2 or not np.all(np.isfinite(position[:2])):
        raise LocalizationError("position_xy must contain at least two finite values")
    n = len(sensors)
    if n < 2:
        return 10.0

    angle_sum = 0.0
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            v1 = np.array([sensors[i].x - position[0], sensors[i].y - position[1]], dtype=np.float64)
            v2 = np.array([sensors[j].x - position[0], sensors[j].y - position[1]], dtype=np.float64)
            norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
            if norm <= 0.0:
                continue
            angle_sum += math.acos(float(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)))
            pair_count += 1
    if pair_count == 0:
        return 10.0
    avg_angle = angle_sum / pair_count
    optimal = math.pi / 2.0
    return max(1.0, abs(avg_angle / optimal - 1.0) + 1.0)


def _coerce_distances(
    distances: Mapping[str, float] | Sequence[DistanceEstimate | tuple[str, float]],
) -> list[DistanceEstimate]:
    if isinstance(distances, Mapping):
        return [DistanceEstimate(str(sensor_id), float(distance)) for sensor_id, distance in distances.items()]
    result: list[DistanceEstimate] = []
    for item in distances:
        if isinstance(item, DistanceEstimate):
            result.append(item)
        else:
            if len(item) != 2:
                raise LocalizationError("distance tuples must contain (sensor_id, distance_m)")
            sensor_id, distance = item
            result.append(DistanceEstimate(str(sensor_id), float(distance)))
    return result


def _iter_measurements(values: Mapping[str, float] | Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
    if isinstance(values, Mapping):
        iterable = values.items()
    else:
        iterable = values
    return [(str(sensor_id), _finite_float(value, "measurement")) for sensor_id, value in iterable]


def _coerce_source(source: EstimateSource | str) -> EstimateSource:
    if isinstance(source, EstimateSource):
        return source
    try:
        return EstimateSource(str(source))
    except ValueError:
        return EstimateSource.FUSED


def _as_point3(value: Sequence[float] | ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise LocalizationError(f"{name} must be a 3D point")
    if not np.all(np.isfinite(arr)):
        raise LocalizationError(f"{name} contains non-finite values")
    return np.array(arr, dtype=np.float64, copy=True)


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise LocalizationError(f"{name} must be finite")
    return result


__all__ = [
    "DepthEstimate",
    "DepthEstimator",
    "DepthEstimatorConfig",
    "DistanceEstimate",
    "EstimateSource",
    "LocationUncertainty",
    "LocalizationError",
    "PositionEstimate",
    "PositionFuser",
    "PositionFusionConfig",
    "RangeConstraint",
    "RangeConstraintFuser",
    "RangeConstraintFusionConfig",
    "RangeRefineResult",
    "RangeValidationResult",
    "SPEED_OF_LIGHT_M_S",
    "SensorPosition",
    "TriangulationConfig",
    "Triangulator",
    "estimate_gdop",
    "fuse_positions",
    "rssi_to_distance",
    "toa_to_distance",
    "trilaterate_2d",
    "validate_range_constraints",
]
