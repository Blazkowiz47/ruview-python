"""Lightweight 17-keypoint constant-velocity pose tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
NUM_KEYPOINTS = 17


class PoseTrackerError(ValueError):
    """Base error for pose tracker operations."""


class TrackStatus(str, Enum):
    """Lifecycle state for a tracked skeleton."""

    TENTATIVE = "tentative"
    ACTIVE = "active"
    LOST = "lost"
    TERMINATED = "terminated"

    @property
    def is_alive(self) -> bool:
        return self in {TrackStatus.TENTATIVE, TrackStatus.ACTIVE, TrackStatus.LOST}

    @property
    def is_visible(self) -> bool:
        return self in {TrackStatus.TENTATIVE, TrackStatus.ACTIVE}


@dataclass(frozen=True)
class PoseTrackerConfig:
    """Parameters for the constant-velocity keypoint filter."""

    default_dt: float = 1.0
    process_noise: float = 0.3
    measurement_noise: float = 0.08
    initial_position_std: float = 0.10
    initial_velocity_std: float = 0.50
    velocity_smoothing: float = 0.45
    confidence_smoothing: float = 0.55
    prediction_confidence_decay: float = 0.85
    min_keypoint_confidence: float = 0.25
    assignment_max_distance: float = 1.25
    birth_hits: int = 2
    max_misses: int = 5
    terminate_misses: int = 100

    def __post_init__(self) -> None:
        if self.default_dt <= 0.0:
            raise PoseTrackerError("default_dt must be positive")
        if self.process_noise < 0.0:
            raise PoseTrackerError("process_noise must be non-negative")
        if self.measurement_noise <= 0.0:
            raise PoseTrackerError("measurement_noise must be positive")
        if self.initial_position_std <= 0.0 or self.initial_velocity_std <= 0.0:
            raise PoseTrackerError("initial standard deviations must be positive")
        if not 0.0 <= self.velocity_smoothing <= 1.0:
            raise PoseTrackerError("velocity_smoothing must be in [0, 1]")
        if not 0.0 <= self.confidence_smoothing <= 1.0:
            raise PoseTrackerError("confidence_smoothing must be in [0, 1]")
        if not 0.0 <= self.prediction_confidence_decay <= 1.0:
            raise PoseTrackerError("prediction_confidence_decay must be in [0, 1]")
        if not 0.0 <= self.min_keypoint_confidence <= 1.0:
            raise PoseTrackerError("min_keypoint_confidence must be in [0, 1]")
        if self.assignment_max_distance <= 0.0:
            raise PoseTrackerError("assignment_max_distance must be positive")
        if self.birth_hits <= 0:
            raise PoseTrackerError("birth_hits must be positive")
        if self.max_misses < 0 or self.terminate_misses < self.max_misses:
            raise PoseTrackerError("miss thresholds must be ordered and non-negative")


@dataclass(frozen=True)
class PoseDetection:
    """One model detection with 17 3D keypoints and confidence scores."""

    positions: FloatArray
    confidences: FloatArray | None = None

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float64)
        confidences = self.confidences

        if positions.shape == (NUM_KEYPOINTS, 4) and confidences is None:
            confidences = positions[:, 3]
            positions = positions[:, :3]

        if positions.shape != (NUM_KEYPOINTS, 3):
            raise PoseTrackerError(
                f"positions must have shape ({NUM_KEYPOINTS}, 3) or ({NUM_KEYPOINTS}, 4)"
            )

        if confidences is None:
            conf = np.ones(NUM_KEYPOINTS, dtype=np.float64)
        else:
            conf = np.asarray(confidences, dtype=np.float64)
            if conf.shape != (NUM_KEYPOINTS,):
                raise PoseTrackerError(f"confidences must have shape ({NUM_KEYPOINTS},)")

        object.__setattr__(self, "positions", positions.astype(np.float64, copy=True))
        object.__setattr__(self, "confidences", np.clip(conf, 0.0, 1.0).astype(np.float64, copy=True))

    @classmethod
    def from_array(cls, keypoints: ArrayLike) -> "PoseDetection":
        return cls(np.asarray(keypoints, dtype=np.float64))

    def valid_mask(self, min_confidence: float = 0.25) -> NDArray[np.bool_]:
        finite_positions = np.all(np.isfinite(self.positions), axis=1)
        finite_confidences = np.isfinite(self.confidences)
        return finite_positions & finite_confidences & (self.confidences >= min_confidence)

    def centroid(self, min_confidence: float = 0.25) -> FloatArray:
        valid = self.valid_mask(min_confidence)
        if not np.any(valid):
            return np.zeros(3, dtype=np.float64)
        weights = self.confidences[valid]
        return np.average(self.positions[valid], axis=0, weights=np.maximum(weights, 1e-12))


@dataclass
class PoseTrack:
    """Persistent 17-keypoint track with diagonal Kalman-style state."""

    track_id: int
    positions: FloatArray
    velocities: FloatArray
    confidences: FloatArray
    position_variance: FloatArray
    velocity_variance: FloatArray
    status: TrackStatus = TrackStatus.TENTATIVE
    age: int = 1
    hits: int = 1
    misses: int = 0
    last_timestamp: float | None = None

    @classmethod
    def from_detection(
        cls,
        track_id: int,
        detection: PoseDetection,
        config: PoseTrackerConfig,
        *,
        timestamp: float | None = None,
    ) -> "PoseTrack":
        valid = detection.valid_mask(config.min_keypoint_confidence)
        positions = detection.positions.copy()
        if np.any(valid):
            fill = detection.centroid(config.min_keypoint_confidence)
        else:
            fill = np.zeros(3, dtype=np.float64)
        positions[~valid] = fill

        confidences = np.where(valid, detection.confidences, 0.0).astype(np.float64)
        status = TrackStatus.ACTIVE if config.birth_hits <= 1 else TrackStatus.TENTATIVE
        return cls(
            track_id=int(track_id),
            positions=positions,
            velocities=np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64),
            confidences=confidences,
            position_variance=np.full((NUM_KEYPOINTS, 3), config.initial_position_std**2),
            velocity_variance=np.full((NUM_KEYPOINTS, 3), config.initial_velocity_std**2),
            status=status,
            last_timestamp=timestamp,
        )

    def predict(self, dt: float, config: PoseTrackerConfig) -> None:
        dt = max(float(dt), 1e-9)
        self.positions = self.positions + self.velocities * dt
        q = config.process_noise**2
        self.position_variance = self.position_variance + self.velocity_variance * dt**2 + q * dt**4 / 4.0
        self.velocity_variance = self.velocity_variance + q * dt**2
        self.confidences = self.confidences * config.prediction_confidence_decay
        self.age += 1
        self.misses += 1

    def update(self, detection: PoseDetection, config: PoseTrackerConfig, *, dt: float) -> bool:
        valid = detection.valid_mask(config.min_keypoint_confidence)
        if not np.any(valid):
            return False

        dt = max(float(dt), 1e-9)
        measurement = detection.positions[valid]
        confidence = detection.confidences[valid]
        predicted = self.positions[valid].copy()
        innovation = measurement - predicted

        r = (config.measurement_noise / np.maximum(confidence[:, None], 1e-3)) ** 2
        gain = self.position_variance[valid] / (self.position_variance[valid] + r)
        self.positions[valid] = predicted + gain * innovation
        self.position_variance[valid] = (1.0 - gain) * self.position_variance[valid]

        observed_velocity = innovation / dt
        velocity_gain = np.clip(config.velocity_smoothing * np.mean(gain, axis=1, keepdims=True), 0.0, 1.0)
        self.velocities[valid] = (1.0 - velocity_gain) * self.velocities[valid] + velocity_gain * observed_velocity

        alpha = config.confidence_smoothing
        self.confidences[valid] = (1.0 - alpha) * self.confidences[valid] + alpha * confidence
        self.misses = 0
        self.hits += 1
        if self.status in {TrackStatus.TENTATIVE, TrackStatus.LOST} and self.hits >= config.birth_hits:
            self.status = TrackStatus.ACTIVE
        return True

    def centroid(self, min_confidence: float = 0.1) -> FloatArray:
        valid = self.confidences >= min_confidence
        if np.any(valid):
            return np.average(self.positions[valid], axis=0, weights=np.maximum(self.confidences[valid], 1e-12))
        return np.mean(self.positions, axis=0)

    def as_array(self, *, include_confidence: bool = True) -> FloatArray:
        if not include_confidence:
            return self.positions.copy()
        return np.column_stack([self.positions, self.confidences])

    def mark_after_miss(self, config: PoseTrackerConfig) -> None:
        if self.status == TrackStatus.TERMINATED:
            return
        if self.misses >= config.terminate_misses:
            self.status = TrackStatus.TERMINATED
        elif self.misses >= config.max_misses:
            self.status = TrackStatus.LOST


class PoseTracker:
    """Greedy multi-person tracker for 17-keypoint pose detections."""

    def __init__(self, config: PoseTrackerConfig | None = None) -> None:
        self.config = config or PoseTrackerConfig()
        self.tracks: list[PoseTrack] = []
        self.next_track_id = 0
        self.last_timestamp: float | None = None

    def update(
        self,
        detections: Sequence[PoseDetection | ArrayLike],
        *,
        timestamp: float | None = None,
        dt: float | None = None,
    ) -> list[PoseTrack]:
        step_dt = self._resolve_dt(timestamp, dt)
        normalized = [_coerce_detection(detection) for detection in detections]

        for track in self.tracks:
            if track.status.is_alive:
                track.predict(step_dt, self.config)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for track_idx, detection_idx in self._greedy_assign(normalized):
            track = self.tracks[track_idx]
            detection = normalized[detection_idx]
            if track.update(detection, self.config, dt=step_dt):
                track.last_timestamp = timestamp
                matched_tracks.add(track_idx)
                matched_detections.add(detection_idx)

        for detection_idx, detection in enumerate(normalized):
            if detection_idx not in matched_detections:
                self.tracks.append(
                    PoseTrack.from_detection(
                        self.next_track_id,
                        detection,
                        self.config,
                        timestamp=timestamp,
                    )
                )
                self.next_track_id += 1

        for track_idx, track in enumerate(self.tracks):
            if track_idx not in matched_tracks:
                track.mark_after_miss(self.config)

        self.last_timestamp = timestamp
        return self.active_tracks()

    def predict(self, *, dt: float | None = None) -> list[PoseTrack]:
        step_dt = self.config.default_dt if dt is None else float(dt)
        for track in self.tracks:
            if track.status.is_alive:
                track.predict(step_dt, self.config)
                track.mark_after_miss(self.config)
        return self.active_tracks()

    def active_tracks(self) -> list[PoseTrack]:
        return [track for track in self.tracks if track.status.is_alive]

    def visible_tracks(self) -> list[PoseTrack]:
        return [track for track in self.tracks if track.status.is_visible]

    def prune_terminated(self) -> None:
        self.tracks = [track for track in self.tracks if track.status != TrackStatus.TERMINATED]

    def find_track(self, track_id: int) -> PoseTrack | None:
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    def _resolve_dt(self, timestamp: float | None, dt: float | None) -> float:
        if dt is not None:
            if dt <= 0.0:
                raise PoseTrackerError("dt must be positive")
            return float(dt)
        if timestamp is not None and self.last_timestamp is not None:
            elapsed = float(timestamp) - float(self.last_timestamp)
            if elapsed > 0.0:
                return elapsed
        return self.config.default_dt

    def _greedy_assign(self, detections: Sequence[PoseDetection]) -> list[tuple[int, int]]:
        candidates: list[tuple[float, int, int]] = []
        for track_idx, track in enumerate(self.tracks):
            if not track.status.is_alive:
                continue
            for detection_idx, detection in enumerate(detections):
                cost = _assignment_cost(track, detection, self.config)
                if cost <= self.config.assignment_max_distance:
                    candidates.append((cost, track_idx, detection_idx))

        assignments: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _cost, track_idx, detection_idx in sorted(candidates, key=lambda item: item[0]):
            if track_idx in used_tracks or detection_idx in used_detections:
                continue
            assignments.append((track_idx, detection_idx))
            used_tracks.add(track_idx)
            used_detections.add(detection_idx)
        return assignments


def _coerce_detection(detection: PoseDetection | ArrayLike) -> PoseDetection:
    if isinstance(detection, PoseDetection):
        return detection
    return PoseDetection.from_array(detection)


def _assignment_cost(track: PoseTrack, detection: PoseDetection, config: PoseTrackerConfig) -> float:
    if not np.any(detection.valid_mask(config.min_keypoint_confidence)):
        return float("inf")
    delta = track.centroid(config.min_keypoint_confidence * 0.5) - detection.centroid(config.min_keypoint_confidence)
    return float(np.linalg.norm(delta))


__all__ = [
    "NUM_KEYPOINTS",
    "PoseDetection",
    "PoseTrack",
    "PoseTracker",
    "PoseTrackerConfig",
    "PoseTrackerError",
    "TrackStatus",
]
