"""MAT constant-velocity tracking primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Hashable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class TrackingError(ValueError):
    """Base error for MAT tracking operations."""


class TrackState(str, Enum):
    """Lifecycle state for a tracked survivor."""

    TENTATIVE = "tentative"
    ACTIVE = "active"
    LOST = "lost"
    TERMINATED = "terminated"
    RESCUED = "rescued"

    @property
    def is_alive(self) -> bool:
        return self in {TrackState.TENTATIVE, TrackState.ACTIVE, TrackState.LOST}

    @property
    def is_visible(self) -> bool:
        return self in {TrackState.TENTATIVE, TrackState.ACTIVE}

    @property
    def is_terminal(self) -> bool:
        return self in {TrackState.TERMINATED, TrackState.RESCUED}


@dataclass(frozen=True)
class TrackerConfig:
    """Configuration for nearest-neighbour survivor tracking."""

    birth_hits_required: int = 2
    max_active_misses: int = 3
    max_lost_age_s: float = 30.0
    reid_threshold: float = 0.35
    gate_mahalanobis_sq: float = 9.0
    obs_noise_var: float = 2.25
    process_noise_var: float = 0.01
    drop_terminal: bool = True

    def __post_init__(self) -> None:
        if self.birth_hits_required <= 0:
            raise TrackingError("birth_hits_required must be positive")
        if self.max_active_misses < 0:
            raise TrackingError("max_active_misses must be non-negative")
        if self.max_lost_age_s < 0.0 or not math.isfinite(float(self.max_lost_age_s)):
            raise TrackingError("max_lost_age_s must be non-negative and finite")
        if self.reid_threshold < 0.0 or not math.isfinite(float(self.reid_threshold)):
            raise TrackingError("reid_threshold must be non-negative and finite")
        if self.gate_mahalanobis_sq <= 0.0 or not math.isfinite(float(self.gate_mahalanobis_sq)):
            raise TrackingError("gate_mahalanobis_sq must be positive and finite")
        if self.obs_noise_var <= 0.0 or not math.isfinite(float(self.obs_noise_var)):
            raise TrackingError("obs_noise_var must be positive and finite")
        if self.process_noise_var < 0.0 or not math.isfinite(float(self.process_noise_var)):
            raise TrackingError("process_noise_var must be non-negative and finite")


class KalmanState:
    """6D constant-velocity Kalman filter for 3D position tracking."""

    def __init__(
        self,
        initial_position: Sequence[float] | ArrayLike,
        process_noise_var: float = 0.01,
        obs_noise_var: float = 2.25,
    ) -> None:
        position = _as_vec3(initial_position, "initial_position")
        self.x = np.zeros(6, dtype=np.float64)
        self.x[:3] = position
        self.p = np.eye(6, dtype=np.float64) * 10.0
        self.process_noise_var = _non_negative_float(process_noise_var, "process_noise_var")
        self.obs_noise_var = _positive_float(obs_noise_var, "obs_noise_var")

    def predict(self, dt_s: float) -> None:
        dt = _positive_float(dt_s, "dt_s")
        transition = np.eye(6, dtype=np.float64)
        transition[0, 3] = dt
        transition[1, 4] = dt
        transition[2, 5] = dt
        self.x = transition @ self.x
        self.p = transition @ self.p @ transition.T + _process_noise(dt, self.process_noise_var)

    def update(self, observation: Sequence[float] | ArrayLike) -> None:
        z = _as_vec3(observation, "observation")
        h = np.zeros((3, 6), dtype=np.float64)
        h[:, :3] = np.eye(3, dtype=np.float64)
        residual = z - self.x[:3]
        s = h @ self.p @ h.T + np.eye(3, dtype=np.float64) * self.obs_noise_var
        try:
            gain = self.p @ h.T @ np.linalg.inv(s)
        except np.linalg.LinAlgError:
            return
        self.x = self.x + gain @ residual
        self.p = (np.eye(6, dtype=np.float64) - gain @ h) @ self.p
        self.p = 0.5 * (self.p + self.p.T)

    def mahalanobis_distance_sq(self, observation: Sequence[float] | ArrayLike) -> float:
        z = _as_vec3(observation, "observation")
        residual = z - self.x[:3]
        s = self.p[:3, :3] + np.eye(3, dtype=np.float64) * self.obs_noise_var
        try:
            solved = np.linalg.solve(s, residual)
        except np.linalg.LinAlgError:
            return math.inf
        return float(residual @ solved)

    @property
    def position(self) -> tuple[float, float, float]:
        return float(self.x[0]), float(self.x[1]), float(self.x[2])

    @property
    def velocity(self) -> tuple[float, float, float]:
        return float(self.x[3]), float(self.x[4]), float(self.x[5])

    @property
    def position_uncertainty(self) -> float:
        return float(np.trace(self.p[:3, :3]))


@dataclass
class TrackLifecycle:
    """Small lifecycle state machine for one track."""

    config: TrackerConfig
    state: TrackState = TrackState.TENTATIVE
    hits: int = 0
    active_misses: int = 0
    lost_age_s: float = 0.0

    def hit(self) -> None:
        if self.state.is_terminal:
            return
        self.hits += 1
        self.active_misses = 0
        self.lost_age_s = 0.0
        if self.state in {TrackState.TENTATIVE, TrackState.LOST} and self.hits >= self.config.birth_hits_required:
            self.state = TrackState.ACTIVE

    def miss(self, dt_s: float) -> None:
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.TERMINATED
        elif self.state == TrackState.ACTIVE:
            self.active_misses += 1
            if self.active_misses >= self.config.max_active_misses:
                self.state = TrackState.LOST
                self.lost_age_s = 0.0
        elif self.state == TrackState.LOST:
            self.lost_age_s += max(float(dt_s), 0.0)
            if self.lost_age_s > self.config.max_lost_age_s:
                self.state = TrackState.TERMINATED

    def rescue(self) -> None:
        self.state = TrackState.RESCUED

    def can_reidentify(self) -> bool:
        return self.state == TrackState.LOST and self.lost_age_s <= self.config.max_lost_age_s


@dataclass(frozen=True)
class CsiFingerprint:
    """Biometric and spatial fingerprint used for re-identification."""

    breathing_rate_bpm: float = 0.0
    breathing_amplitude: float = 0.0
    heartbeat_rate_bpm: float | None = None
    location_hint: Sequence[float] = (0.0, 0.0, 0.0)
    sample_count: int = 1

    def __post_init__(self) -> None:
        rate = _non_negative_float(self.breathing_rate_bpm, "breathing_rate_bpm")
        amplitude = float(np.clip(_finite_float(self.breathing_amplitude, "breathing_amplitude"), 0.0, 1.0))
        heartbeat = None
        if self.heartbeat_rate_bpm is not None:
            heartbeat = _non_negative_float(self.heartbeat_rate_bpm, "heartbeat_rate_bpm")
        location = _as_vec3(self.location_hint, "location_hint")
        if self.sample_count <= 0:
            raise TrackingError("sample_count must be positive")
        object.__setattr__(self, "breathing_rate_bpm", rate)
        object.__setattr__(self, "breathing_amplitude", amplitude)
        object.__setattr__(self, "heartbeat_rate_bpm", heartbeat)
        object.__setattr__(self, "location_hint", tuple(float(v) for v in location))
        object.__setattr__(self, "sample_count", int(self.sample_count))

    def distance(self, other: "CsiFingerprint") -> float:
        return fingerprint_distance(self, other)

    def similarity(self, other: "CsiFingerprint") -> float:
        return fingerprint_similarity(self, other)

    def matches(self, other: "CsiFingerprint", threshold: float = 0.35) -> bool:
        return self.distance(other) < threshold

    def update(self, other: "CsiFingerprint", *, alpha: float = 0.3) -> "CsiFingerprint":
        blend = float(np.clip(alpha, 0.0, 1.0))
        keep = 1.0 - blend
        heartbeat = self.heartbeat_rate_bpm
        if heartbeat is not None and other.heartbeat_rate_bpm is not None:
            heartbeat = keep * heartbeat + blend * other.heartbeat_rate_bpm
        elif other.heartbeat_rate_bpm is not None:
            heartbeat = other.heartbeat_rate_bpm
        location = tuple(
            keep * float(a) + blend * float(b) for a, b in zip(self.location_hint, other.location_hint, strict=True)
        )
        return CsiFingerprint(
            breathing_rate_bpm=keep * self.breathing_rate_bpm + blend * other.breathing_rate_bpm,
            breathing_amplitude=keep * self.breathing_amplitude + blend * other.breathing_amplitude,
            heartbeat_rate_bpm=heartbeat,
            location_hint=location,
            sample_count=self.sample_count + 1,
        )


def fingerprint_distance(left: CsiFingerprint, right: CsiFingerprint) -> float:
    """Weighted normalized distance between two CSI fingerprints."""

    d_breathing_rate = abs(left.breathing_rate_bpm - right.breathing_rate_bpm) / 30.0
    d_breathing_amp = abs(left.breathing_amplitude - right.breathing_amplitude)
    left_loc = np.asarray(left.location_hint, dtype=np.float64)
    right_loc = np.asarray(right.location_hint, dtype=np.float64)
    d_location = float(np.linalg.norm(left_loc - right_loc) / 20.0)

    weights: dict[str, float] = {
        "breathing_rate": 0.40,
        "breathing_amp": 0.25,
        "location": 0.15,
    }
    terms = {
        "breathing_rate": d_breathing_rate,
        "breathing_amp": d_breathing_amp,
        "location": d_location,
    }
    if left.heartbeat_rate_bpm is not None and right.heartbeat_rate_bpm is not None:
        weights["heartbeat"] = 0.20
        terms["heartbeat"] = abs(left.heartbeat_rate_bpm - right.heartbeat_rate_bpm) / 80.0
    total_weight = sum(weights.values())
    return float(sum(weights[name] * terms[name] for name in weights) / max(total_weight, 1e-12))


def fingerprint_similarity(left: CsiFingerprint, right: CsiFingerprint) -> float:
    """Convert fingerprint distance to a bounded similarity score."""

    return float(1.0 / (1.0 + fingerprint_distance(left, right)))


@dataclass(frozen=True)
class DetectionObservation:
    """One tracker observation for a MAT update tick."""

    position: Sequence[float] | None
    fingerprint: CsiFingerprint | None = None
    confidence: float = 1.0
    zone_id: Hashable | None = None

    def __post_init__(self) -> None:
        if self.position is not None:
            object.__setattr__(self, "position", tuple(float(v) for v in _as_vec3(self.position, "position")))
        confidence = _finite_float(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise TrackingError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)

    @property
    def position_array(self) -> FloatArray | None:
        if self.position is None:
            return None
        return np.asarray(self.position, dtype=np.float64)


@dataclass
class Track:
    """Persistent MAT survivor track."""

    track_id: int
    kalman: KalmanState
    fingerprint: CsiFingerprint
    lifecycle: TrackLifecycle
    confidence: float = 1.0
    zone_id: Hashable | None = None
    age: int = 1
    total_misses: int = 0

    @classmethod
    def from_observation(cls, track_id: int, observation: DetectionObservation, config: TrackerConfig) -> "Track":
        position = observation.position_array if observation.position_array is not None else np.zeros(3, dtype=np.float64)
        fingerprint = observation.fingerprint or CsiFingerprint(location_hint=position)
        lifecycle = TrackLifecycle(config)
        lifecycle.hit()
        return cls(
            track_id=int(track_id),
            kalman=KalmanState(position, config.process_noise_var, config.obs_noise_var),
            fingerprint=fingerprint,
            lifecycle=lifecycle,
            confidence=observation.confidence,
            zone_id=observation.zone_id,
        )

    @property
    def state(self) -> TrackState:
        return self.lifecycle.state

    @property
    def position(self) -> tuple[float, float, float]:
        return self.kalman.position

    @property
    def velocity(self) -> tuple[float, float, float]:
        return self.kalman.velocity

    def predict(self, dt_s: float) -> None:
        self.kalman.predict(dt_s)
        self.age += 1

    def update(self, observation: DetectionObservation) -> None:
        if observation.position_array is not None:
            self.kalman.update(observation.position_array)
        if observation.fingerprint is not None:
            self.fingerprint = self.fingerprint.update(observation.fingerprint)
        self.confidence = 0.7 * self.confidence + 0.3 * observation.confidence
        self.lifecycle.hit()
        self.total_misses = 0

    def miss(self, dt_s: float) -> None:
        self.lifecycle.miss(dt_s)
        self.total_misses += 1
        self.confidence *= 0.85

    def mark_rescued(self) -> None:
        self.lifecycle.rescue()


@dataclass(frozen=True)
class TrackerUpdateResult:
    """Summary of one tracker update."""

    matched_track_ids: tuple[int, ...] = ()
    born_track_ids: tuple[int, ...] = ()
    lost_track_ids: tuple[int, ...] = ()
    reidentified_track_ids: tuple[int, ...] = ()
    terminated_track_ids: tuple[int, ...] = ()
    dropped_track_ids: tuple[int, ...] = ()


class SurvivorTracker:
    """Greedy nearest-neighbour tracker for MAT 3D position observations."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.tracks: list[Track] = []
        self.next_track_id = 1

    def update(self, observations: Sequence[DetectionObservation | Sequence[float]], dt_s: float = 1.0) -> TrackerUpdateResult:
        dt = _positive_float(dt_s, "dt_s")
        normalized = [_coerce_observation(observation) for observation in observations]

        for track in self.tracks:
            if track.state.is_alive:
                track.predict(dt)

        matched_ids: list[int] = []
        born_ids: list[int] = []
        lost_ids: list[int] = []
        reidentified_ids: list[int] = []
        terminated_ids: list[int] = []

        active_indices = [index for index, track in enumerate(self.tracks) if track.state in {TrackState.TENTATIVE, TrackState.ACTIVE}]
        assignments = self._assign(active_indices, normalized)
        assigned_observations = {obs_idx for _track_idx, obs_idx in assignments}
        matched_track_indices = {track_idx for track_idx, _obs_idx in assignments}

        for track_idx, obs_idx in assignments:
            track = self.tracks[track_idx]
            track.update(normalized[obs_idx])
            matched_ids.append(track.track_id)

        for obs_idx, observation in enumerate(normalized):
            if obs_idx in assigned_observations:
                continue
            lost_idx = self._best_reid_track(observation)
            if lost_idx is not None:
                track = self.tracks[lost_idx]
                track.update(observation)
                reidentified_ids.append(track.track_id)
                matched_track_indices.add(lost_idx)
                assigned_observations.add(obs_idx)

        for obs_idx, observation in enumerate(normalized):
            if obs_idx in assigned_observations:
                continue
            track = Track.from_observation(self.next_track_id, observation, self.config)
            self.next_track_id += 1
            born_ids.append(track.track_id)
            self.tracks.append(track)
            matched_track_indices.add(len(self.tracks) - 1)

        for track_idx, track in enumerate(self.tracks):
            if track_idx in matched_track_indices:
                continue
            if not track.state.is_alive:
                continue
            previous_state = track.state
            track.miss(dt)
            if previous_state == TrackState.ACTIVE and track.state == TrackState.LOST:
                lost_ids.append(track.track_id)
            if track.state == TrackState.TERMINATED:
                terminated_ids.append(track.track_id)

        dropped_ids: list[int] = []
        if self.config.drop_terminal:
            kept: list[Track] = []
            for track in self.tracks:
                if track.state.is_terminal:
                    dropped_ids.append(track.track_id)
                else:
                    kept.append(track)
            self.tracks = kept

        return TrackerUpdateResult(
            matched_track_ids=tuple(matched_ids),
            born_track_ids=tuple(born_ids),
            lost_track_ids=tuple(lost_ids),
            reidentified_track_ids=tuple(reidentified_ids),
            terminated_track_ids=tuple(terminated_ids),
            dropped_track_ids=tuple(dropped_ids),
        )

    def active_tracks(self) -> list[Track]:
        return [track for track in self.tracks if track.state.is_alive]

    def visible_tracks(self) -> list[Track]:
        return [track for track in self.tracks if track.state.is_visible]

    def get_track(self, track_id: int) -> Track | None:
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    def mark_rescued(self, track_id: int) -> bool:
        track = self.get_track(track_id)
        if track is None:
            return False
        track.mark_rescued()
        if self.config.drop_terminal:
            self.tracks = [candidate for candidate in self.tracks if candidate.track_id != track_id]
        return True

    @property
    def track_count(self) -> int:
        return len(self.tracks)

    def _assign(self, active_indices: Sequence[int], observations: Sequence[DetectionObservation]) -> list[tuple[int, int]]:
        candidates: list[tuple[float, int, int]] = []
        for track_idx in active_indices:
            track = self.tracks[track_idx]
            for obs_idx, observation in enumerate(observations):
                point = observation.position_array
                if point is None:
                    continue
                cost = track.kalman.mahalanobis_distance_sq(point)
                if cost <= self.config.gate_mahalanobis_sq:
                    candidates.append((cost, track_idx, obs_idx))

        assignments: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_observations: set[int] = set()
        for _cost, track_idx, obs_idx in sorted(candidates, key=lambda item: item[0]):
            if track_idx in used_tracks or obs_idx in used_observations:
                continue
            assignments.append((track_idx, obs_idx))
            used_tracks.add(track_idx)
            used_observations.add(obs_idx)
        return assignments

    def _best_reid_track(self, observation: DetectionObservation) -> int | None:
        if observation.fingerprint is None:
            return None
        best_idx: int | None = None
        best_distance = math.inf
        for track_idx, track in enumerate(self.tracks):
            if not track.lifecycle.can_reidentify():
                continue
            distance = track.fingerprint.distance(observation.fingerprint)
            if distance < best_distance:
                best_idx = track_idx
                best_distance = distance
        if best_idx is not None and best_distance < self.config.reid_threshold:
            return best_idx
        return None


NearestNeighborTracker = SurvivorTracker


def _process_noise(dt: float, variance: float) -> FloatArray:
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    q = np.zeros((6, 6), dtype=np.float64)
    for axis in range(3):
        q[axis, axis] = dt4 / 4.0 * variance
        q[axis, axis + 3] = dt3 / 2.0 * variance
        q[axis + 3, axis] = dt3 / 2.0 * variance
        q[axis + 3, axis + 3] = dt2 * variance
    return q


def _coerce_observation(observation: DetectionObservation | Sequence[float]) -> DetectionObservation:
    if isinstance(observation, DetectionObservation):
        return observation
    return DetectionObservation(position=observation)


def _as_vec3(value: Sequence[float] | ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise TrackingError(f"{name} must be a 3D vector")
    if not np.all(np.isfinite(arr)):
        raise TrackingError(f"{name} contains non-finite values")
    return np.array(arr, dtype=np.float64, copy=True)


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise TrackingError(f"{name} must be finite")
    return result


def _positive_float(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise TrackingError(f"{name} must be positive")
    return result


def _non_negative_float(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise TrackingError(f"{name} must be non-negative")
    return result


__all__ = [
    "CsiFingerprint",
    "DetectionObservation",
    "KalmanState",
    "NearestNeighborTracker",
    "SurvivorTracker",
    "Track",
    "TrackLifecycle",
    "TrackState",
    "TrackerConfig",
    "TrackerUpdateResult",
    "TrackingError",
    "fingerprint_distance",
    "fingerprint_similarity",
]
