"""Simulation-only CSI payload and multiview fusion helpers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from ruview.swarm import CsiDetection, NodeId, Position3D, coerce_node_id


@dataclass(frozen=True)
class PayloadConfig:
    """Configuration for a simulated CSI sensing payload."""

    scan_freq_hz: float = 10.0
    detection_range_m: float = 28.0
    confidence_threshold: float = 0.6
    esp32_baud_rate: int = 921600

    def __post_init__(self) -> None:
        if self.scan_freq_hz <= 0.0:
            raise ValueError("scan_freq_hz must be positive")
        if self.detection_range_m <= 0.0:
            raise ValueError("detection_range_m must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")


@dataclass(frozen=True)
class FusedDetection:
    """Confidence-weighted victim estimate from multiple viewpoints."""

    confidence: float
    estimated_position: Position3D
    contributing_drones: tuple[NodeId, ...]
    uncertainty_m: float
    geometric_diversity: float


@dataclass(frozen=True)
class MultiViewFusion:
    """Fuse CSI detections using confidence weights and viewpoint geometry."""

    min_viewpoints: int = 2
    min_confidence: float = 0.5
    base_uncertainty_m: float = 5.0

    def fuse(
        self,
        detections: Sequence[CsiDetection],
        drone_positions: Sequence[tuple[NodeId | int, Position3D]],
    ) -> FusedDetection | None:
        """Fuse detections or return ``None`` if viewpoints are insufficient."""

        positions_by_id = {coerce_node_id(node_id): position for node_id, position in drone_positions}
        valid: list[tuple[CsiDetection, Position3D]] = []
        for detection in detections:
            if detection.confidence < self.min_confidence or detection.victim_position is None:
                continue
            drone_position = positions_by_id.get(detection.drone_id)
            if drone_position is not None:
                valid.append((detection, drone_position))

        if len(valid) < self.min_viewpoints:
            return None

        total_weight = sum(detection.confidence for detection, _ in valid)
        if total_weight <= 0.0:
            return None

        fused_x = fused_y = fused_z = 0.0
        fused_conf = 0.0
        for detection, _ in valid:
            weight = detection.confidence / total_weight
            assert detection.victim_position is not None
            fused_x += weight * detection.victim_position.x
            fused_y += weight * detection.victim_position.y
            fused_z += weight * detection.victim_position.z
            fused_conf += weight * detection.confidence

        drone_pos_list = [position for _, position in valid]
        gdi = geometric_diversity_index(drone_pos_list)
        gdi_factor = _clamp(1.0 + gdi / math.pi, 1.0, 2.0)
        uncertainty = self.base_uncertainty_m / (math.sqrt(len(valid)) * gdi_factor)

        return FusedDetection(
            confidence=fused_conf,
            estimated_position=Position3D(fused_x, fused_y, fused_z),
            contributing_drones=tuple(detection.drone_id for detection, _ in valid),
            uncertainty_m=uncertainty,
            geometric_diversity=gdi,
        )


def synthetic_payload_scan(
    node_id: NodeId | int,
    drone_pos: Position3D,
    victims: Sequence[Position3D],
    *,
    config: PayloadConfig | None = None,
    noise_std: float = 0.0,
    seed: int = 0,
    timestamp_ms: int = 0,
) -> CsiDetection | None:
    """Return a deterministic synthetic CSI detection for simulation tests."""

    cfg = config or PayloadConfig()
    rng = random.Random(int(seed) + int(coerce_node_id(node_id)) * 1_000_003)
    for victim in victims:
        distance = drone_pos.distance_to(victim)
        if distance >= cfg.detection_range_m:
            continue
        base_confidence = math.exp(-distance / cfg.detection_range_m)
        noise = rng.uniform(-noise_std, noise_std) if noise_std > 0.0 else 0.0
        confidence = _clamp(base_confidence + noise, 0.0, 1.0)
        if confidence < cfg.confidence_threshold:
            continue
        pos_noise_x = rng.uniform(-noise_std * 5.0, noise_std * 5.0) if noise_std > 0.0 else 0.0
        pos_noise_y = rng.uniform(-noise_std * 5.0, noise_std * 5.0) if noise_std > 0.0 else 0.0
        return CsiDetection(
            drone_id=coerce_node_id(node_id),
            confidence=confidence,
            victim_position=Position3D(victim.x + pos_noise_x, victim.y + pos_noise_y, victim.z),
            timestamp_ms=timestamp_ms,
        )
    return None


def geometric_diversity_index(positions: Sequence[Position3D]) -> float:
    """Average pairwise angular separation around the viewpoint centroid."""

    if len(positions) < 2:
        return 0.0
    n = float(len(positions))
    centroid_x = sum(position.x for position in positions) / n
    centroid_y = sum(position.y for position in positions) / n
    total_angle = 0.0
    pairs = 0
    for i, first in enumerate(positions):
        ax = first.x - centroid_x
        ay = first.y - centroid_y
        mag_a = max(1e-9, math.hypot(ax, ay))
        for second in positions[i + 1 :]:
            bx = second.x - centroid_x
            by = second.y - centroid_y
            mag_b = max(1e-9, math.hypot(bx, by))
            cos_angle = _clamp((ax * bx + ay * by) / (mag_a * mag_b), -1.0, 1.0)
            total_angle += math.acos(cos_angle)
            pairs += 1
    return total_angle / pairs if pairs else 0.0


def gdop(observers: Sequence[Position3D], target: Position3D) -> float | None:
    """2-D geometric dilution of precision for observer-target geometry."""

    if len(observers) < 2:
        return None
    a = b = d = 0.0
    for observer in observers:
        dx = observer.x - target.x
        dy = observer.y - target.y
        range_xy = math.hypot(dx, dy)
        if range_xy < 1e-9:
            return None
        ux = dx / range_xy
        uy = dy / range_xy
        a += ux * ux
        b += ux * uy
        d += uy * uy
    det = a * d - b * b
    if abs(det) < 1e-12:
        return None
    trace_inv = (a + d) / det
    if trace_inv <= 0.0 or not math.isfinite(trace_inv):
        return None
    return math.sqrt(trace_inv)


def mean_uncertainty(results: Sequence[FusedDetection | float]) -> float:
    """Mean finite uncertainty from fused detections or raw uncertainty values."""

    values: list[float] = []
    for item in results:
        value = item.uncertainty_m if isinstance(item, FusedDetection) else float(item)
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else 0.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(float(hi), max(float(lo), float(value)))


__all__ = [
    "FusedDetection",
    "MultiViewFusion",
    "PayloadConfig",
    "gdop",
    "geometric_diversity_index",
    "mean_uncertainty",
    "synthetic_payload_scan",
]
