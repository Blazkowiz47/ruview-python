"""Research-domain models for the MAT survivor-detection pipeline.

These classes mirror the behavior of the Rust MAT domain layer at the value
object and aggregate level while keeping the Python surface small and
deterministic for experiments.
"""

from __future__ import annotations

import math
from dataclasses import InitVar, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from itertools import count
from typing import Iterable


_event_ids = count(1)
_zone_ids = count(1)
_survivor_ids = count(1)


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


class DisasterType(str, Enum):
    """Type of disaster scenario under search."""

    BUILDING_COLLAPSE = "building_collapse"
    EARTHQUAKE = "earthquake"
    LANDSLIDE = "landslide"
    AVALANCHE = "avalanche"
    FLOOD = "flood"
    MINE_COLLAPSE = "mine_collapse"
    INDUSTRIAL = "industrial"
    TUNNEL_COLLAPSE = "tunnel_collapse"
    UNKNOWN = "unknown"

    def typical_debris_profile(self) -> "DebrisProfile":
        """Return a lightweight default debris profile for this event type."""

        profiles = {
            DisasterType.BUILDING_COLLAPSE: DebrisProfile(
                primary_material=DebrisMaterial.MIXED,
                void_fraction=0.25,
                moisture_content=MoistureLevel.DRY,
                metal_content=MetalContent.MODERATE,
            ),
            DisasterType.EARTHQUAKE: DebrisProfile(
                primary_material=DebrisMaterial.HEAVY_CONCRETE,
                void_fraction=0.20,
                moisture_content=MoistureLevel.DRY,
                metal_content=MetalContent.MODERATE,
            ),
            DisasterType.AVALANCHE: DebrisProfile(
                primary_material=DebrisMaterial.SNOW,
                void_fraction=0.40,
                moisture_content=MoistureLevel.WET,
                metal_content=MetalContent.NONE,
            ),
            DisasterType.LANDSLIDE: DebrisProfile(
                primary_material=DebrisMaterial.SOIL,
                void_fraction=0.15,
                moisture_content=MoistureLevel.WET,
                metal_content=MetalContent.NONE,
            ),
            DisasterType.FLOOD: DebrisProfile(
                primary_material=DebrisMaterial.MIXED,
                void_fraction=0.30,
                moisture_content=MoistureLevel.SATURATED,
                metal_content=MetalContent.LOW,
            ),
            DisasterType.MINE_COLLAPSE: DebrisProfile(
                primary_material=DebrisMaterial.SOIL,
                void_fraction=0.20,
                moisture_content=MoistureLevel.DAMP,
                metal_content=MetalContent.LOW,
            ),
            DisasterType.TUNNEL_COLLAPSE: DebrisProfile(
                primary_material=DebrisMaterial.SOIL,
                void_fraction=0.20,
                moisture_content=MoistureLevel.DAMP,
                metal_content=MetalContent.LOW,
            ),
            DisasterType.INDUSTRIAL: DebrisProfile(
                primary_material=DebrisMaterial.METAL,
                void_fraction=0.35,
                moisture_content=MoistureLevel.DRY,
                metal_content=MetalContent.HIGH,
            ),
        }
        return profiles.get(self, DebrisProfile())

    def expected_survival_hours(self) -> int:
        """Return a rough research prior for event urgency."""

        return {
            DisasterType.AVALANCHE: 2,
            DisasterType.FLOOD: 6,
            DisasterType.MINE_COLLAPSE: 72,
            DisasterType.BUILDING_COLLAPSE: 96,
            DisasterType.EARTHQUAKE: 120,
            DisasterType.LANDSLIDE: 48,
            DisasterType.TUNNEL_COLLAPSE: 72,
            DisasterType.INDUSTRIAL: 72,
            DisasterType.UNKNOWN: 72,
        }[self]


class EventStatus(str, Enum):
    """Operational state of a disaster event."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    SECONDARY_SEARCH = "secondary_search"
    CLOSED = "closed"


class DebrisMaterial(str, Enum):
    """Primary material between sensor and survivor."""

    LIGHT_CONCRETE = "light_concrete"
    HEAVY_CONCRETE = "heavy_concrete"
    WOOD = "wood"
    SOIL = "soil"
    MIXED = "mixed"
    SNOW = "snow"
    METAL = "metal"

    def attenuation_coefficient(self) -> float:
        """Approximate RF attenuation coefficient in dB/m."""

        return {
            DebrisMaterial.SNOW: 0.5,
            DebrisMaterial.WOOD: 1.5,
            DebrisMaterial.LIGHT_CONCRETE: 3.0,
            DebrisMaterial.SOIL: 4.0,
            DebrisMaterial.MIXED: 4.5,
            DebrisMaterial.HEAVY_CONCRETE: 6.0,
            DebrisMaterial.METAL: 20.0,
        }[self]


class MoistureLevel(str, Enum):
    """Moisture content in debris."""

    DRY = "dry"
    DAMP = "damp"
    WET = "wet"
    SATURATED = "saturated"

    def attenuation_multiplier(self) -> float:
        """Return attenuation multiplier for this moisture level."""

        return {
            MoistureLevel.DRY: 1.0,
            MoistureLevel.DAMP: 1.3,
            MoistureLevel.WET: 1.8,
            MoistureLevel.SATURATED: 2.5,
        }[self]


class MetalContent(str, Enum):
    """Qualitative metal content in a debris profile."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    BLOCKING = "blocking"


@dataclass(frozen=True)
class LocationUncertainty:
    """Uncertainty bounds for a position estimate."""

    horizontal_error: float = 2.0
    vertical_error: float = 1.0
    confidence: float = 0.95

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizontal_error", max(float(self.horizontal_error), 0.0))
        object.__setattr__(self, "vertical_error", max(float(self.vertical_error), 0.0))
        object.__setattr__(self, "confidence", _clamp01(self.confidence))

    @classmethod
    def high_confidence(
        cls,
        horizontal_error: float,
        vertical_error: float,
    ) -> "LocationUncertainty":
        """Create a 99% confidence uncertainty estimate."""

        return cls(horizontal_error=horizontal_error, vertical_error=vertical_error, confidence=0.99)

    def is_actionable(self) -> bool:
        """Return whether this precision is useful for rescue action."""

        return self.horizontal_error <= 3.0 and self.confidence >= 0.8

    def combine(self, other: "LocationUncertainty") -> "LocationUncertainty":
        """Fuse two independent uncertainty estimates using inverse variance."""

        horizontal = _combine_standard_errors(self.horizontal_error, other.horizontal_error)
        vertical = _combine_standard_errors(self.vertical_error, other.vertical_error)
        total_confidence = max(self.confidence + other.confidence, 1e-12)
        weight_self = self.confidence / total_confidence
        weight_other = other.confidence / total_confidence
        confidence = min(weight_self * self.confidence + weight_other * other.confidence, 0.99)
        return LocationUncertainty(horizontal, vertical, confidence)


@dataclass(frozen=True)
class Coordinates3D:
    """3D coordinates in meters relative to a local reference point."""

    x: float
    y: float
    z: float
    uncertainty: LocationUncertainty = field(default_factory=LocationUncertainty)

    def distance_to(self, other: "Coordinates3D") -> float:
        """Return 3D Euclidean distance to another point."""

        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )

    def horizontal_distance_to(self, other: "Coordinates3D") -> float:
        """Return horizontal 2D distance to another point."""

        return math.hypot(self.x - other.x, self.y - other.y)

    @property
    def depth(self) -> float:
        """Depth below surface in meters, with surface/above-surface as zero."""

        return -min(self.z, 0.0)

    def is_buried(self) -> bool:
        """Return whether the coordinate is below the surface."""

        return self.z < 0.0

    def confidence_radius(self) -> float:
        """Return horizontal 95% confidence radius."""

        return self.uncertainty.horizontal_error


@dataclass(frozen=True)
class DebrisProfile:
    """Basic debris composition and propagation modifiers."""

    primary_material: DebrisMaterial = DebrisMaterial.MIXED
    void_fraction: float = 0.3
    moisture_content: MoistureLevel = MoistureLevel.DRY
    metal_content: MetalContent = MetalContent.NONE

    def __post_init__(self) -> None:
        object.__setattr__(self, "void_fraction", _clamp01(self.void_fraction))

    def attenuation_factor(self) -> float:
        """Return a compact signal-attenuation factor."""

        base = self.primary_material.attenuation_coefficient()
        moisture = self.moisture_content.attenuation_multiplier()
        voids = 1.0 - self.void_fraction * 0.3
        return base * moisture * voids

    def is_penetrable(self) -> bool:
        """Return whether the debris profile is likely to permit WiFi sensing."""

        return (
            self.metal_content not in {MetalContent.HIGH, MetalContent.BLOCKING}
            and self.primary_material.attenuation_coefficient() < 5.0
        )


@dataclass(frozen=True)
class DepthEstimate:
    """Depth estimate with uncertainty and debris context."""

    depth: float
    uncertainty: float
    debris_profile: DebrisProfile = field(default_factory=DebrisProfile)
    confidence: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "depth", max(float(self.depth), 0.0))
        object.__setattr__(self, "uncertainty", max(float(self.uncertainty), 0.0))
        object.__setattr__(self, "confidence", _clamp01(self.confidence))

    def min_depth(self) -> float:
        """Return lower depth bound."""

        return max(self.depth - self.uncertainty, 0.0)

    def max_depth(self) -> float:
        """Return upper depth bound."""

        return self.depth + self.uncertainty

    def is_shallow(self) -> bool:
        """Return whether the depth is below 1.5 m."""

        return self.depth < 1.5

    def is_moderate(self) -> bool:
        """Return whether the depth is between 1.5 m and 3 m."""

        return 1.5 <= self.depth < 3.0

    def is_deep(self) -> bool:
        """Return whether the depth is 3 m or deeper."""

        return self.depth >= 3.0


class ZoneShape(str, Enum):
    """Geometric shape used by scan-zone bounds."""

    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    POLYGON = "polygon"


@dataclass(frozen=True)
class ZoneBounds:
    """2D scan-zone bounds in local coordinates."""

    shape: ZoneShape
    values: tuple[float, ...] = ()
    vertices: tuple[tuple[float, float], ...] = ()

    @classmethod
    def rectangle(cls, min_x: float, min_y: float, max_x: float, max_y: float) -> "ZoneBounds":
        """Create rectangular bounds."""

        return cls(ZoneShape.RECTANGLE, (float(min_x), float(min_y), float(max_x), float(max_y)))

    @classmethod
    def circle(cls, center_x: float, center_y: float, radius: float) -> "ZoneBounds":
        """Create circular bounds."""

        return cls(ZoneShape.CIRCLE, (float(center_x), float(center_y), max(float(radius), 0.0)))

    @classmethod
    def polygon(cls, vertices: Iterable[tuple[float, float]]) -> "ZoneBounds":
        """Create polygon bounds from ordered vertices."""

        points = tuple((float(x), float(y)) for x, y in vertices)
        return cls(ZoneShape.POLYGON, (), points)

    def area(self) -> float:
        """Return area in square meters."""

        if self.shape is ZoneShape.RECTANGLE:
            min_x, min_y, max_x, max_y = self.values
            return abs((max_x - min_x) * (max_y - min_y))
        if self.shape is ZoneShape.CIRCLE:
            _, _, radius = self.values
            return math.pi * radius * radius
        if len(self.vertices) < 3:
            return 0.0
        total = 0.0
        for index, (x_i, y_i) in enumerate(self.vertices):
            x_j, y_j = self.vertices[(index + 1) % len(self.vertices)]
            total += x_i * y_j - x_j * y_i
        return abs(total) / 2.0

    def contains(self, x: float, y: float) -> bool:
        """Return whether a point is inside the bounds."""

        x = float(x)
        y = float(y)
        if self.shape is ZoneShape.RECTANGLE:
            min_x, min_y, max_x, max_y = self.values
            low_x, high_x = sorted((min_x, max_x))
            low_y, high_y = sorted((min_y, max_y))
            return low_x <= x <= high_x and low_y <= y <= high_y
        if self.shape is ZoneShape.CIRCLE:
            center_x, center_y, radius = self.values
            return math.hypot(x - center_x, y - center_y) <= radius
        return _point_in_polygon(x, y, self.vertices)

    def center(self) -> tuple[float, float]:
        """Return a simple geometric center for the bounds."""

        if self.shape is ZoneShape.RECTANGLE:
            min_x, min_y, max_x, max_y = self.values
            return ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        if self.shape is ZoneShape.CIRCLE:
            center_x, center_y, _ = self.values
            return (center_x, center_y)
        if not self.vertices:
            return (0.0, 0.0)
        return (
            sum(x for x, _ in self.vertices) / len(self.vertices),
            sum(y for _, y in self.vertices) / len(self.vertices),
        )


class ZoneStatus(str, Enum):
    """Operational state of a scan zone."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETE = "complete"
    INACCESSIBLE = "inaccessible"
    DEACTIVATED = "deactivated"


class ScanResolution(str, Enum):
    """Scan-resolution preset."""

    QUICK = "quick"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"

    def time_multiplier(self) -> float:
        """Return relative scan time for this preset."""

        return {
            ScanResolution.QUICK: 0.5,
            ScanResolution.STANDARD: 1.0,
            ScanResolution.HIGH: 2.0,
            ScanResolution.MAXIMUM: 4.0,
        }[self]


@dataclass
class ScanParameters:
    """User-facing scan settings."""

    sensitivity: float = 0.8
    max_depth: float = 5.0
    resolution: ScanResolution = ScanResolution.STANDARD
    enhanced_breathing: bool = True
    heartbeat_detection: bool = False

    def __post_init__(self) -> None:
        self.sensitivity = _clamp01(self.sensitivity)
        self.max_depth = max(float(self.max_depth), 0.0)


class SensorType(str, Enum):
    """Role of a WiFi sensor in a zone."""

    TRANSMITTER = "transmitter"
    RECEIVER = "receiver"
    TRANSCEIVER = "transceiver"


@dataclass
class SensorPosition:
    """Known position of one MAT sensing node."""

    id: str
    x: float
    y: float
    z: float
    sensor_type: SensorType = SensorType.TRANSCEIVER
    is_operational: bool = True


@dataclass
class ScanZone:
    """Defined area being monitored for survivors."""

    name: str
    bounds: ZoneBounds
    id: str = field(default_factory=lambda: f"zone-{next(_zone_ids)}")
    sensor_positions: list[SensorPosition] = field(default_factory=list)
    parameters: ScanParameters = field(default_factory=ScanParameters)
    status: ZoneStatus = ZoneStatus.ACTIVE
    created_at: datetime = field(default_factory=_now_utc)
    last_scan: datetime | None = None
    scan_count: int = 0
    detections_count: int = 0
    expected_scans: int = 1

    def __post_init__(self) -> None:
        self.expected_scans = max(int(self.expected_scans), 1)
        self.scan_count = max(int(self.scan_count), 0)
        self.detections_count = max(int(self.detections_count), 0)

    def add_sensor(self, sensor: SensorPosition) -> None:
        """Add a sensor position to this zone."""

        self.sensor_positions.append(sensor)

    def remove_sensor(self, sensor_id: str) -> None:
        """Remove a sensor by id."""

        self.sensor_positions = [sensor for sensor in self.sensor_positions if sensor.id != sensor_id]

    def pause(self) -> None:
        """Pause scanning."""

        self.status = ZoneStatus.PAUSED

    def resume(self) -> None:
        """Resume scanning if the zone was paused."""

        if self.status is ZoneStatus.PAUSED:
            self.status = ZoneStatus.ACTIVE

    def complete(self) -> None:
        """Mark the zone complete."""

        self.status = ZoneStatus.COMPLETE

    def record_scan(self, found_detections: int = 0, *, at: datetime | None = None) -> None:
        """Record one completed scan pass."""

        self.last_scan = at or _now_utc()
        self.scan_count += 1
        self.detections_count += max(int(found_detections), 0)

    def contains_point(self, x: float, y: float) -> bool:
        """Return whether a local point is inside this zone."""

        return self.bounds.contains(x, y)

    def area(self) -> float:
        """Return zone area in square meters."""

        return self.bounds.area()

    def progress_fraction(self) -> float:
        """Return scan progress in [0, 1]."""

        if self.status is ZoneStatus.COMPLETE:
            return 1.0
        return _clamp01(self.scan_count / self.expected_scans)

    def has_sufficient_sensors(self) -> bool:
        """Return whether at least three operational sensors are available."""

        return sum(sensor.is_operational for sensor in self.sensor_positions) >= 3


class SurvivorStatus(str, Enum):
    """Tracking status for a detected survivor."""

    ACTIVE = "active"
    RESCUED = "rescued"
    LOST = "lost"
    DECEASED = "deceased"
    FALSE_POSITIVE = "false_positive"


class SurvivorCondition(str, Enum):
    """Research triage-style condition label."""

    UNKNOWN = "unknown"
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    MINOR = "minor"
    DECEASED = "deceased"


class AgeCategory(str, Enum):
    """Estimated age category from vital patterns."""

    INFANT = "infant"
    CHILD = "child"
    ADULT = "adult"
    ELDERLY = "elderly"
    UNKNOWN = "unknown"


@dataclass
class SurvivorMetadata:
    """Operator metadata attached to a survivor."""

    estimated_age_category: AgeCategory | None = None
    notes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    assigned_team: str | None = None


@dataclass(frozen=True)
class ConfidenceScore:
    """Clamped confidence score in [0, 1]."""

    value: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _clamp01(self.value))

    def is_high(self) -> bool:
        """Return whether confidence is at least 0.8."""

        return self.value >= 0.8

    def is_medium(self) -> bool:
        """Return whether confidence is at least 0.5."""

        return self.value >= 0.5

    def is_low(self) -> bool:
        """Return whether confidence is below 0.5."""

        return self.value < 0.5

    def __float__(self) -> float:
        return self.value


class BreathingType(str, Enum):
    """Detected breathing pattern."""

    NORMAL = "normal"
    SHALLOW = "shallow"
    LABORED = "labored"
    IRREGULAR = "irregular"
    AGONAL = "agonal"
    APNEA = "apnea"


@dataclass(frozen=True)
class BreathingPattern:
    """Breathing pattern detected from CSI amplitude modulation."""

    rate_bpm: float
    amplitude: float
    regularity: float
    pattern_type: BreathingType = BreathingType.NORMAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "rate_bpm", max(float(self.rate_bpm), 0.0))
        object.__setattr__(self, "amplitude", max(float(self.amplitude), 0.0))
        object.__setattr__(self, "regularity", _clamp01(self.regularity))

    def is_normal_rate(self) -> bool:
        """Return whether the rate is normal for an adult."""

        return 12.0 <= self.rate_bpm <= 20.0

    def is_bradypnea(self) -> bool:
        """Return whether the rate is critically low."""

        return self.rate_bpm < 10.0

    def is_tachypnea(self) -> bool:
        """Return whether the rate is critically high."""

        return self.rate_bpm > 30.0

    def confidence(self) -> float:
        """Return signal confidence from amplitude and regularity."""

        return _clamp01(self.amplitude * self.regularity)


class SignalStrength(str, Enum):
    """Heartbeat signal-strength category."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    VERY_WEAK = "very_weak"

    def confidence(self) -> float:
        """Return confidence associated with this strength."""

        return {
            SignalStrength.STRONG: 0.9,
            SignalStrength.MODERATE: 0.7,
            SignalStrength.WEAK: 0.4,
            SignalStrength.VERY_WEAK: 0.2,
        }[self]


@dataclass(frozen=True)
class HeartbeatSignature:
    """Heartbeat signature from phase or micro-Doppler features."""

    rate_bpm: float
    variability: float
    strength: SignalStrength = SignalStrength.WEAK

    def __post_init__(self) -> None:
        object.__setattr__(self, "rate_bpm", max(float(self.rate_bpm), 0.0))
        object.__setattr__(self, "variability", _clamp01(self.variability))

    def is_normal_rate(self) -> bool:
        """Return whether the heart rate is normal for an adult."""

        return 60.0 <= self.rate_bpm <= 100.0

    def is_bradycardia(self) -> bool:
        """Return whether the rate indicates bradycardia."""

        return self.rate_bpm < 50.0

    def is_tachycardia(self) -> bool:
        """Return whether the rate indicates tachycardia."""

        return self.rate_bpm > 120.0

    def confidence(self) -> float:
        """Return signal confidence from strength category."""

        return self.strength.confidence()


class MovementType(str, Enum):
    """Type of movement detected from CSI variation."""

    NONE = "none"
    GROSS = "gross"
    FINE = "fine"
    TREMOR = "tremor"
    PERIODIC = "periodic"


@dataclass(frozen=True)
class MovementActivity:
    """Movement profile inferred from CSI variation."""

    movement_type: MovementType = MovementType.NONE
    intensity: float = 0.0
    frequency: float = 0.0
    is_voluntary: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "intensity", _clamp01(self.intensity))
        object.__setattr__(self, "frequency", max(float(self.frequency), 0.0))

    def confidence(self) -> float:
        """Return confidence associated with the movement class."""

        return {
            MovementType.NONE: 0.0,
            MovementType.GROSS: 0.9,
            MovementType.FINE: 0.7,
            MovementType.TREMOR: 0.6,
            MovementType.PERIODIC: 0.5,
        }[self.movement_type]

    def indicates_consciousness(self) -> bool:
        """Return whether movement appears voluntary and gross."""

        return self.is_voluntary and self.movement_type is MovementType.GROSS


MovementProfile = MovementActivity


@dataclass
class VitalSignsReading:
    """Combined vital-sign reading at one point in time."""

    breathing: BreathingPattern | None = None
    heartbeat: HeartbeatSignature | None = None
    movement: MovementActivity = field(default_factory=MovementActivity)
    timestamp: datetime = field(default_factory=_now_utc)
    confidence: ConfidenceScore | None = None

    def __post_init__(self) -> None:
        if self.movement is None:
            self.movement = MovementActivity()
        if self.confidence is None:
            self.confidence = self.calculate_confidence(
                self.breathing,
                self.heartbeat,
                self.movement,
            )

    @staticmethod
    def calculate_confidence(
        breathing: BreathingPattern | None,
        heartbeat: HeartbeatSignature | None,
        movement: MovementActivity,
    ) -> ConfidenceScore:
        """Combine available vital confidences with breathing weighted highest."""

        total = 0.0
        weight = 0.0
        if breathing is not None:
            total += breathing.confidence() * 1.5
            weight += 1.5
        if heartbeat is not None:
            total += heartbeat.confidence()
            weight += 1.0
        if movement.movement_type is not MovementType.NONE:
            total += movement.confidence()
            weight += 1.0
        return ConfidenceScore(total / weight if weight else 0.0)

    def has_vitals(self) -> bool:
        """Return whether any vital or movement signal was detected."""

        return self.has_breathing() or self.has_heartbeat() or self.has_movement()

    def has_breathing(self) -> bool:
        """Return whether breathing was detected."""

        return self.breathing is not None

    def has_heartbeat(self) -> bool:
        """Return whether heartbeat was detected."""

        return self.heartbeat is not None

    def has_movement(self) -> bool:
        """Return whether movement was detected."""

        return self.movement.movement_type is not MovementType.NONE


@dataclass
class VitalSignsHistory:
    """Bounded history of vital-sign readings."""

    max_history: int = 100
    readings: list[VitalSignsReading] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.max_history = max(int(self.max_history), 1)
        if len(self.readings) > self.max_history:
            self.readings = self.readings[-self.max_history :]

    def add(self, reading: VitalSignsReading) -> None:
        """Append a reading, dropping oldest entries beyond max history."""

        if len(self.readings) >= self.max_history:
            del self.readings[0]
        self.readings.append(reading)

    def latest(self) -> VitalSignsReading | None:
        """Return the most recent reading if present."""

        return self.readings[-1] if self.readings else None

    def all(self) -> tuple[VitalSignsReading, ...]:
        """Return all readings as an immutable tuple."""

        return tuple(self.readings)

    def average_confidence(self) -> float:
        """Return average confidence across retained readings."""

        if not self.readings:
            return 0.0
        return sum(float(reading.confidence.value) for reading in self.readings) / len(self.readings)

    def is_deteriorating(self) -> bool:
        """Return whether the last three readings show decline."""

        if len(self.readings) < 3:
            return False
        older, previous, latest = self.readings[-3:]
        rates = [
            reading.breathing.rate_bpm if reading.breathing is not None else math.inf
            for reading in (older, previous, latest)
        ]
        breathing_declining = rates[2] < rates[1] < rates[0]
        confidence_declining = (
            latest.confidence.value < previous.confidence.value < older.confidence.value
        )
        return breathing_declining or confidence_declining

    def __len__(self) -> int:
        return len(self.readings)

    def __bool__(self) -> bool:
        return bool(self.readings)


@dataclass
class Survivor:
    """Detected human in a scan zone."""

    zone_id: str
    initial_vitals: InitVar[VitalSignsReading | None] = None
    location: Coordinates3D | None = None
    id: str = field(default_factory=lambda: f"survivor-{next(_survivor_ids)}")
    first_detected: datetime = field(default_factory=_now_utc)
    last_updated: datetime = field(default_factory=_now_utc)
    condition: SurvivorCondition = SurvivorCondition.UNKNOWN
    status: SurvivorStatus = SurvivorStatus.ACTIVE
    confidence: float = 0.0
    metadata: SurvivorMetadata = field(default_factory=SurvivorMetadata)
    alert_sent: bool = False
    history_size: int = 100
    vital_signs: VitalSignsHistory = field(init=False)

    def __post_init__(self, initial_vitals: VitalSignsReading | None) -> None:
        self.vital_signs = VitalSignsHistory(max_history=self.history_size)
        if initial_vitals is not None:
            self.vital_signs.add(initial_vitals)
            self.confidence = self.vital_signs.average_confidence()
            if self.condition is SurvivorCondition.UNKNOWN:
                self.condition = condition_from_vitals(initial_vitals)
        self.confidence = _clamp01(self.confidence)

    def update_vitals(self, reading: VitalSignsReading) -> None:
        """Append a vital reading and refresh condition/confidence."""

        self.vital_signs.add(reading)
        self.confidence = self.vital_signs.average_confidence()
        self.condition = condition_from_vitals(reading)
        if self.condition is SurvivorCondition.DECEASED:
            self.status = SurvivorStatus.DECEASED
        self.last_updated = _now_utc()

    def update_location(self, location: Coordinates3D) -> None:
        """Update the location estimate."""

        self.location = location
        self.last_updated = _now_utc()

    def mark_rescued(self) -> None:
        """Mark the survivor rescued."""

        self.status = SurvivorStatus.RESCUED
        self.last_updated = _now_utc()

    def mark_lost(self) -> None:
        """Mark the survivor signal as lost."""

        self.status = SurvivorStatus.LOST
        self.last_updated = _now_utc()

    def mark_deceased(self) -> None:
        """Mark the survivor deceased."""

        self.status = SurvivorStatus.DECEASED
        self.condition = SurvivorCondition.DECEASED
        self.last_updated = _now_utc()

    def mark_false_positive(self) -> None:
        """Mark the detection as a false positive."""

        self.status = SurvivorStatus.FALSE_POSITIVE
        self.last_updated = _now_utc()

    def should_alert(self) -> bool:
        """Return whether this survivor should emit a first alert."""

        return (
            not self.alert_sent
            and self.status is SurvivorStatus.ACTIVE
            and self.condition in {SurvivorCondition.IMMEDIATE, SurvivorCondition.DELAYED}
            and self.confidence >= 0.5
        )

    def mark_alert_sent(self) -> None:
        """Record that an alert was sent."""

        self.alert_sent = True

    def is_deteriorating(self) -> bool:
        """Return whether vital history is deteriorating."""

        return self.vital_signs.is_deteriorating()


@dataclass
class EventMetadata:
    """Metadata attached to a disaster event."""

    estimated_occupancy: int | None = None
    confirmed_rescued: int = 0
    confirmed_deceased: int = 0
    weather: str | None = None
    lead_agency: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SurvivorCounts:
    """Simple survivor counts by tracking status."""

    active: int = 0
    rescued: int = 0
    lost: int = 0
    deceased: int = 0
    false_positive: int = 0

    @property
    def total(self) -> int:
        """Return total survivor records."""

        return self.active + self.rescued + self.lost + self.deceased + self.false_positive

    @property
    def living(self) -> int:
        """Return active/rescued/lost records not marked deceased or false-positive."""

        return self.active + self.rescued + self.lost


@dataclass
class DisasterEvent:
    """Aggregate root for one MAT disaster response event."""

    event_type: DisasterType = DisasterType.UNKNOWN
    location: tuple[float, float] = (0.0, 0.0)
    description: str = ""
    id: str = field(default_factory=lambda: f"event-{next(_event_ids)}")
    start_time: datetime = field(default_factory=_now_utc)
    scan_zones: list[ScanZone] = field(default_factory=list)
    survivors: list[Survivor] = field(default_factory=list)
    status: EventStatus = EventStatus.INITIALIZING
    metadata: EventMetadata = field(default_factory=EventMetadata)

    def add_zone(self, zone: ScanZone) -> None:
        """Add a zone and activate the event if it was initializing."""

        self.scan_zones.append(zone)
        if self.status is EventStatus.INITIALIZING:
            self.status = EventStatus.ACTIVE

    def activate(self) -> None:
        """Mark the event active."""

        self.status = EventStatus.ACTIVE

    def close(self) -> None:
        """Close the event."""

        self.status = EventStatus.CLOSED

    def suspend(self, reason: str | None = None) -> None:
        """Suspend the event and optionally record a note."""

        self.status = EventStatus.SUSPENDED
        if reason:
            self.metadata.notes.append(f"{_now_utc().isoformat()} suspended: {reason}")

    def resume(self) -> None:
        """Resume active operations after suspension."""

        if self.status is EventStatus.SUSPENDED:
            self.status = EventStatus.ACTIVE
            self.metadata.notes.append(f"{_now_utc().isoformat()} resumed operations")

    def get_zone(self, zone_id: str) -> ScanZone | None:
        """Return a zone by id."""

        return next((zone for zone in self.scan_zones if zone.id == zone_id), None)

    def add_survivor(self, survivor: Survivor) -> Survivor:
        """Add or replace a survivor by id."""

        for index, existing in enumerate(self.survivors):
            if existing.id == survivor.id:
                self.survivors[index] = survivor
                return survivor
        self.survivors.append(survivor)
        return survivor

    def update_survivor(
        self,
        survivor_id: str,
        *,
        vitals: VitalSignsReading | None = None,
        location: Coordinates3D | None = None,
        status: SurvivorStatus | None = None,
        condition: SurvivorCondition | None = None,
    ) -> Survivor | None:
        """Update a survivor record if found."""

        survivor = self.get_survivor(survivor_id)
        if survivor is None:
            return None
        if vitals is not None:
            survivor.update_vitals(vitals)
        if location is not None:
            survivor.update_location(location)
        if status is not None:
            survivor.status = status
            survivor.last_updated = _now_utc()
        if condition is not None:
            survivor.condition = condition
            survivor.last_updated = _now_utc()
        return survivor

    def record_detection(
        self,
        zone_id: str,
        vitals: VitalSignsReading,
        location: Coordinates3D | None = None,
        *,
        match_radius_m: float = 2.0,
    ) -> Survivor:
        """Add a new survivor or update a nearby existing survivor."""

        existing = self.find_nearby_survivor(location, match_radius_m) if location else None
        if existing is not None:
            existing.update_vitals(vitals)
            if location is not None:
                existing.update_location(location)
            return existing
        survivor = Survivor(zone_id=zone_id, initial_vitals=vitals, location=location)
        self.survivors.append(survivor)
        zone = self.get_zone(zone_id)
        if zone is not None:
            zone.detections_count += 1
        return survivor

    def find_nearby_survivor(
        self,
        location: Coordinates3D | None,
        radius_m: float,
    ) -> Survivor | None:
        """Return the first survivor within ``radius_m`` of ``location``."""

        if location is None:
            return None
        for survivor in self.survivors:
            if survivor.location is not None and survivor.location.distance_to(location) < radius_m:
                return survivor
        return None

    def get_survivor(self, survivor_id: str) -> Survivor | None:
        """Return a survivor by id."""

        return next((survivor for survivor in self.survivors if survivor.id == survivor_id), None)

    def survivor_counts(self) -> SurvivorCounts:
        """Return counts by survivor tracking status."""

        counts_by_status = {status: 0 for status in SurvivorStatus}
        for survivor in self.survivors:
            counts_by_status[survivor.status] += 1
        return SurvivorCounts(
            active=counts_by_status[SurvivorStatus.ACTIVE],
            rescued=counts_by_status[SurvivorStatus.RESCUED],
            lost=counts_by_status[SurvivorStatus.LOST],
            deceased=counts_by_status[SurvivorStatus.DECEASED],
            false_positive=counts_by_status[SurvivorStatus.FALSE_POSITIVE],
        )

    def condition_counts(self) -> dict[SurvivorCondition, int]:
        """Return counts by survivor condition."""

        counts_by_condition = {condition: 0 for condition in SurvivorCondition}
        for survivor in self.survivors:
            counts_by_condition[survivor.condition] += 1
        return counts_by_condition

    @property
    def zone_count(self) -> int:
        """Return number of configured zones."""

        return len(self.scan_zones)

    @property
    def survivor_count(self) -> int:
        """Return number of survivor records."""

        return len(self.survivors)


def condition_from_vitals(reading: VitalSignsReading) -> SurvivorCondition:
    """Return a compact triage-style condition from a vital-sign reading."""

    if reading.breathing is not None:
        breathing = reading.breathing
        if breathing.pattern_type in {BreathingType.AGONAL, BreathingType.APNEA}:
            return SurvivorCondition.IMMEDIATE
        if breathing.rate_bpm < 10.0 or breathing.rate_bpm > 30.0:
            return SurvivorCondition.IMMEDIATE
        if not 12.0 <= breathing.rate_bpm <= 24.0:
            return SurvivorCondition.DELAYED if reading.has_movement() else SurvivorCondition.IMMEDIATE
        return SurvivorCondition.MINOR if reading.has_movement() else SurvivorCondition.DELAYED
    if reading.has_movement() or reading.has_heartbeat():
        return SurvivorCondition.IMMEDIATE
    return SurvivorCondition.DECEASED


def _combine_standard_errors(first: float, second: float) -> float:
    first = max(float(first), 0.0)
    second = max(float(second), 0.0)
    if first <= 0.0 or second <= 0.0:
        return 0.0
    return math.sqrt(1.0 / (1.0 / first**2 + 1.0 / second**2))


def _point_in_polygon(x: float, y: float, vertices: tuple[tuple[float, float], ...]) -> bool:
    if len(vertices) < 3:
        return False
    inside = False
    previous = len(vertices) - 1
    for current, (x_i, y_i) in enumerate(vertices):
        x_j, y_j = vertices[previous]
        if (y_i > y) != (y_j > y):
            x_crossing = (x_j - x_i) * (y - y_i) / (y_j - y_i) + x_i
            if x < x_crossing:
                inside = not inside
        previous = current
    return inside


def _clamp01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return min(max(float(value), 0.0), 1.0)


__all__ = [
    "AgeCategory",
    "BreathingPattern",
    "BreathingType",
    "ConfidenceScore",
    "Coordinates3D",
    "DebrisMaterial",
    "DebrisProfile",
    "DepthEstimate",
    "DisasterEvent",
    "DisasterType",
    "EventMetadata",
    "EventStatus",
    "HeartbeatSignature",
    "LocationUncertainty",
    "MetalContent",
    "MoistureLevel",
    "MovementActivity",
    "MovementProfile",
    "MovementType",
    "ScanParameters",
    "ScanResolution",
    "ScanZone",
    "SensorPosition",
    "SensorType",
    "SignalStrength",
    "Survivor",
    "SurvivorCondition",
    "SurvivorCounts",
    "SurvivorMetadata",
    "SurvivorStatus",
    "VitalSignsHistory",
    "VitalSignsReading",
    "ZoneBounds",
    "ZoneShape",
    "ZoneStatus",
    "condition_from_vitals",
]
