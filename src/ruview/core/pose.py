"""Pose contracts ported from ``wifi-densepose-core``."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from ruview.core.confidence import Confidence
from ruview.core.errors import ValidationError
from ruview.core.frames import FrameId, Timestamp

MAX_KEYPOINTS = 17


class KeypointType(IntEnum):
    """COCO-format keypoint index."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    @classmethod
    def all(cls) -> tuple["KeypointType", ...]:
        return tuple(cls)

    @classmethod
    def from_index(cls, index: int) -> "KeypointType":
        try:
            return cls(index)
        except ValueError as exc:
            raise ValidationError(f"Invalid keypoint type: {index}") from exc

    @property
    def label(self) -> str:
        return _KEYPOINT_LABELS[int(self)]

    def is_face(self) -> bool:
        return self in {
            KeypointType.NOSE,
            KeypointType.LEFT_EYE,
            KeypointType.RIGHT_EYE,
            KeypointType.LEFT_EAR,
            KeypointType.RIGHT_EAR,
        }

    def is_upper_body(self) -> bool:
        return self in {
            KeypointType.LEFT_SHOULDER,
            KeypointType.RIGHT_SHOULDER,
            KeypointType.LEFT_ELBOW,
            KeypointType.RIGHT_ELBOW,
            KeypointType.LEFT_WRIST,
            KeypointType.RIGHT_WRIST,
        }

    def is_lower_body(self) -> bool:
        return self in {
            KeypointType.LEFT_HIP,
            KeypointType.RIGHT_HIP,
            KeypointType.LEFT_KNEE,
            KeypointType.RIGHT_KNEE,
            KeypointType.LEFT_ANKLE,
            KeypointType.RIGHT_ANKLE,
        }


_KEYPOINT_LABELS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class Keypoint:
    """Single body keypoint with optional depth and confidence."""

    keypoint_type: KeypointType
    x: float
    y: float
    confidence: Confidence
    z: float | None = None

    @classmethod
    def new(cls, keypoint_type: KeypointType, x: float, y: float, confidence: Confidence) -> "Keypoint":
        return cls(keypoint_type, x, y, confidence)

    @classmethod
    def new_3d(
        cls, keypoint_type: KeypointType, x: float, y: float, z: float, confidence: Confidence
    ) -> "Keypoint":
        return cls(keypoint_type, x, y, confidence, z)

    def is_visible(self) -> bool:
        return self.confidence.is_high()

    def position_2d(self) -> tuple[float, float]:
        return (self.x, self.y)

    def position_3d(self) -> tuple[float, float, float] | None:
        return None if self.z is None else (self.x, self.y, self.z)

    def distance_to(self, other: "Keypoint") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        if self.z is not None and other.z is not None:
            return math.sqrt(dx * dx + dy * dy + (self.z - other.z) ** 2)
        return math.hypot(dx, dy)


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @classmethod
    def from_center(cls, cx: float, cy: float, width: float, height: float) -> "BoundingBox":
        return cls(cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)

    def width(self) -> float:
        return self.x_max - self.x_min

    def height(self) -> float:
        return self.y_max - self.y_min

    def area(self) -> float:
        return self.width() * self.height()

    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def iou(self, other: "BoundingBox") -> float:
        x_min = max(self.x_min, other.x_min)
        y_min = max(self.y_min, other.y_min)
        x_max = min(self.x_max, other.x_max)
        y_max = min(self.y_max, other.y_max)
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        intersection = (x_max - x_min) * (y_max - y_min)
        union = self.area() + other.area() - intersection
        return 0.0 if union <= 0.0 else intersection / union


@dataclass
class PersonPose:
    """Pose estimate for one person."""

    id: int | None = None
    keypoints: list[Keypoint | None] = field(default_factory=lambda: [None] * MAX_KEYPOINTS)
    bounding_box: BoundingBox | None = None
    confidence: Confidence = field(default_factory=lambda: Confidence.MIN)

    def __post_init__(self) -> None:
        if len(self.keypoints) != MAX_KEYPOINTS:
            raise ValidationError(f"PersonPose requires {MAX_KEYPOINTS} keypoint slots")

    def set_keypoint(self, keypoint: Keypoint) -> None:
        self.keypoints[int(keypoint.keypoint_type)] = keypoint

    def get_keypoint(self, keypoint_type: KeypointType) -> Keypoint | None:
        return self.keypoints[int(keypoint_type)]

    def visible_keypoint_count(self) -> int:
        return sum(1 for keypoint in self.keypoints if keypoint is not None and keypoint.is_visible())

    def visible_keypoints(self) -> list[Keypoint]:
        return [keypoint for keypoint in self.keypoints if keypoint is not None and keypoint.is_visible()]

    def compute_bounding_box(self) -> BoundingBox | None:
        visible = self.visible_keypoints()
        if not visible:
            return None
        return BoundingBox(
            min(keypoint.x for keypoint in visible),
            min(keypoint.y for keypoint in visible),
            max(keypoint.x for keypoint in visible),
            max(keypoint.y for keypoint in visible),
        )

    def to_flat_array(self) -> NDArray[np.float32]:
        values = np.zeros(MAX_KEYPOINTS * 3, dtype=np.float32)
        for index, keypoint in enumerate(self.keypoints):
            if keypoint is None:
                continue
            values[index * 3] = keypoint.x
            values[index * 3 + 1] = keypoint.y
            values[index * 3 + 2] = keypoint.confidence.value
        return values


@dataclass
class PoseEstimate:
    """Complete pose-estimation result for one inference frame."""

    source_signal_ids: list[FrameId]
    persons: list[PersonPose]
    confidence: Confidence
    latency_ms: float
    model_version: str
    id: FrameId = field(default_factory=FrameId.new)
    timestamp: Timestamp = field(default_factory=Timestamp.now)

    def person_count(self) -> int:
        return len(self.persons)

    def has_detections(self) -> bool:
        return bool(self.persons)

    def highest_confidence_person(self) -> PersonPose | None:
        if not self.persons:
            return None
        return max(self.persons, key=lambda person: person.confidence.value)

