"""Core data contracts shared across the Python port."""

from ruview.core.canonical import CanonicalFrame, witness_hash
from ruview.core.confidence import DEFAULT_CONFIDENCE_THRESHOLD, Confidence
from ruview.core.errors import RuViewError, ValidationError
from ruview.core.frames import (
    AntennaConfig,
    ComplexSample,
    CsiFrame,
    CsiMetadata,
    DeviceId,
    FrameId,
    FrequencyBand,
    Timestamp,
)
from ruview.core.pose import (
    MAX_KEYPOINTS,
    BoundingBox,
    Keypoint,
    KeypointType,
    PersonPose,
    PoseEstimate,
)

__all__ = [
    "AntennaConfig",
    "BoundingBox",
    "CanonicalFrame",
    "ComplexSample",
    "Confidence",
    "CsiFrame",
    "CsiMetadata",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DeviceId",
    "FrameId",
    "FrequencyBand",
    "Keypoint",
    "KeypointType",
    "MAX_KEYPOINTS",
    "PersonPose",
    "PoseEstimate",
    "RuViewError",
    "Timestamp",
    "ValidationError",
    "witness_hash",
]

