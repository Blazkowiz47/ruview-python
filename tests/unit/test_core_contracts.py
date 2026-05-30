from __future__ import annotations

import math
import struct
import uuid

import numpy as np
import pytest

from ruview.core import (
    AntennaConfig,
    BoundingBox,
    ComplexSample,
    Confidence,
    CsiFrame,
    CsiMetadata,
    DeviceId,
    FrameId,
    FrequencyBand,
    Keypoint,
    KeypointType,
    PersonPose,
    PoseEstimate,
    Timestamp,
    ValidationError,
)


def test_confidence_validation_and_thresholds() -> None:
    assert Confidence(0.0).value == 0.0
    assert Confidence(1.0).value == 1.0
    assert Confidence(0.8).is_high()
    assert not Confidence(0.3).is_high()
    assert Confidence(0.7).exceeds(0.7)

    for value in (-0.1, 1.1, math.nan, math.inf):
        with pytest.raises(ValidationError):
            Confidence(value)


def test_complex_sample_little_endian_roundtrip() -> None:
    sample = ComplexSample(1.5, -2.25)
    assert sample.to_le_bytes() == struct.pack("<dd", 1.5, -2.25)
    assert ComplexSample.from_le_bytes(sample.to_le_bytes()) == sample
    assert np.isclose(sample.norm(), abs(1.5 - 2.25j))
    assert np.isclose(sample.arg(), np.angle(1.5 - 2.25j))


def test_timestamp_duration_and_datetime_roundtrip() -> None:
    earlier = Timestamp(100, 0)
    later = Timestamp(101, 500_000_000)
    assert later.duration_since(earlier) == 1.5
    assert Timestamp.from_datetime(later.to_datetime()) == later


def test_metadata_defaults_and_provenance_setters() -> None:
    metadata = CsiMetadata(DeviceId("esp32-s3-com9"), FrequencyBand.BAND_2_4_GHZ, 6)
    assert metadata.bandwidth_mhz == 20
    assert metadata.antenna_config == AntennaConfig.simo_1x3()
    assert metadata.snr_db() == 40.0
    assert metadata.calibration_id is None
    assert metadata.model_id == 0
    assert metadata.model_version == 0

    calibration_id = uuid.uuid4()
    metadata.set_calibration(calibration_id)
    metadata.set_model(7, 0x0102)
    assert metadata.calibration_id == calibration_id
    assert metadata.model_id == 7
    assert metadata.model_version == 0x0102


def test_csi_frame_derives_amplitude_phase_and_shape() -> None:
    metadata = CsiMetadata("node-1", "band_5_ghz", 36)
    data = np.array([[1 + 0j, 0 + 1j], [3 + 4j, -1 - 1j]], dtype=np.complex128)
    frame = CsiFrame(metadata, data)

    assert frame.num_spatial_streams() == 2
    assert frame.num_subcarriers() == 2
    np.testing.assert_allclose(frame.amplitude, np.abs(data))
    np.testing.assert_allclose(frame.phase, np.angle(data))
    assert np.isclose(frame.mean_amplitude(), np.mean(np.abs(data)))


def test_csi_frame_canonical_bytes_are_deterministic_and_provenance_sensitive() -> None:
    metadata = CsiMetadata(
        "node-1",
        FrequencyBand.BAND_5_GHZ,
        36,
        timestamp=Timestamp(1_700_000_000, 123),
        sequence_number=99,
    )
    data = np.array([[0.5 + 0j, 1.0 + 0.25j], [1.5 - 0.5j, 2.0 + 0.75j]], dtype=np.complex128)
    frame = CsiFrame(metadata, data, id=FrameId.from_uuid("00000000-0000-0000-0000-000000000001"))

    canonical = frame.to_canonical_bytes()
    assert canonical == frame.to_canonical_bytes()
    assert frame.witness_hash() == frame.witness_hash()
    assert canonical[:16] == uuid.UUID("00000000-0000-0000-0000-000000000001").bytes
    assert canonical[16:24] == struct.pack("<q", 1_700_000_000)
    assert canonical[-64:-48] == ComplexSample(0.5, 0.0).to_le_bytes()

    changed = CsiFrame(metadata, data, id=frame.id)
    changed.metadata.set_model(1, 1)
    assert frame.witness_hash() != changed.witness_hash()


def test_pose_keypoints_bounding_box_and_flat_array() -> None:
    pose = PersonPose(confidence=Confidence(0.8))
    pose.set_keypoint(Keypoint(KeypointType.NOSE, 0.2, 0.3, Confidence(0.95)))
    pose.set_keypoint(Keypoint(KeypointType.LEFT_SHOULDER, 0.1, 0.6, Confidence(0.8)))
    pose.set_keypoint(Keypoint(KeypointType.RIGHT_ANKLE, 0.9, 0.9, Confidence(0.2)))

    assert pose.get_keypoint(KeypointType.NOSE) is not None
    assert pose.visible_keypoint_count() == 2
    assert pose.compute_bounding_box() == BoundingBox(0.1, 0.3, 0.2, 0.6)
    flat = pose.to_flat_array()
    assert flat.shape == (51,)
    assert np.isclose(flat[int(KeypointType.NOSE) * 3 + 2], 0.95)


def test_pose_estimate_helpers() -> None:
    low = PersonPose(confidence=Confidence(0.2))
    high = PersonPose(confidence=Confidence(0.9))
    estimate = PoseEstimate([FrameId.new()], [low, high], Confidence(0.7), 12.5, "test-model")

    assert estimate.person_count() == 2
    assert estimate.has_detections()
    assert estimate.highest_confidence_person() is high

