"""Deterministic synthetic CSI generation for examples and tests."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ruview.core import (
    AntennaConfig,
    CsiFrame,
    CsiMetadata,
    FrameId,
    FrequencyBand,
    Timestamp,
    ValidationError,
)

SyntheticCsiScenario = Literal["empty_room", "person_present", "walking", "stillness"]

SCENARIOS: tuple[SyntheticCsiScenario, ...] = (
    "empty_room",
    "person_present",
    "walking",
    "stillness",
)

_SCENARIO_ALIASES: dict[str, SyntheticCsiScenario] = {
    "empty": "empty_room",
    "empty_room": "empty_room",
    "empty-room": "empty_room",
    "empty room": "empty_room",
    "person": "person_present",
    "present": "person_present",
    "person_present": "person_present",
    "person-present": "person_present",
    "person present": "person_present",
    "walking": "walking",
    "motion": "walking",
    "moving": "walking",
    "still": "stillness",
    "stillness": "stillness",
}

_FIXTURE_SCHEMA = "ruview.synthetic_csi.v1"


@dataclass(frozen=True)
class SyntheticCsiConfig:
    """Configuration for synthetic CSI windows.

    The generated arrays are shaped ``[frames, streams, subcarriers]`` for
    windows and ``[streams, subcarriers]`` for individual ``CsiFrame`` objects.
    """

    seed: int = 42
    streams: int = 3
    subcarriers: int = 56
    frames: int = 128
    base_amplitude: float = 1.0
    noise_std: float = 0.02
    phase_noise_std: float = 0.01
    sample_rate_hz: float = 20.0
    person_amplitude_scale: float = 1.24
    presence_phase_offset: float = 0.22
    motion_amplitude: float = 0.18
    motion_frequency_hz: float = 0.9
    stillness_amplitude: float = 0.025
    stillness_frequency_hz: float = 0.25
    device_id: str = "synthetic-node-1"
    frequency_band: FrequencyBand | str | int = FrequencyBand.BAND_5_GHZ
    channel: int = 36
    bandwidth_mhz: int = 20
    rssi_dbm: int = -50
    noise_floor_dbm: int = -90
    start_time_seconds: int = 1_700_000_000

    def __post_init__(self) -> None:
        if self.streams <= 0:
            raise ValidationError("streams must be positive")
        if self.streams > 255:
            raise ValidationError("streams must fit in the metadata antenna layout")
        if self.subcarriers <= 0:
            raise ValidationError("subcarriers must be positive")
        if self.frames <= 0:
            raise ValidationError("frames must be positive")
        if self.base_amplitude <= 0:
            raise ValidationError("base_amplitude must be positive")
        if self.noise_std < 0:
            raise ValidationError("noise_std must be non-negative")
        if self.phase_noise_std < 0:
            raise ValidationError("phase_noise_std must be non-negative")
        if self.sample_rate_hz <= 0:
            raise ValidationError("sample_rate_hz must be positive")
        if self.motion_frequency_hz < 0:
            raise ValidationError("motion_frequency_hz must be non-negative")
        if self.stillness_frequency_hz < 0:
            raise ValidationError("stillness_frequency_hz must be non-negative")
        object.__setattr__(
            self,
            "frequency_band",
            FrequencyBand.from_value(self.frequency_band),
        )


@dataclass(frozen=True)
class SyntheticCsiFixture:
    """Loaded synthetic CSI fixture data and JSON-style metadata."""

    frames: list[CsiFrame]
    metadata: dict[str, object]

    def window(self) -> NDArray[np.complex128]:
        return np.stack([frame.data for frame in self.frames], axis=0)


def normalize_scenario(scenario: str) -> SyntheticCsiScenario:
    """Return the canonical scenario name for a user-facing alias."""

    key = scenario.strip().lower().replace("_", " ")
    normalized = _SCENARIO_ALIASES.get(key) or _SCENARIO_ALIASES.get(
        key.replace(" ", "_"),
    )
    if normalized is None:
        valid = ", ".join(SCENARIOS)
        raise ValidationError(
            f"unknown synthetic CSI scenario {scenario!r}; expected one of {valid}",
        )
    return normalized


def generate_synthetic_frame(
    scenario: str,
    config: SyntheticCsiConfig | None = None,
    *,
    frame_index: int = 0,
) -> CsiFrame:
    """Generate one deterministic synthetic CSI frame."""

    config = config or SyntheticCsiConfig()
    if frame_index < 0:
        raise ValidationError("frame_index must be non-negative")

    scenario = normalize_scenario(scenario)
    data = _generate_frame_data(scenario, config, frame_index)
    metadata = _metadata_for_frame(scenario, config, frame_index)
    frame_id = _frame_id(scenario, config, frame_index)
    return CsiFrame(metadata, data, id=frame_id)


def generate_synthetic_sequence(
    scenario: str,
    config: SyntheticCsiConfig | None = None,
) -> list[CsiFrame]:
    """Generate ``config.frames`` synthetic ``CsiFrame`` objects."""

    config = config or SyntheticCsiConfig()
    scenario = normalize_scenario(scenario)
    return [
        generate_synthetic_frame(scenario, config, frame_index=index)
        for index in range(config.frames)
    ]


def generate_synthetic_window(
    scenario: str,
    config: SyntheticCsiConfig | None = None,
) -> NDArray[np.complex128]:
    """Generate a complex CSI window shaped ``[frames, streams, subcarriers]``."""

    frames = generate_synthetic_sequence(scenario, config)
    return np.stack([frame.data for frame in frames], axis=0)


def save_synthetic_fixture(
    path: str | Path,
    frames: Sequence[CsiFrame],
    *,
    scenario: str,
    config: SyntheticCsiConfig,
) -> Path:
    """Save synthetic frames to a compact ``.npz`` fixture."""

    if not frames:
        raise ValidationError("cannot save an empty synthetic CSI fixture")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scenario = normalize_scenario(scenario)
    window = np.stack([frame.data for frame in frames], axis=0)
    metadata = {
        "schema": _FIXTURE_SCHEMA,
        "scenario": scenario,
        "config": _config_to_json_dict(config),
        "shape": list(window.shape),
        "frame_ids": [str(frame.id) for frame in frames],
        "sequence_numbers": [int(frame.metadata.sequence_number) for frame in frames],
        "timestamps": [
            {
                "seconds": int(frame.metadata.timestamp.seconds),
                "nanos": int(frame.metadata.timestamp.nanos),
            }
            for frame in frames
        ],
    }
    np.savez_compressed(path, csi=window, metadata=json.dumps(metadata, sort_keys=True))
    return path


def load_synthetic_fixture(path: str | Path) -> SyntheticCsiFixture:
    """Load frames previously saved by :func:`save_synthetic_fixture`."""

    with np.load(Path(path), allow_pickle=False) as archive:
        window = np.asarray(archive["csi"], dtype=np.complex128)
        metadata = json.loads(str(archive["metadata"].item()))

    if metadata.get("schema") != _FIXTURE_SCHEMA:
        raise ValidationError(
            f"unsupported synthetic CSI fixture schema: {metadata.get('schema')!r}",
        )
    if window.ndim != 3:
        raise ValidationError(f"fixture CSI array must be 3D, got shape {window.shape}")

    config = SyntheticCsiConfig(**metadata["config"])
    config = replace(
        config,
        frames=int(window.shape[0]),
        streams=int(window.shape[1]),
        subcarriers=int(window.shape[2]),
    )
    scenario = normalize_scenario(str(metadata["scenario"]))
    frame_ids = [FrameId.from_uuid(value) for value in metadata.get("frame_ids", [])]
    sequence_numbers = [int(value) for value in metadata.get("sequence_numbers", [])]
    timestamps = metadata.get("timestamps", [])

    frames: list[CsiFrame] = []
    for index, data in enumerate(window):
        metadata_for_frame = _metadata_for_frame(scenario, config, index)
        if index < len(sequence_numbers):
            metadata_for_frame.sequence_number = sequence_numbers[index]
        if index < len(timestamps):
            timestamp = timestamps[index]
            metadata_for_frame.timestamp = Timestamp(
                int(timestamp["seconds"]),
                int(timestamp["nanos"]),
            )
        frame_id = (
            frame_ids[index]
            if index < len(frame_ids)
            else _frame_id(scenario, config, index)
        )
        frames.append(CsiFrame(metadata_for_frame, data, id=frame_id))

    return SyntheticCsiFixture(frames=frames, metadata=metadata)


def _generate_frame_data(
    scenario: SyntheticCsiScenario,
    config: SyntheticCsiConfig,
    frame_index: int,
) -> NDArray[np.complex128]:
    base_amplitude, base_phase = _base_channel(config)
    amplitude_scale, phase_shift = _scenario_perturbation(scenario, config, frame_index)
    amplitude = np.maximum(base_amplitude * amplitude_scale, 1e-6)
    phase = base_phase + phase_shift

    rng = np.random.default_rng(_seed_for(config.seed, scenario, frame_index))
    if config.phase_noise_std:
        phase = phase + rng.normal(0.0, config.phase_noise_std, phase.shape)

    data = amplitude * np.exp(1j * phase)
    if config.noise_std:
        noise = rng.normal(0.0, config.noise_std, data.shape) + 1j * rng.normal(
            0.0,
            config.noise_std,
            data.shape,
        )
        data = data + noise
    return np.asarray(data, dtype=np.complex128)


def _base_channel(
    config: SyntheticCsiConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(_seed_for(config.seed, "base", 0))
    subcarrier = np.arange(config.subcarriers, dtype=np.float64)[None, :]
    stream = np.arange(config.streams, dtype=np.float64)[:, None]
    denominator = max(config.subcarriers - 1, 1)
    normalized_subcarrier = subcarrier / denominator

    smooth_ripple = (
        1.0
        + 0.10 * np.sin(2.0 * np.pi * normalized_subcarrier)
        + 0.04 * np.cos(6.0 * np.pi * normalized_subcarrier)
    )
    stream_gain = 1.0 + 0.04 * (stream - (config.streams - 1) / 2.0)
    random_gain = rng.normal(0.0, 0.015, (config.streams, 1)) + rng.normal(
        0.0,
        0.010,
        (1, config.subcarriers),
    )
    amplitude = config.base_amplitude * np.maximum(
        smooth_ripple * stream_gain + random_gain,
        0.05,
    )

    phase = (
        0.09 * subcarrier
        + 0.45 * stream
        + 0.12 * np.cos(2.0 * np.pi * normalized_subcarrier + 0.3 * stream)
    )
    phase = phase + rng.normal(0.0, 0.03, (config.streams, 1))
    phase = phase + rng.normal(0.0, 0.01, (1, config.subcarriers))
    return np.asarray(amplitude, dtype=np.float64), np.asarray(phase, dtype=np.float64)


def _scenario_perturbation(
    scenario: SyntheticCsiScenario,
    config: SyntheticCsiConfig,
    frame_index: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    subcarrier = np.arange(config.subcarriers, dtype=np.float64)[None, :]
    stream = np.arange(config.streams, dtype=np.float64)[:, None]
    denominator = max(config.subcarriers - 1, 1)
    normalized_subcarrier = subcarrier / denominator
    reflector_lobe = np.exp(-0.5 * ((normalized_subcarrier - 0.58) / 0.18) ** 2)
    stream_lobe = 1.0 + 0.08 * np.sin(stream + 0.7)
    t = frame_index / config.sample_rate_hz

    amplitude_scale = np.ones((config.streams, config.subcarriers), dtype=np.float64)
    phase_shift = np.zeros_like(amplitude_scale)

    if scenario == "empty_room":
        drift = 0.004 * np.sin(2.0 * np.pi * 0.03 * t + normalized_subcarrier)
        amplitude_scale = amplitude_scale + drift
        phase_shift = phase_shift + 0.003 * drift
        return amplitude_scale, phase_shift

    presence = 0.09 * reflector_lobe * stream_lobe
    amplitude_scale = amplitude_scale * config.person_amplitude_scale + presence
    phase_shift = phase_shift + config.presence_phase_offset * (0.55 + reflector_lobe)

    if scenario == "person_present":
        return amplitude_scale, phase_shift

    if scenario == "stillness":
        breathing = np.sin(
            2.0 * np.pi * config.stillness_frequency_hz * t + 0.4 * stream,
        )
        local_breathing = config.stillness_amplitude * breathing * (0.35 + reflector_lobe)
        amplitude_scale = amplitude_scale + local_breathing
        phase_shift = phase_shift + 0.20 * local_breathing
        return amplitude_scale, phase_shift

    stride = np.sin(
        2.0 * np.pi * config.motion_frequency_hz * t
        + 0.35 * stream
        + 2.0 * np.pi * normalized_subcarrier,
    )
    harmonic = 0.35 * np.sin(
        4.0 * np.pi * config.motion_frequency_hz * t
        + 0.25 * stream
        - np.pi * normalized_subcarrier,
    )
    motion = config.motion_amplitude * (stride + harmonic) * (0.45 + reflector_lobe)
    amplitude_scale = amplitude_scale + motion
    phase_shift = phase_shift + 0.45 * motion
    return amplitude_scale, phase_shift


def _metadata_for_frame(
    scenario: SyntheticCsiScenario,
    config: SyntheticCsiConfig,
    frame_index: int,
) -> CsiMetadata:
    timestamp = _timestamp_for_index(config, frame_index)
    rssi_offset = {
        "empty_room": -4,
        "person_present": 0,
        "stillness": -1,
        "walking": 1,
    }[scenario]
    return CsiMetadata(
        config.device_id,
        config.frequency_band,
        channel=config.channel,
        timestamp=timestamp,
        bandwidth_mhz=config.bandwidth_mhz,
        antenna_config=AntennaConfig(1, config.streams),
        rssi_dbm=config.rssi_dbm + rssi_offset,
        noise_floor_dbm=config.noise_floor_dbm,
        sequence_number=frame_index,
    )


def _timestamp_for_index(config: SyntheticCsiConfig, frame_index: int) -> Timestamp:
    offset_nanos = int(round(frame_index * 1_000_000_000 / config.sample_rate_hz))
    return Timestamp(
        config.start_time_seconds + offset_nanos // 1_000_000_000,
        offset_nanos % 1_000_000_000,
    )


def _frame_id(
    scenario: SyntheticCsiScenario,
    config: SyntheticCsiConfig,
    frame_index: int,
) -> FrameId:
    namespace = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"ruview.synthetic_csi:{scenario}:{_config_cache_key(config)}",
    )
    return FrameId.from_uuid(uuid.uuid5(namespace, str(frame_index)))


def _seed_for(seed: int, scenario: str, frame_index: int) -> int:
    payload = f"{seed}:{scenario}:{frame_index}".encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _config_to_json_dict(config: SyntheticCsiConfig) -> dict[str, object]:
    result = asdict(config)
    result["frequency_band"] = int(config.frequency_band)
    return result


def _config_cache_key(config: SyntheticCsiConfig) -> str:
    return json.dumps(_config_to_json_dict(config), sort_keys=True, separators=(",", ":"))
