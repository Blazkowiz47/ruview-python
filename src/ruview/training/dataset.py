"""CSI dataset and batching utilities for training research."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ruview.nn.tensor import Tensor
from ruview.signal import extract_window_features
from ruview.training.config import TrainingConfig


FloatArray: TypeAlias = NDArray[np.float32]
JsonRecord: TypeAlias = Mapping[str, Any]

SIGNAL_FEATURE_NAMES: tuple[str, ...] = (
    "mean_amplitude",
    "amplitude_variance",
    "phase_variance",
    "motion_energy",
)


@runtime_checkable
class CsiDataset(Protocol):
    """Minimal dataset protocol used by :class:`DataLoader`."""

    def __len__(self) -> int:
        ...

    def get(self, index: int) -> "CsiSample":
        ...


@dataclass(frozen=True)
class CsiSample:
    """A windowed CSI sample with pose labels.

    Arrays are row-major NumPy tensors:

    - ``amplitude``: ``[T, tx, rx, subcarrier]``
    - ``phase``: ``[T, tx, rx, subcarrier]``
    - ``keypoints``: ``[J, 2]`` normalized ``x, y`` coordinates
    - ``visibility``: ``[J]`` COCO-style visibility values
    """

    amplitude: ArrayLike
    phase: ArrayLike
    keypoints: ArrayLike
    visibility: ArrayLike
    subject_id: int = 0
    action_id: int = 0
    frame_id: int = 0
    sample_id: str | None = None
    recording_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        amplitude = _float32_array(self.amplitude, "amplitude")
        phase = _float32_array(self.phase, "phase")
        keypoints = _float32_array(self.keypoints, "keypoints")
        visibility = _float32_array(self.visibility, "visibility").reshape(-1)

        if amplitude.ndim != 4:
            raise ValueError(f"amplitude must be 4-D [T, tx, rx, sc], got {amplitude.shape}")
        if phase.shape != amplitude.shape:
            raise ValueError(f"phase shape must match amplitude, got {phase.shape} and {amplitude.shape}")
        if keypoints.ndim != 2 or keypoints.shape[1] != 2:
            raise ValueError(f"keypoints must have shape [J, 2], got {keypoints.shape}")
        if visibility.shape != (keypoints.shape[0],):
            raise ValueError(
                f"visibility must have shape [{keypoints.shape[0]}], got {visibility.shape}"
            )

        subject_id = int(self.subject_id)
        action_id = int(self.action_id)
        frame_id = int(self.frame_id)
        recording_id = None if self.recording_id is None else str(self.recording_id)
        sample_id = self.sample_id
        if sample_id is None:
            prefix = recording_id or "sample"
            sample_id = f"{prefix}:{subject_id}:{action_id}:{frame_id}"

        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "keypoints", keypoints)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "sample_id", str(sample_id))
        object.__setattr__(self, "recording_id", recording_id)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def keypoint_visibility(self) -> FloatArray:
        """Alias matching the Rust field name."""

        return self.visibility

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Amplitude and phase tensor shape."""

        return tuple(int(dim) for dim in self.amplitude.shape)

    @property
    def num_keypoints(self) -> int:
        """Number of keypoints in the label."""

        return int(self.keypoints.shape[0])

    def signal_features(self) -> FloatArray:
        """Return compact scalar signal features derived from amplitude/phase."""

        features = extract_window_features(self.amplitude, self.phase, time_axis=0)
        return np.asarray(
            [
                features.mean_amplitude,
                features.amplitude_variance,
                features.phase_variance,
                features.motion_energy,
            ],
            dtype=np.float32,
        )

    def as_arrays(self) -> dict[str, FloatArray]:
        """Return model-ready NumPy arrays."""

        return {
            "amplitude": self.amplitude,
            "phase": self.phase,
            "keypoints": self.keypoints,
            "visibility": self.visibility,
            "signal_features": self.signal_features(),
        }

    def as_tensors(self) -> dict[str, Tensor]:
        """Return lightweight NumPy-backed tensors."""

        return {name: Tensor(value, dtype=np.float32) for name, value in self.as_arrays().items()}

    def to_torch(self, *, device: str | None = None) -> dict[str, Any]:
        """Return a dict of PyTorch tensors, importing torch only on demand."""

        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is optional; install ruview-python[nn] to use to_torch") from exc

        tensors = {name: torch.from_numpy(value) for name, value in self.as_arrays().items()}
        return tensors if device is None else {name: value.to(device) for name, value in tensors.items()}


@dataclass(frozen=True, init=False)
class CsiBatch:
    """A batch of CSI samples stacked into model-ready arrays."""

    samples: tuple[CsiSample, ...]
    amplitude: FloatArray
    phase: FloatArray
    keypoints: FloatArray
    visibility: FloatArray
    signal_features: FloatArray
    frame_ids: NDArray[np.int64]
    subject_ids: NDArray[np.int64]
    action_ids: NDArray[np.int64]
    sample_ids: tuple[str, ...]

    def __init__(self, samples: Sequence[CsiSample]) -> None:
        sample_tuple = tuple(samples)
        if not sample_tuple:
            raise ValueError("CsiBatch requires at least one sample")

        object.__setattr__(self, "samples", sample_tuple)
        object.__setattr__(self, "amplitude", np.stack([s.amplitude for s in sample_tuple]).astype(np.float32, copy=False))
        object.__setattr__(self, "phase", np.stack([s.phase for s in sample_tuple]).astype(np.float32, copy=False))
        object.__setattr__(self, "keypoints", np.stack([s.keypoints for s in sample_tuple]).astype(np.float32, copy=False))
        object.__setattr__(self, "visibility", np.stack([s.visibility for s in sample_tuple]).astype(np.float32, copy=False))
        object.__setattr__(
            self,
            "signal_features",
            np.stack([s.signal_features() for s in sample_tuple]).astype(np.float32, copy=False),
        )
        object.__setattr__(self, "frame_ids", np.asarray([s.frame_id for s in sample_tuple], dtype=np.int64))
        object.__setattr__(self, "subject_ids", np.asarray([s.subject_id for s in sample_tuple], dtype=np.int64))
        object.__setattr__(self, "action_ids", np.asarray([s.action_id for s in sample_tuple], dtype=np.int64))
        object.__setattr__(self, "sample_ids", tuple(s.sample_id for s in sample_tuple))

    def __len__(self) -> int:
        return len(self.samples)

    def as_arrays(self) -> dict[str, NDArray[Any]]:
        """Return batch arrays keyed by model input/target name."""

        return {
            "amplitude": self.amplitude,
            "phase": self.phase,
            "keypoints": self.keypoints,
            "visibility": self.visibility,
            "signal_features": self.signal_features,
            "frame_ids": self.frame_ids,
            "subject_ids": self.subject_ids,
            "action_ids": self.action_ids,
        }

    def to_torch(self, *, device: str | None = None) -> dict[str, Any]:
        """Return a dict of PyTorch tensors, importing torch only on demand."""

        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is optional; install ruview-python[nn] to use to_torch") from exc

        tensors = {name: torch.from_numpy(value) for name, value in self.as_arrays().items()}
        return tensors if device is None else {name: value.to(device) for name, value in tensors.items()}


@dataclass(frozen=True)
class SyntheticCsiConfig:
    """Configuration for deterministic synthetic CSI samples."""

    num_subcarriers: int = 56
    num_antennas_tx: int = 3
    num_antennas_rx: int = 3
    window_frames: int = 100
    num_keypoints: int = 17
    signal_frequency_hz: float = 2.4e9

    def __post_init__(self) -> None:
        for name in (
            "num_subcarriers",
            "num_antennas_tx",
            "num_antennas_rx",
            "window_frames",
            "num_keypoints",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
            object.__setattr__(self, name, value)
        if self.signal_frequency_hz <= 0.0:
            raise ValueError("signal_frequency_hz must be > 0.0")
        object.__setattr__(self, "signal_frequency_hz", float(self.signal_frequency_hz))

    @classmethod
    def from_training_config(cls, config: TrainingConfig) -> "SyntheticCsiConfig":
        """Build a synthetic config from training data dimensions."""

        return cls(
            num_subcarriers=config.num_subcarriers,
            num_antennas_tx=config.num_antennas_tx,
            num_antennas_rx=config.num_antennas_rx,
            window_frames=config.window_frames,
            num_keypoints=config.num_keypoints,
        )


class SyntheticCsiDataset:
    """Fully deterministic CSI dataset generated from analytic formulas."""

    def __init__(
        self,
        num_samples: int,
        config: SyntheticCsiConfig | TrainingConfig | None = None,
        *,
        subject_id: int = 0,
        action_id: int = 0,
        start_frame_id: int = 0,
    ) -> None:
        if num_samples < 0:
            raise ValueError("num_samples must be >= 0")
        if config is None:
            synthetic_config = SyntheticCsiConfig()
        elif isinstance(config, TrainingConfig):
            synthetic_config = SyntheticCsiConfig.from_training_config(config)
        else:
            synthetic_config = config

        self._num_samples = int(num_samples)
        self.config = synthetic_config
        self.subject_id = int(subject_id)
        self.action_id = int(action_id)
        self.start_frame_id = int(start_frame_id)

    @property
    def name(self) -> str:
        return "SyntheticCsiDataset"

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> CsiSample:
        return self.get(index)

    def is_empty(self) -> bool:
        return len(self) == 0

    def get(self, index: int) -> CsiSample:
        """Return the deterministic sample at ``index``."""

        if index < 0 or index >= self._num_samples:
            raise IndexError(f"sample index {index} out of range for dataset of length {self._num_samples}")

        cfg = self.config
        t = np.arange(cfg.window_frames, dtype=np.float32)[:, None, None, None]
        tx = np.arange(cfg.num_antennas_tx, dtype=np.float32)[None, :, None, None]
        rx = np.arange(cfg.num_antennas_rx, dtype=np.float32)[None, None, :, None]
        sc = np.arange(cfg.num_subcarriers, dtype=np.float32)[None, None, None, :]

        amp_phase = 2.0 * np.pi * (index * 0.01 + t * 0.1 + sc * 0.05)
        amplitude = 0.5 + 0.3 * np.sin(amp_phase)
        amplitude = np.broadcast_to(
            amplitude,
            (cfg.window_frames, cfg.num_antennas_tx, cfg.num_antennas_rx, cfg.num_subcarriers),
        ).astype(np.float32, copy=True)

        phase = (2.0 * np.pi * sc / cfg.num_subcarriers) * (tx + 1.0) * (rx + 1.0)
        phase = np.broadcast_to(
            phase,
            (cfg.window_frames, cfg.num_antennas_tx, cfg.num_antennas_rx, cfg.num_subcarriers),
        ).astype(np.float32, copy=True)

        joints = np.arange(cfg.num_keypoints, dtype=np.float32)
        keypoints = np.column_stack(
            [
                0.5 + 0.1 * np.sin(2.0 * np.pi * index * 0.007 + joints),
                0.3 + joints * 0.04,
            ]
        ).astype(np.float32, copy=False)
        keypoints = np.clip(keypoints, 0.0, 1.0)
        visibility = np.full(cfg.num_keypoints, 2.0, dtype=np.float32)
        frame_id = self.start_frame_id + index

        return CsiSample(
            amplitude=amplitude,
            phase=phase,
            keypoints=keypoints,
            visibility=visibility,
            subject_id=self.subject_id,
            action_id=self.action_id,
            frame_id=frame_id,
            sample_id=f"synthetic:{frame_id}",
            recording_id="synthetic",
            metadata={"source": "synthetic", "signal_frequency_hz": cfg.signal_frequency_hz},
        )


class ReplayCsiDataset:
    """Small JSONL/recording dataset for replayed sensing-server style rows."""

    def __init__(
        self,
        records: str | Path | Iterable[str | JsonRecord],
        config: TrainingConfig | None = None,
        *,
        source: str = "replay",
    ) -> None:
        self.config = (config or TrainingConfig()).validate()
        loaded = _load_replay_records(records)
        if not loaded:
            raise ValueError("replay recording contains no records")
        self._samples = tuple(
            sample_from_record(record, self.config, index=index, source=source)
            for index, record in enumerate(loaded)
        )
        self.source = str(source)

    @property
    def name(self) -> str:
        return "ReplayCsiDataset"

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> CsiSample:
        return self.get(index)

    def get(self, index: int) -> CsiSample:
        if index < 0 or index >= len(self._samples):
            raise IndexError(f"sample index {index} out of range for dataset of length {len(self._samples)}")
        return self._samples[index]


JsonlCsiDataset = ReplayCsiDataset


class DataLoader:
    """Deterministic batched iterator over a CSI dataset."""

    def __init__(
        self,
        dataset: CsiDataset,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int = 42,
        stack: bool = False,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.stack = bool(stack)
        self.drop_last = bool(drop_last)

    def __iter__(self) -> Iterator[list[CsiSample] | CsiBatch]:
        return self.iter_batches()

    def __len__(self) -> int:
        return self.num_batches()

    def num_batches(self) -> int:
        """Number of complete or partial batches in one epoch."""

        n_samples = len(self.dataset)
        if n_samples == 0:
            return 0
        if self.drop_last:
            return n_samples // self.batch_size
        return math.ceil(n_samples / self.batch_size)

    def indices_for_epoch(self) -> list[int]:
        """Return deterministic indices for one epoch."""

        indices = list(range(len(self.dataset)))
        if self.shuffle:
            xorshift_shuffle(indices, self.seed)
        return indices

    def iter_batches(self) -> Iterator[list[CsiSample] | CsiBatch]:
        """Yield batches as sample lists or :class:`CsiBatch` objects."""

        indices = self.indices_for_epoch()
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            samples = [self.dataset.get(index) for index in batch_indices]
            yield CsiBatch(samples) if self.stack else samples

    def iter_samples(self) -> Iterator[CsiSample]:
        """Yield individual samples in deterministic epoch order."""

        for index in self.indices_for_epoch():
            yield self.dataset.get(index)


def xorshift_shuffle(indices: list[int], seed: int) -> None:
    """In-place Fisher-Yates shuffle with a platform-stable Xorshift64 PRNG."""

    if len(indices) <= 1:
        return

    mask = (1 << 64) - 1
    state = int(seed) & mask
    if state == 0:
        state = 0x853C49E6748FEA9B

    for i in range(len(indices) - 1, 0, -1):
        state ^= (state << 13) & mask
        state &= mask
        state ^= state >> 7
        state &= mask
        state ^= (state << 17) & mask
        state &= mask
        j = state % (i + 1)
        indices[i], indices[j] = indices[j], indices[i]


def resample_subcarriers(values: ArrayLike, target_subcarriers: int) -> FloatArray:
    """Linearly resample the last axis to ``target_subcarriers``."""

    if target_subcarriers <= 0:
        raise ValueError("target_subcarriers must be > 0")
    array = _float32_array(values, "values")
    if array.ndim == 0:
        array = array.reshape(1)
    source_subcarriers = array.shape[-1]
    if source_subcarriers == 0:
        raise ValueError("cannot resample an empty subcarrier axis")
    if source_subcarriers == target_subcarriers:
        return array.astype(np.float32, copy=True)
    if source_subcarriers == 1:
        return np.repeat(array, target_subcarriers, axis=-1).astype(np.float32, copy=False)

    old_x = np.linspace(0.0, 1.0, source_subcarriers, dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, target_subcarriers, dtype=np.float32)
    flat = array.reshape(-1, source_subcarriers)
    resampled = np.empty((flat.shape[0], target_subcarriers), dtype=np.float32)
    for row_index, row in enumerate(flat):
        resampled[row_index] = np.interp(new_x, old_x, row).astype(np.float32)
    return resampled.reshape(*array.shape[:-1], target_subcarriers)


def sample_from_record(
    record: JsonRecord,
    config: TrainingConfig | None = None,
    *,
    index: int = 0,
    source: str = "replay",
) -> CsiSample:
    """Convert one JSON-style recording row to a :class:`CsiSample`."""

    cfg = (config or TrainingConfig()).validate()
    if "record" in record and isinstance(record["record"], Mapping):
        return sample_from_record(record["record"], cfg, index=index, source=source)
    if "update" in record and isinstance(record["update"], Mapping):
        return sample_from_record(record["update"], cfg, index=index, source=source)

    amplitude, phase = _record_amplitude_phase(record, cfg)
    keypoints, visibility = _record_pose(record, cfg)
    tick = _safe_int(_first(record, "frame_id", "tick", "sequence", "sequence_number"), index)
    subject_id = _safe_int(_first(record, "subject_id", "subject"), 0)
    action_id = _safe_int(_first(record, "action_id", "action"), 0)
    recording_id = str(_first(record, "source", "recording_id") or source)

    return CsiSample(
        amplitude=amplitude,
        phase=phase,
        keypoints=keypoints,
        visibility=visibility,
        subject_id=subject_id,
        action_id=action_id,
        frame_id=tick,
        sample_id=str(_first(record, "sample_id", "id") or f"{recording_id}:{index}"),
        recording_id=recording_id,
        metadata={
            "source": recording_id,
            "timestamp": _first(record, "timestamp", "ts", "time"),
            "features": dict(record.get("features", {}) or {}),
            "classification": dict(record.get("classification", {}) or {}),
        },
    )


def _record_amplitude_phase(record: JsonRecord, config: TrainingConfig) -> tuple[FloatArray, FloatArray]:
    csi = record.get("csi")
    if csi is not None:
        csi_array = np.asarray(csi)
        if np.iscomplexobj(csi_array):
            amplitude = np.abs(csi_array).astype(np.float32)
            phase = np.angle(csi_array).astype(np.float32)
            return _fit_csi_window(amplitude, config), _fit_csi_window(phase, config)
        if csi_array.ndim > 0 and csi_array.shape[-1] == 2:
            complex_array = csi_array[..., 0].astype(np.float32) + 1j * csi_array[..., 1].astype(np.float32)
            return _fit_csi_window(np.abs(complex_array), config), _fit_csi_window(np.angle(complex_array), config)

    amplitude_value = _record_values(record, ("amplitude", "amplitudes", "subcarriers", "values"))
    if amplitude_value is None:
        signal_field = record.get("signal_field")
        if isinstance(signal_field, Mapping):
            amplitude_value = _record_values(signal_field, ("values", "amplitude", "amplitudes"))

    if amplitude_value is None:
        features = record.get("features")
        if isinstance(features, Mapping) and "mean_amplitude" in features:
            amplitude_value = [features["mean_amplitude"]]

    if amplitude_value is None:
        raise ValueError("record does not contain CSI amplitude values")

    phase_value = _record_values(record, ("phase", "phases"))
    amplitude = _fit_csi_window(amplitude_value, config)
    if phase_value is None:
        phase = np.zeros_like(amplitude, dtype=np.float32)
    else:
        phase = _fit_csi_window(phase_value, config)
    return amplitude, phase


def _record_values(record: JsonRecord, keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]

    nodes = record.get("nodes")
    if isinstance(nodes, Sequence) and not isinstance(nodes, (str, bytes, bytearray)):
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            for key in keys:
                if key in node and node[key] is not None:
                    return node[key]
    return None


def _fit_csi_window(values: ArrayLike, config: TrainingConfig) -> FloatArray:
    array = _float32_array(values, "record CSI")
    if array.ndim == 0:
        array = array.reshape(1)

    if array.ndim == 1:
        array = array.reshape(1, 1, 1, array.shape[0])
    elif array.ndim == 2:
        array = array.reshape(array.shape[0], 1, 1, array.shape[1])
    elif array.ndim == 3:
        stream_count = array.shape[1]
        if stream_count == config.num_antennas_tx * config.num_antennas_rx:
            array = array.reshape(array.shape[0], config.num_antennas_tx, config.num_antennas_rx, array.shape[2])
        else:
            array = array.reshape(array.shape[0], 1, stream_count, array.shape[2])
    elif array.ndim != 4:
        raise ValueError(f"CSI values must be 1-D to 4-D, got shape {array.shape}")

    array = resample_subcarriers(array, config.num_subcarriers)
    array = _fit_axis(array, axis=0, target=config.window_frames)
    array = _fit_axis(array, axis=1, target=config.num_antennas_tx)
    array = _fit_axis(array, axis=2, target=config.num_antennas_rx)
    return array.astype(np.float32, copy=False)


def _fit_axis(array: FloatArray, *, axis: int, target: int) -> FloatArray:
    current = array.shape[axis]
    if current == target:
        return array
    if current <= 0:
        raise ValueError("cannot fit an empty CSI axis")
    if current == 1:
        return np.repeat(array, target, axis=axis).astype(np.float32, copy=False)
    if current > target:
        slices = [slice(None)] * array.ndim
        slices[axis] = slice(current - target, current)
        return array[tuple(slices)].astype(np.float32, copy=False)

    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, target - current)
    return np.pad(array, pad_width, mode="edge").astype(np.float32, copy=False)


def _record_pose(record: JsonRecord, config: TrainingConfig) -> tuple[FloatArray, FloatArray]:
    keypoints_value = _first(record, "keypoints", "gt_keypoints")
    visibility_value = _first(record, "visibility", "keypoint_visibility")

    if keypoints_value is None:
        keypoints = np.zeros((config.num_keypoints, 2), dtype=np.float32)
        visibility = np.zeros(config.num_keypoints, dtype=np.float32)
        return keypoints, visibility

    keypoint_array = _float32_array(keypoints_value, "keypoints")
    if keypoint_array.ndim == 3:
        keypoint_array = keypoint_array[-1]
    if keypoint_array.ndim != 2 or keypoint_array.shape[1] not in (2, 3):
        raise ValueError(f"keypoints must have shape [J, 2] or [J, 3], got {keypoint_array.shape}")

    if keypoint_array.shape[1] == 3 and visibility_value is None:
        visibility = keypoint_array[:, 2]
        keypoints = keypoint_array[:, :2]
    else:
        keypoints = keypoint_array[:, :2]
        visibility = (
            np.zeros(keypoint_array.shape[0], dtype=np.float32)
            if visibility_value is None
            else _float32_array(visibility_value, "visibility").reshape(-1)
        )

    keypoints = _fit_keypoints(keypoints, config.num_keypoints)
    visibility = _fit_visibility(visibility, config.num_keypoints)
    return keypoints, visibility


def _fit_keypoints(keypoints: FloatArray, target: int) -> FloatArray:
    if keypoints.shape[0] == target:
        return keypoints.astype(np.float32, copy=False)
    output = np.zeros((target, 2), dtype=np.float32)
    count = min(target, keypoints.shape[0])
    output[:count] = keypoints[:count]
    return output


def _fit_visibility(visibility: FloatArray, target: int) -> FloatArray:
    if visibility.shape == (target,):
        return visibility.astype(np.float32, copy=False)
    output = np.zeros(target, dtype=np.float32)
    count = min(target, visibility.shape[0])
    output[:count] = visibility[:count]
    return output


def _load_replay_records(records: str | Path | Iterable[str | JsonRecord]) -> list[dict[str, Any]]:
    if isinstance(records, (str, Path)):
        lines = Path(records).read_text(encoding="utf-8").splitlines()
        return [_parse_jsonl_line(line) for line in lines if _jsonl_data_line(line)]

    loaded: list[dict[str, Any]] = []
    for item in records:
        if isinstance(item, Mapping):
            loaded.append(dict(item))
        elif _jsonl_data_line(item):
            loaded.append(_parse_jsonl_line(item))
    return loaded


def _jsonl_data_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped and not stripped.startswith("#"))


def _parse_jsonl_line(line: str) -> dict[str, Any]:
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("each replay JSONL line must decode to an object")
    return payload


def _first(mapping: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float32_array(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array


__all__ = [
    "CsiBatch",
    "CsiDataset",
    "CsiSample",
    "DataLoader",
    "JsonlCsiDataset",
    "ReplayCsiDataset",
    "SIGNAL_FEATURE_NAMES",
    "SyntheticCsiConfig",
    "SyntheticCsiDataset",
    "resample_subcarriers",
    "sample_from_record",
    "xorshift_shuffle",
]
