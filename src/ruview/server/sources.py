"""Sensing update sources for local research servers."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ruview.core import AntennaConfig, CsiFrame, CsiMetadata, FrequencyBand
from ruview.hardware import (
    SCENARIOS,
    PacketParser,
    SyntheticCsiConfig,
    UdpPacket,
    UdpReceiver,
    generate_synthetic_sequence,
    normalize_scenario,
)
from ruview.protocols import EdgeVitalsPacket, RawCsiPacket, SyncPacket, WasmEventPacket
from ruview.signal import (
    RollingBaseline,
    calculate_motion_score,
    classify_presence,
    measure_baseline,
    stack_csi_frames,
)
from ruview.server.schemas import (
    ClassificationSummary,
    FeatureSummary,
    NodeInfo,
    SensingUpdate,
    SignalFieldSummary,
)

JsonRecord: TypeAlias = Mapping[str, Any]
ReplayInput: TypeAlias = str | Path | Iterable[str | JsonRecord]


@runtime_checkable
class SensingSource(Protocol):
    """Minimal synchronous interface shared by sensing sources."""

    source: str

    def next_update(self) -> SensingUpdate | None:
        """Return the next update, or ``None`` when no update is available."""
        ...

    def iter_updates(self, *, limit: int | None = None) -> Iterator[SensingUpdate]:
        """Yield updates until exhausted or until ``limit`` is reached."""
        ...


@dataclass
class LatestState:
    """Small in-memory latest-value plus bounded history store."""

    capacity: int = 64
    _latest: SensingUpdate | None = field(default=None, init=False, repr=False)
    _history: deque[SensingUpdate] = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self._history = deque(maxlen=int(self.capacity))

    def update(self, update: SensingUpdate | Mapping[str, Any]) -> SensingUpdate:
        coerced = update if isinstance(update, SensingUpdate) else SensingUpdate.from_dict(update)
        self._latest = coerced
        self._history.append(coerced)
        return coerced

    @property
    def latest(self) -> SensingUpdate | None:
        return self._latest

    def latest_dict(self) -> dict[str, Any] | None:
        return None if self._latest is None else self._latest.to_dict()

    @property
    def history(self) -> tuple[SensingUpdate, ...]:
        return tuple(self._history)

    def history_dicts(self) -> list[dict[str, Any]]:
        return [update.to_dict() for update in self._history]


class SimulatedSensingSource:
    """Deterministic synthetic CSI source backed by the hardware simulator."""

    def __init__(
        self,
        *,
        scenarios: Sequence[str] = SCENARIOS,
        config: SyntheticCsiConfig | None = None,
        source: str = "simulated",
        baseline_scenario: str = "empty_room",
        start_tick: int = 0,
        node_position: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        if not scenarios:
            raise ValueError("scenarios must contain at least one scenario")

        self.source = source
        self.scenarios = tuple(normalize_scenario(scenario) for scenario in scenarios)
        self.config = config or SyntheticCsiConfig()
        self.node_position = tuple(float(value) for value in node_position)
        self._tick = int(start_tick)
        self._index = 0
        self._baseline = RollingBaseline(alpha=1.0, window=8)
        baseline_frames = generate_synthetic_sequence(baseline_scenario, self.config)
        self._baseline.update(stack_csi_frames(baseline_frames))

    def next_update(self) -> SensingUpdate:
        scenario = self.scenarios[self._index % len(self.scenarios)]
        self._index += 1
        self._tick += 1
        frames = generate_synthetic_sequence(scenario, self.config)
        window = stack_csi_frames(frames)
        return sensing_update_from_window(
            window,
            source=self.source,
            tick=self._tick,
            baseline=self._baseline,
            scenario=scenario,
            node_position=self.node_position,
        )

    def iter_updates(self, *, limit: int | None = None) -> Iterator[SensingUpdate]:
        produced = 0
        while limit is None or produced < limit:
            yield self.next_update()
            produced += 1


class ReplaySensingSource:
    """Deterministic JSONL replay source for update or CSI-ish records."""

    def __init__(
        self,
        recording: ReplayInput,
        *,
        source: str = "replay",
        loop: bool = False,
        start_tick: int = 0,
        preserve_record_source: bool = False,
        preserve_record_tick: bool = True,
    ) -> None:
        self.source = source
        self.loop = bool(loop)
        self.preserve_record_source = bool(preserve_record_source)
        self.preserve_record_tick = bool(preserve_record_tick)
        self._records = _load_replay_records(recording)
        if not self._records:
            raise ValueError("replay recording contains no records")
        self._index = 0
        self._tick = int(start_tick)

    def next_update(self) -> SensingUpdate | None:
        if self._index >= len(self._records):
            if not self.loop:
                return None
            self._index = 0

        record = self._records[self._index]
        self._index += 1
        self._tick += 1
        tick = self._tick
        if self.preserve_record_tick and "tick" in record:
            tick = int(record["tick"])
            self._tick = tick

        return sensing_update_from_record(
            record,
            source=self.source,
            tick=tick,
            preserve_record_source=self.preserve_record_source,
        )

    def iter_updates(self, *, limit: int | None = None) -> Iterator[SensingUpdate]:
        produced = 0
        while limit is None or produced < limit:
            update = self.next_update()
            if update is None:
                break
            yield update
            produced += 1


class UdpSensingSource:
    """UDP-backed ESP32 source with explicit timeout and iteration limits."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5005,
        *,
        parser: PacketParser[Any] | None = None,
        timeout: float | None = 0.1,
        max_datagram_size: int = 65_535,
        reuse_address: bool = False,
        receiver: UdpReceiver[Any] | None = None,
        source: str = "esp32",
        start_tick: int = 0,
        window_size: int = 16,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        self.source = source
        self.timeout = timeout
        self._tick = int(start_tick)
        self._window_size = int(window_size)
        self._frames_by_node: dict[str, deque[CsiFrame]] = defaultdict(
            lambda: deque(maxlen=self._window_size),
        )
        self._receiver = receiver or UdpReceiver(
            host,
            port,
            parser=parser,
            timeout=timeout,
            max_datagram_size=max_datagram_size,
            reuse_address=reuse_address,
        )

    @property
    def address(self) -> tuple[str, int]:
        return self._receiver.address

    @property
    def closed(self) -> bool:
        return bool(getattr(self._receiver, "closed", False))

    def close(self) -> None:
        self._receiver.close()

    def __enter__(self) -> "UdpSensingSource":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def next_update(self) -> SensingUpdate | None:
        packet = self._receiver.receive(timeout=self.timeout)
        if packet is None:
            return None
        self._tick += 1
        return self._update_from_packet(packet, tick=self._tick)

    def iter_updates(
        self,
        *,
        limit: int | None = None,
        stop_on_timeout: bool = True,
    ) -> Iterator[SensingUpdate]:
        produced = 0
        while limit is None or produced < limit:
            update = self.next_update()
            if update is None:
                if stop_on_timeout:
                    break
                continue
            yield update
            produced += 1

    def _update_from_packet(self, packet: UdpPacket[Any], *, tick: int) -> SensingUpdate:
        parsed = packet.parsed
        if isinstance(parsed, RawCsiPacket):
            return self._raw_csi_update(parsed, packet=packet, tick=tick)
        if isinstance(parsed, EdgeVitalsPacket):
            return _edge_vitals_update(parsed, source=self.source, tick=tick)
        if isinstance(parsed, SyncPacket):
            return _sync_update(parsed, source=self.source, tick=tick)
        if isinstance(parsed, WasmEventPacket):
            return _wasm_event_update(parsed, source=self.source, tick=tick)
        if isinstance(parsed, Mapping):
            return sensing_update_from_record(
                parsed,
                source=self.source,
                tick=tick,
                preserve_record_source=False,
            )
        return _generic_packet_update(parsed, packet=packet, source=self.source, tick=tick)

    def _raw_csi_update(
        self,
        parsed: RawCsiPacket,
        *,
        packet: UdpPacket[Any],
        tick: int,
    ) -> SensingUpdate:
        frame = _raw_packet_to_frame(parsed)
        node_key = str(frame.metadata.device_id)
        history = self._frames_by_node[node_key]
        history.append(frame)
        update = sensing_update_from_window(
            stack_csi_frames(history),
            source=self.source,
            tick=tick,
            scenario=None,
        )

        payload = update.to_dict()
        payload["features"].update(
            {
                "packet_type": "raw_csi",
                "sequence_number": int(parsed.sequence_number),
                "frequency_mhz": int(parsed.frequency_mhz),
                "raw_bytes": len(packet.raw),
            }
        )
        for node in payload["nodes"]:
            node.setdefault("metadata", {})
            node["metadata"].update(
                {
                    "packet_node_id": int(parsed.node_id),
                    "noise_floor_dbm": int(parsed.noise_floor_dbm),
                    "ppdu_type": int(parsed.ppdu_type),
                    "bandwidth_40mhz": parsed.bandwidth_40mhz,
                    "sync_valid": parsed.sync_valid,
                }
            )
        return SensingUpdate.from_dict(payload)


def sensing_update_from_window(
    window: Any,
    *,
    source: str,
    tick: int,
    baseline: RollingBaseline | None = None,
    scenario: str | None = None,
    node_position: Sequence[float] = (0.0, 0.0, 0.0),
) -> SensingUpdate:
    """Convert a CSI window into the Milestone 7 update shape."""

    stats = measure_baseline(window)
    motion = calculate_motion_score(window, baseline=baseline)
    presence = classify_presence(window, baseline=baseline)
    amplitude = _mean_subcarrier_amplitude(window.amplitude)
    mean_rssi = _mean_rssi(window.frames)
    frame = window.frames[-1]
    node_id = str(frame.metadata.device_id)

    node = NodeInfo(
        node_id=node_id,
        rssi_dbm=mean_rssi,
        position=node_position,
        amplitude=amplitude,
        subcarrier_count=len(amplitude),
        metadata={
            "frequency_band": int(frame.metadata.frequency_band),
            "channel": int(frame.metadata.channel),
            "bandwidth_mhz": int(frame.metadata.bandwidth_mhz),
            "sequence_number": int(frame.metadata.sequence_number),
        },
    )
    features = FeatureSummary(
        mean_rssi=mean_rssi,
        variance=stats.amplitude_variance,
        motion_band_power=motion.temporal_delta,
        breathing_band_power=stats.phase_variance,
        dominant_freq_hz=0.0,
        change_points=_count_change_points(_temporal_mean_amplitude(window.amplitude)),
        spectral_power=stats.motion_energy,
        mean_amplitude=stats.mean_amplitude,
        temporal_delta=stats.temporal_delta,
        phase_variance=stats.phase_variance,
        subcarrier_variance=stats.subcarrier_variance,
        motion_energy=stats.motion_energy,
        motion_score=motion.total,
        frame_count=stats.frame_count,
        scenario=scenario,
    )
    classification = ClassificationSummary(
        motion_level=_motion_level_from_presence_state(presence.state),
        presence=presence.present,
        confidence=presence.confidence,
        state=presence.state,
        moving=presence.moving,
        presence_score=presence.presence_score,
        motion_score=motion.total,
        baseline_ready=presence.baseline_ready,
    )
    signal_field = SignalFieldSummary(
        grid_size=(len(amplitude), 1, 1),
        values=amplitude,
    )
    return SensingUpdate(
        source=source,
        tick=tick,
        nodes=(node,),
        features=features,
        classification=classification,
        signal_field=signal_field,
        vital_signs={},
        estimated_persons=1 if presence.present else 0,
    )


def sensing_update_from_record(
    record: JsonRecord,
    *,
    source: str,
    tick: int,
    preserve_record_source: bool = False,
) -> SensingUpdate:
    """Convert a JSONL update or simple CSI-ish record to ``SensingUpdate``."""

    if _looks_like_update(record):
        payload = dict(record)
        if not preserve_record_source:
            payload["source"] = source
        payload.setdefault("tick", tick)
        return SensingUpdate.from_dict(payload, default_source=source, default_tick=tick)

    amplitude = _record_amplitude(record)
    mean_amplitude = float(np.mean(amplitude)) if amplitude.size else 0.0
    variance = float(np.var(amplitude)) if amplitude.size else 0.0
    motion_score = _clamp01(math.sqrt(max(variance, 0.0)) / (abs(mean_amplitude) + 1e-9))
    presence = bool(record.get("presence", motion_score > 0.04))
    rssi = _safe_float(record.get("rssi_dbm", record.get("rssi", 0.0)))

    node = NodeInfo(
        node_id=record.get("node_id", record.get("id", "replay-node-1")),
        rssi_dbm=rssi,
        position=record.get("position", (0.0, 0.0, 0.0)),
        amplitude=_as_float_list(amplitude),
        subcarrier_count=int(record.get("subcarrier_count", amplitude.size)),
        metadata={"timestamp": record.get("timestamp"), "noise_floor": record.get("noise_floor")},
    )

    features = dict(record.get("features", {}) or {})
    features.setdefault("mean_rssi", rssi)
    features.setdefault("variance", variance)
    features.setdefault("mean_amplitude", mean_amplitude)
    features.setdefault("motion_score", motion_score)
    features.setdefault("motion_band_power", motion_score)
    features.setdefault("breathing_band_power", 0.0)
    features.setdefault("dominant_freq_hz", 0.0)
    features.setdefault("change_points", _count_change_points(amplitude))
    features.setdefault("spectral_power", variance)

    classification = dict(record.get("classification", {}) or {})
    classification.setdefault("presence", presence)
    classification.setdefault("motion_level", _motion_level_from_score(motion_score, presence))
    classification.setdefault("confidence", motion_score if presence else 1.0 - motion_score)
    classification.setdefault("state", "moving" if motion_score > 0.12 else "still" if presence else "empty")
    classification.setdefault("moving", bool(presence and motion_score > 0.12))
    classification.setdefault("presence_score", motion_score)
    classification.setdefault("motion_score", motion_score)

    signal_field = dict(record.get("signal_field", {}) or {})
    if not signal_field:
        signal_field = SignalFieldSummary(grid_size=(int(amplitude.size), 1, 1), values=_as_float_list(amplitude)).to_dict()

    vital_signs = dict(record.get("vital_signs", {}) or {})
    return SensingUpdate(
        source=source,
        tick=tick,
        nodes=record.get("nodes", (node,)),
        features=features,
        classification=classification,
        signal_field=signal_field,
        vital_signs=vital_signs,
        estimated_persons=int(record.get("estimated_persons", 1 if presence else 0)),
    )


def _load_replay_records(recording: ReplayInput) -> list[dict[str, Any]]:
    if isinstance(recording, (str, Path)):
        lines = Path(recording).read_text(encoding="utf-8").splitlines()
        return [_parse_jsonl_line(line) for line in lines if _jsonl_data_line(line)]

    records: list[dict[str, Any]] = []
    for item in recording:
        if isinstance(item, Mapping):
            records.append(dict(item))
        elif _jsonl_data_line(item):
            records.append(_parse_jsonl_line(item))
    return records


def _jsonl_data_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped and not stripped.startswith("#"))


def _parse_jsonl_line(line: str) -> dict[str, Any]:
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("each replay JSONL line must decode to an object")
    return payload


def _looks_like_update(record: JsonRecord) -> bool:
    return record.get("type") == "sensing_update" or (
        "nodes" in record
        and "features" in record
        and "classification" in record
        and "signal_field" in record
    )


def _record_amplitude(record: JsonRecord) -> NDArray[np.float64]:
    for key in ("subcarriers", "amplitude", "amplitudes", "values"):
        if key in record:
            return np.asarray(record[key], dtype=np.float64).reshape(-1)
    csi = record.get("csi")
    if csi is not None:
        values = np.asarray(csi)
        if np.iscomplexobj(values):
            return np.abs(values).astype(np.float64, copy=False).reshape(-1)
        return np.asarray(values, dtype=np.float64).reshape(-1)
    return np.asarray([], dtype=np.float64)


def _edge_vitals_update(packet: EdgeVitalsPacket, *, source: str, tick: int) -> SensingUpdate:
    motion_score = _clamp01(float(packet.motion_energy))
    presence_score = _clamp01(float(packet.presence_score))
    node = NodeInfo(
        node_id=f"esp32-node-{packet.node_id}",
        rssi_dbm=float(packet.rssi_dbm),
        metadata={"packet_node_id": int(packet.node_id), "timestamp_ms": int(packet.timestamp_ms)},
    )
    return SensingUpdate(
        source=source,
        tick=tick,
        nodes=(node,),
        features={
            "packet_type": "edge_vitals",
            "mean_rssi": float(packet.rssi_dbm),
            "motion_band_power": float(packet.motion_energy),
            "presence_score": float(packet.presence_score),
            "timestamp_ms": int(packet.timestamp_ms),
        },
        classification=ClassificationSummary(
            motion_level="present_moving" if packet.motion else "present_still" if packet.presence else "absent",
            presence=packet.presence,
            confidence=presence_score,
            state="moving" if packet.motion else "still" if packet.presence else "empty",
            moving=packet.motion,
            presence_score=presence_score,
            motion_score=motion_score,
        ),
        signal_field={},
        vital_signs={
            "breathing_rate_bpm": packet.breathing_rate_bpm,
            "heart_rate_bpm": packet.heartrate_bpm,
            "motion": packet.motion,
            "fall_detected": packet.fall_detected,
            "n_persons": int(packet.n_persons),
        },
        estimated_persons=int(packet.n_persons),
    )


def _sync_update(packet: SyncPacket, *, source: str, tick: int) -> SensingUpdate:
    node = NodeInfo(
        node_id=f"esp32-node-{packet.node_id}",
        metadata={
            "packet_node_id": int(packet.node_id),
            "protocol_version": int(packet.protocol_version),
            "local_us": int(packet.local_us),
            "epoch_us": int(packet.epoch_us),
        },
    )
    return SensingUpdate(
        source=source,
        tick=tick,
        nodes=(node,),
        features={"packet_type": "sync", "sequence_high_water": int(packet.sequence_high_water)},
        classification={"motion_level": "absent", "presence": False, "confidence": 0.0},
        signal_field={},
        vital_signs={},
        estimated_persons=0,
    )


def _wasm_event_update(packet: WasmEventPacket, *, source: str, tick: int) -> SensingUpdate:
    node = NodeInfo(
        node_id=f"esp32-node-{packet.node_id}",
        metadata={"packet_node_id": int(packet.node_id), "module_id": int(packet.module_id)},
    )
    return SensingUpdate(
        source=source,
        tick=tick,
        nodes=(node,),
        features={
            "packet_type": "wasm_event",
            "module_id": int(packet.module_id),
            "events": [
                {"event_type": int(event.event_type), "value": float(event.value)}
                for event in packet.events
            ],
        },
        classification={"motion_level": "absent", "presence": False, "confidence": 0.0},
        signal_field={},
        vital_signs={},
        estimated_persons=0,
    )


def _generic_packet_update(
    parsed: Any,
    *,
    packet: UdpPacket[Any],
    source: str,
    tick: int,
) -> SensingUpdate:
    return SensingUpdate(
        source=source,
        tick=tick,
        nodes=(),
        features={
            "packet_type": type(parsed).__name__,
            "raw_bytes": len(packet.raw),
            "remote_address": list(packet.address),
            "received_at": float(packet.received_at),
        },
        classification={"motion_level": "absent", "presence": False, "confidence": 0.0},
        signal_field={},
        vital_signs={},
        estimated_persons=0,
    )


def _raw_packet_to_frame(packet: RawCsiPacket) -> CsiFrame:
    iq = np.asarray(packet.iq_pairs(), dtype=np.float64).reshape(
        packet.n_antennas,
        packet.n_subcarriers,
        2,
    )
    data = iq[..., 0] + 1j * iq[..., 1]
    metadata = CsiMetadata(
        device_id=f"esp32-node-{packet.node_id}",
        frequency_band=_frequency_band_from_mhz(packet.frequency_mhz),
        channel=0,
        bandwidth_mhz=40 if packet.bandwidth_40mhz else 20,
        antenna_config=AntennaConfig(1, packet.n_antennas),
        rssi_dbm=packet.rssi_dbm,
        noise_floor_dbm=packet.noise_floor_dbm,
        sequence_number=packet.sequence_number,
    )
    return CsiFrame(metadata, data)


def _frequency_band_from_mhz(frequency_mhz: int) -> FrequencyBand:
    if frequency_mhz >= 5925:
        return FrequencyBand.BAND_6_GHZ
    if frequency_mhz >= 4900:
        return FrequencyBand.BAND_5_GHZ
    return FrequencyBand.BAND_2_4_GHZ


def _mean_subcarrier_amplitude(amplitude: NDArray[np.float64]) -> list[float]:
    values = np.asarray(amplitude, dtype=np.float64)
    if values.size == 0:
        return []
    if values.ndim == 1:
        collapsed = values
    elif values.ndim == 2:
        collapsed = np.mean(values, axis=0)
    else:
        collapsed = np.mean(values, axis=tuple(range(values.ndim - 1)))
    return _as_float_list(collapsed)


def _temporal_mean_amplitude(amplitude: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(amplitude, dtype=np.float64)
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    if values.ndim <= 1:
        return values.reshape(-1)
    return np.mean(values, axis=tuple(range(1, values.ndim)))


def _count_change_points(values: Sequence[float] | NDArray[np.float64]) -> int:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 3:
        return 0
    deltas = np.abs(np.diff(array))
    threshold = float(np.mean(deltas) + 2.0 * np.std(deltas))
    if threshold <= 0.0:
        return 0
    return int(np.count_nonzero(deltas > threshold))


def _mean_rssi(frames: Sequence[CsiFrame]) -> float:
    if not frames:
        return 0.0
    return float(np.mean([frame.metadata.rssi_dbm for frame in frames]))


def _motion_level_from_presence_state(state: str) -> str:
    return {
        "empty": "absent",
        "still": "present_still",
        "moving": "present_moving",
    }.get(state, state)


def _motion_level_from_score(score: float, presence: bool) -> str:
    if not presence:
        return "absent"
    if score > 0.25:
        return "active"
    if score > 0.12:
        return "present_moving"
    return "present_still"


def _as_float_list(values: Sequence[float] | NDArray[np.float64]) -> list[float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return [float(value) for value in array if np.isfinite(value)]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


__all__ = [
    "LatestState",
    "ReplaySensingSource",
    "SensingSource",
    "SimulatedSensingSource",
    "UdpSensingSource",
    "sensing_update_from_record",
    "sensing_update_from_window",
]
