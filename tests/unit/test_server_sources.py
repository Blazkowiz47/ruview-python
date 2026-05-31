from __future__ import annotations

import json

from ruview.hardware import SyntheticCsiConfig, UdpPacket
from ruview.protocols import EdgeVitalsPacket
from ruview.server import (
    ClassificationSummary,
    FeatureSummary,
    LatestState,
    NodeInfo,
    ReplaySensingSource,
    SensingUpdate,
    SignalFieldSummary,
    SimulatedSensingSource,
    UdpSensingSource,
)


def test_sensing_update_to_dict_matches_plan_shape() -> None:
    update = SensingUpdate(
        source="simulated",
        tick=7,
        nodes=(NodeInfo("node-1", rssi_dbm=-48.0, amplitude=(1.0, 2.0)),),
        features=FeatureSummary(mean_rssi=-48.0, variance=0.2, scenario="walking"),
        classification=ClassificationSummary(
            motion_level="present_moving",
            presence=True,
            confidence=0.8,
        ),
        signal_field=SignalFieldSummary(grid_size=(2, 1, 1), values=(1.0, 2.0)),
        vital_signs={"status": "not_estimated"},
        estimated_persons=1,
    )

    payload = update.to_dict()

    assert list(payload) == [
        "type",
        "source",
        "tick",
        "nodes",
        "features",
        "classification",
        "signal_field",
        "vital_signs",
        "estimated_persons",
    ]
    assert payload["type"] == "sensing_update"
    assert payload["source"] == "simulated"
    assert payload["tick"] == 7
    assert payload["nodes"][0]["subcarrier_count"] == 2
    assert payload["features"]["scenario"] == "walking"
    assert payload["classification"]["presence"] is True
    assert payload["signal_field"]["grid_size"] == [2, 1, 1]
    assert payload["estimated_persons"] == 1
    json.dumps(payload)


def test_simulated_source_cycles_scenarios_and_classifies_presence() -> None:
    config = SyntheticCsiConfig(seed=456, frames=96, streams=3, subcarriers=32)
    source = SimulatedSensingSource(
        scenarios=("empty_room", "person_present", "walking"),
        config=config,
    )

    payloads = [update.to_dict() for update in source.iter_updates(limit=4)]

    assert [payload["tick"] for payload in payloads] == [1, 2, 3, 4]
    assert [payload["features"]["scenario"] for payload in payloads] == [
        "empty_room",
        "person_present",
        "walking",
        "empty_room",
    ]
    assert payloads[0]["classification"]["motion_level"] == "absent"
    assert payloads[0]["estimated_persons"] == 0
    assert payloads[1]["classification"]["motion_level"] == "present_still"
    assert payloads[1]["classification"]["presence"] is True
    assert payloads[2]["classification"]["motion_level"] == "present_moving"
    assert payloads[2]["classification"]["moving"] is True
    assert payloads[2]["nodes"][0]["node_id"] == "synthetic-node-1"
    assert payloads[2]["signal_field"]["grid_size"] == [32, 1, 1]


def test_replay_source_reads_update_and_csi_jsonl_records(tmp_path) -> None:
    path = tmp_path / "recording.jsonl"
    update_record = {
        "type": "sensing_update",
        "source": "simulated",
        "tick": 41,
        "nodes": [{"node_id": "recorded-node", "rssi_dbm": -50.0}],
        "features": {"scenario": "recorded"},
        "classification": {"motion_level": "present_still", "presence": True, "confidence": 0.7},
        "signal_field": {"grid_size": [1, 1, 1], "values": [1.0]},
        "vital_signs": {},
        "estimated_persons": 1,
    }
    csi_record = {
        "timestamp": 123.0,
        "node_id": "replay-node-2",
        "subcarriers": [1.0, 1.4, 0.8, 1.2],
        "rssi": -47.0,
        "noise_floor": -92.0,
        "features": {"label": "fixture"},
    }
    path.write_text(
        "\n".join([json.dumps(update_record), json.dumps(csi_record)]),
        encoding="utf-8",
    )

    source = ReplaySensingSource(path)
    first = source.next_update()
    second = source.next_update()

    assert first is not None
    assert second is not None
    first_payload = first.to_dict()
    second_payload = second.to_dict()
    assert first_payload["source"] == "replay"
    assert first_payload["tick"] == 41
    assert first_payload["nodes"][0]["node_id"] == "recorded-node"
    assert second_payload["source"] == "replay"
    assert second_payload["tick"] == 42
    assert second_payload["nodes"][0]["node_id"] == "replay-node-2"
    assert second_payload["features"]["label"] == "fixture"
    assert second_payload["classification"]["presence"] is True
    assert source.next_update() is None


def test_latest_state_keeps_latest_and_bounded_history() -> None:
    state = LatestState(capacity=2)
    updates = [
        SensingUpdate(source="simulated", tick=tick, estimated_persons=tick)
        for tick in (1, 2, 3)
    ]

    for update in updates:
        state.update(update)

    assert state.latest is updates[-1]
    assert state.latest_dict()["tick"] == 3
    assert [update.tick for update in state.history] == [2, 3]
    assert [payload["tick"] for payload in state.history_dicts()] == [2, 3]


def test_udp_source_constructs_and_times_out_on_localhost() -> None:
    with UdpSensingSource(
        "127.0.0.1",
        0,
        parser=lambda payload: payload,
        timeout=0.01,
    ) as source:
        assert source.address[0] == "127.0.0.1"
        assert source.next_update() is None
        assert list(source.iter_updates(limit=1)) == []


def test_udp_source_converts_edge_vitals_packets() -> None:
    receiver = _OnePacketReceiver(
        UdpPacket(
            raw=b"edge",
            parsed=EdgeVitalsPacket(
                magic=0xC5110002,
                node_id=3,
                flags=0x07,
                breathing_rate_raw=1842,
                heartrate_raw=725000,
                rssi_dbm=-42,
                n_persons=2,
                reserved=b"\x00\x00",
                motion_energy=0.6,
                presence_score=0.9,
                timestamp_ms=123456,
                reserved2=0,
            ),
            address=("127.0.0.1", 9),
            received_at=1.0,
        )
    )
    source = UdpSensingSource(receiver=receiver, timeout=0.01)

    payload = source.next_update().to_dict()

    assert payload["source"] == "esp32"
    assert payload["tick"] == 1
    assert payload["nodes"][0]["node_id"] == "esp32-node-3"
    assert payload["classification"]["motion_level"] == "present_moving"
    assert payload["classification"]["presence"] is True
    assert payload["vital_signs"]["breathing_rate_bpm"] == 18.42
    assert payload["vital_signs"]["heart_rate_bpm"] == 72.5
    assert payload["estimated_persons"] == 2
    assert source.next_update() is None


class _OnePacketReceiver:
    address = ("127.0.0.1", 9999)
    closed = False

    def __init__(self, packet: UdpPacket[EdgeVitalsPacket]) -> None:
        self._packets = [packet]

    def receive(self, *, timeout: float | None = None):
        return self._packets.pop(0) if self._packets else None

    def close(self) -> None:
        self.closed = True
