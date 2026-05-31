from __future__ import annotations

from fastapi.testclient import TestClient

from ruview.hardware import SyntheticCsiConfig
from ruview.server import LatestState, ServerConfig, SimulatedSensingSource, create_app, source_from_config


def test_latest_endpoint_polls_source_and_stores_state() -> None:
    source = SimulatedSensingSource(
        scenarios=("empty_room", "walking"),
        config=SyntheticCsiConfig(seed=1122, frames=64, streams=2, subcarriers=24),
    )
    state = LatestState()
    client = TestClient(create_app(source=source, state=state))

    first = client.get("/api/v1/sensing/latest").json()
    second = client.get("/api/v1/sensing/latest").json()

    assert first["type"] == "sensing_update"
    assert first["source"] == "simulated"
    assert first["tick"] == 1
    assert first["features"]["scenario"] == "empty_room"
    assert second["tick"] == 2
    assert second["features"]["scenario"] == "walking"
    assert state.latest is not None
    assert state.latest.tick == 2


def test_vital_signs_endpoint_uses_latest_update_shape() -> None:
    source = SimulatedSensingSource(
        config=SyntheticCsiConfig(seed=3344, frames=32, streams=2, subcarriers=16),
    )
    client = TestClient(create_app(source=source, state=LatestState()))

    payload = client.get("/api/v1/vital-signs").json()

    assert payload["source"] == "simulated"
    assert payload["tick"] == 1
    assert payload["status"] == "not_estimated"


def test_websocket_sensing_stream_sends_limited_updates() -> None:
    source = SimulatedSensingSource(
        scenarios=("empty_room", "person_present"),
        config=SyntheticCsiConfig(seed=5566, frames=48, streams=2, subcarriers=16),
    )
    client = TestClient(create_app(source=source, state=LatestState(), config=ServerConfig(tick_ms=0)))

    with client.websocket_connect("/ws/sensing?limit=2&tick_ms=0") as websocket:
        first = websocket.receive_json()
        second = websocket.receive_json()

    assert first["tick"] == 1
    assert second["tick"] == 2
    assert first["features"]["scenario"] == "empty_room"
    assert second["features"]["scenario"] == "person_present"


def test_health_reports_local_only_config() -> None:
    client = TestClient(create_app(config=ServerConfig(bind_host="127.0.0.1")))

    payload = client.get("/health").json()

    assert payload["status"] == "ok"
    assert payload["local_only"] is True


def test_source_from_config_requires_replay_path() -> None:
    try:
        source_from_config(ServerConfig(source="replay"))
    except ValueError as exc:
        assert "replay_path" in str(exc)
    else:
        raise AssertionError("expected replay source without path to fail")
