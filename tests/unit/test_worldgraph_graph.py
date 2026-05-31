from __future__ import annotations

import pytest

from ruview.worldgraph.graph import UnknownNodeError, WorldGraph
from ruview.worldgraph.model import EnuPoint, SensorModality, WorldEdge, WorldId, WorldNode, ZoneBoundsEnu
from ruview.worldgraph.provenance import SemanticProvenance


def test_upsert_allocates_replaces_and_validates_unknown_edges() -> None:
    graph = WorldGraph()
    room_id = graph.upsert_node(_living_room())

    assert not room_id.is_unassigned()
    assert graph.node_count() == 1

    graph.upsert_node(
        WorldNode.room(
            id=room_id,
            area_id="living_room",
            name="Lounge",
            bounds_enu=ZoneBoundsEnu.rectangle(0.0, 0.0, 5.0, 4.0),
        )
    )

    assert graph.node_count() == 1
    assert graph.node(room_id).name == "Lounge"

    with pytest.raises(UnknownNodeError) as error:
        graph.add_edge(room_id, WorldId(999), WorldEdge.observes(quality=1.0, last_seen_unix_ms=0))
    assert error.value.node_id == WorldId(999)


def test_observability_and_location_queries() -> None:
    graph = WorldGraph()
    room_id = graph.upsert_node(_living_room())
    sensor_id = graph.upsert_node(
        WorldNode.sensor(
            device_id="esp32-com9",
            position=EnuPoint(1.0, 1.0, 0.0),
            modality=SensorModality.WIFI_CSI,
        )
    )
    person_id = graph.upsert_node(
        WorldNode.person_track(
            track_id=7,
            last_position=EnuPoint(2.0, 2.0, 0.0),
        )
    )

    graph.add_edge(sensor_id, room_id, WorldEdge.observes(quality=0.9, last_seen_unix_ms=1))
    graph.add_edge(person_id, room_id, WorldEdge.located_in(since_unix_ms=100))

    assert graph.room_for_area("living_room") == room_id
    assert graph.observed_by(sensor_id) == [room_id]
    assert graph.contents_of(room_id) == [person_id]
    assert graph.neighbors(sensor_id) == [(room_id, WorldEdge.observes(quality=0.9, last_seen_unix_ms=1))]


def test_semantic_provenance_and_contradiction_are_queryable() -> None:
    graph = WorldGraph()
    event_id = graph.upsert_node(
        WorldNode.event(
            event_type="motion",
            at_unix_ms=10,
        )
    )
    provenance = SemanticProvenance(
        evidence=("ev:abc",),
        model_version="rfenc-1.0",
        calibration_version="cal:uuid",
        privacy_decision="PrivateHome/Allow",
    )

    present_id = graph.add_semantic_state("present", 0.9, 11, provenance, [event_id, WorldId(404)])
    absent_id = graph.add_semantic_state("absent", 0.6, 12, provenance, [event_id])
    graph.add_contradiction(present_id, absent_id, magnitude=0.3, flag="flag:ts")

    assert graph.node(present_id).provenance == provenance
    assert graph.node(absent_id) is not None
    assert graph.neighbors(present_id).count((event_id, WorldEdge.derived_from("ev:abc"))) == 1
    assert (absent_id, WorldEdge.contradicts(magnitude=0.3, flag="flag:ts")) in graph.neighbors(present_id)


def test_privacy_rollup_suppresses_person_tracks() -> None:
    graph = WorldGraph()
    room_id = graph.upsert_node(_living_room())
    person_id = graph.upsert_node(
        WorldNode.person_track(
            track_id=1,
            last_position=EnuPoint(1.0, 1.0, 0.0),
        )
    )
    sensor_id = graph.upsert_node(
        WorldNode.sensor(
            device_id="s",
            position=EnuPoint(0.0, 0.0, 0.0),
            modality=SensorModality.WIFI_CSI,
        )
    )
    graph.add_edge(sensor_id, room_id, WorldEdge.observes(quality=1.0, last_seen_unix_ms=0))
    graph.add_edge(sensor_id, person_id, WorldEdge.observes(quality=1.0, last_seen_unix_ms=0))

    rollup = graph.apply_privacy_mode(
        "StrictNoIdentity",
        "SuppressIdentity",
        lambda _sensor_kind, node_kind: node_kind != "person_track",
    )

    assert rollup.allowed_pairs == 1
    assert rollup.denied_pairs == ((sensor_id, person_id),)
    assert rollup.suppressed_nodes == (person_id,)
    assert (person_id, WorldEdge.privacy_limited_by("StrictNoIdentity", "SuppressIdentity", False)) in graph.neighbors(sensor_id)


def test_deterministic_json_roundtrip_preserves_graph_behavior() -> None:
    graph = WorldGraph(registration={"origin": {"lat": 59.91, "lon": 10.75}})
    room_id = graph.upsert_node(_living_room())
    sensor_id = graph.upsert_node(
        WorldNode.sensor(
            device_id="s",
            position=EnuPoint(0.0, 0.0, 0.0),
            modality=SensorModality.WIFI_CSI,
        )
    )
    graph.add_edge(sensor_id, room_id, WorldEdge.observes(quality=0.8, last_seen_unix_ms=5))

    encoded = graph.to_json()
    decoded = WorldGraph.from_json(encoded)

    assert decoded.node_count() == 2
    assert decoded.room_for_area("living_room") == room_id
    assert decoded.observed_by(sensor_id) == [room_id]
    assert decoded.to_json() == encoded
    assert WorldGraph.from_json(encoded.encode("utf-8")).to_json() == encoded


def _living_room() -> WorldNode:
    return WorldNode.room(
        area_id="living_room",
        name="Living Room",
        bounds_enu=ZoneBoundsEnu.rectangle(0.0, 0.0, 5.0, 4.0),
        floor=0,
    )
