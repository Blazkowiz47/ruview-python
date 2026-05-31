from __future__ import annotations

from ruview.privacy import (
    BfldFrame,
    BfldFrameHeader,
    BfldPayload,
    PrivacyClass,
    PrivacyMode,
    PrivacyModeRegistry,
)
from ruview.worldgraph import (
    EnuPoint,
    WorldEdge,
    WorldGraph,
    WorldNode,
    ZoneBoundsEnu,
    apply_active_privacy_mode,
    record_trusted_semantic_state,
    witness_of,
)


def test_worldgraph_privacy_exports_and_trust_recording() -> None:
    graph = WorldGraph({"origin": "lab"})
    room = graph.upsert_node(
        WorldNode.room(
            area_id="living_room",
            name="Living Room",
            bounds_enu=ZoneBoundsEnu.rectangle(0.0, 0.0, 5.0, 4.0),
        )
    )
    sensor = graph.upsert_node(
        WorldNode.sensor(
            device_id="esp32-a",
            position=EnuPoint(0.5, 0.5, 1.2),
            modality="wifi_csi",
        )
    )
    person = graph.upsert_node(
        WorldNode.person_track(track_id=7, last_position=EnuPoint(2.0, 1.5, 0.0))
    )
    graph.add_edge(sensor, room, WorldEdge.observes(quality=0.9, last_seen_unix_ms=1_000))
    graph.add_edge(sensor, person, WorldEdge.observes(quality=0.8, last_seen_unix_ms=1_000))
    graph.add_edge(person, room, WorldEdge.located_in(since_unix_ms=1_000))

    registry = PrivacyModeRegistry(PrivacyMode.PrivateHome)
    rollup = apply_active_privacy_mode(graph, registry)
    trusted = record_trusted_semantic_state(
        graph,
        statement="occupancy coherence=0.91 nodes=2 demoted=True",
        confidence=0.91,
        evidence=("ev:node-a", "ev:node-b"),
        model_version="rfenc-v1",
        calibration_version=None,
        privacy=registry,
        valid_from_unix_ms=1_100,
        evidence_sources=(room,),
    )

    assert rollup.suppressed_nodes == (person,)
    assert trusted.demoted is True
    assert trusted.effective_class is PrivacyClass.Restricted
    assert trusted.witness == witness_of(trusted.provenance, trusted.effective_class)
    assert graph.node(trusted.semantic_id) is not None
    assert any(target == room and edge.rel == "derived_from" for target, edge in graph.neighbors(trusted.semantic_id))


def test_privacy_package_exports_bfld_frame_roundtrip() -> None:
    payload = BfldPayload(
        compressed_angle_matrix=b"angle",
        amplitude_proxy=b"amp",
        phase_proxy=b"phase",
        snr_vector=b"snr",
    )
    frame = BfldFrame.from_payload(
        BfldFrameHeader(privacy_class=PrivacyClass.Anonymous.as_u8(), channel=6, n_subcarriers=56),
        payload,
    )

    parsed = BfldFrame.from_bytes(frame.to_bytes())

    assert parsed.header.magic == 0xBF1D_0001
    assert parsed.header.privacy_class == PrivacyClass.Anonymous.as_u8()
    assert parsed.parse_payload().amplitude_proxy == b"amp"
