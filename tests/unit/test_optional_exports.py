from __future__ import annotations

import numpy as np

from ruview.browser_visualization import build_canvas_pose_payload, build_websocket_url, fuse_embeddings
from ruview.hardware import SerialPortInfo, build_nvs_provisioning_plan, is_esp32_compatible
from ruview.homecore import Automation, ServiceAction, StateMachine, StateTrigger
from ruview.nvsim import DipoleSource, Pipeline, PipelineConfig, Scene
from ruview.swarm import (
    CsiDetection,
    DroneState,
    MeshTopology,
    MultiViewFusion,
    NodeId,
    Position3D,
)


def test_homecore_exports_support_state_and_automation_construction() -> None:
    states = StateMachine()
    current = states.set("sensor.temperature", "21", {"unit": "C"})
    automation = Automation(
        "auto.temperature",
        trigger=StateTrigger("sensor.temperature", to="22"),
        action=ServiceAction("climate", "set_preset", {"preset": "eco"}),
    )

    assert current.entity_id.as_str() == "sensor.temperature"
    assert automation.trigger[0].entity_id.as_str() == "sensor.temperature"
    assert automation.action[0].domain == "climate"
    assert automation.action[0].service == "set_preset"


def test_nvsim_exports_run_deterministic_pipeline() -> None:
    scene = Scene(
        dipoles=(DipoleSource((0.0, 0.0, 0.5), (0.0, 0.0, 1.0e-3)),),
        sensors=((0.0, 0.0, 0.0),),
    )
    config = PipelineConfig(noise_enabled=False)

    frames, witness = Pipeline(scene, config, seed=7).run_with_witness(2)

    assert len(frames) == 2
    assert len(witness) == 32
    assert frames[0].sensor_id == 0


def test_swarm_exports_cover_topology_and_multiview_fusion() -> None:
    topology = MeshTopology(cluster_head=NodeId(0))
    topology.update_node(DroneState(id=NodeId(0), position=Position3D(0.0, 0.0, -10.0)))
    topology.update_node(DroneState(id=NodeId(1), position=Position3D(3.0, 4.0, -10.0)))

    fused = MultiViewFusion(min_viewpoints=2).fuse(
        [
            CsiDetection(NodeId(0), 0.8, Position3D(9.0, 10.0, 0.0)),
            CsiDetection(NodeId(1), 0.9, Position3D(11.0, 10.0, 0.0)),
        ],
        [
            (NodeId(0), Position3D(0.0, 0.0, -10.0)),
            (NodeId(1), Position3D(20.0, 0.0, -10.0)),
        ],
    )

    assert topology.nearest_k(NodeId(0), 1) == [NodeId(1)]
    assert fused is not None
    assert fused.contributing_drones == (NodeId(0), NodeId(1))
    assert fused.estimated_position.x > 9.0


def test_desktop_and_browser_exports_are_public_and_pure_data() -> None:
    assert is_esp32_compatible(0x303A, 0x1001)
    port = SerialPortInfo("/dev/cu.usbserial-1410")
    plan = build_nvs_provisioning_plan({"wifi_ssid": "Lab", "node_id": 2}, chunk_size=8)
    payload = build_canvas_pose_payload([(0.5, 0.25, 0.9)] * 17, width=640, height=480)

    assert port.is_esp32_compatible
    assert plan.size > 0
    assert build_websocket_url("localhost", 3030) == "ws://localhost:3030/ws/csi"
    assert payload["keypoints"][0]["x"] == 320.0
    np.testing.assert_allclose(
        fuse_embeddings([1.0, 2.0], [3.0, 6.0], 0.25),
        np.array([2.5, 5.0]),
    )
