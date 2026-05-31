from __future__ import annotations

import hashlib

import numpy as np
import pytest

from ruview.browser_visualization import (
    COCO_SKELETON_EDGES,
    ModalityQuality,
    build_canvas_pose_payload,
    build_websocket_url,
    fuse_embeddings,
    quality_gated_attention_weights,
)
from ruview.hardware.desktop import (
    PROVISION_MAGIC,
    SerialPortInfo,
    build_nvs_provisioning_plan,
    build_wifi_serial_command_plan,
    fallback_serial_port_candidates,
    is_esp32_compatible,
    sort_serial_ports,
)


def test_known_esp32_vid_pid_compatibility() -> None:
    assert is_esp32_compatible(0x10C4, 0xEA60)
    assert is_esp32_compatible(0x1A86, 0x7523)
    assert is_esp32_compatible(0x0403, 0x6015)
    assert is_esp32_compatible("0x303A", "0x1001")

    assert not is_esp32_compatible(0x1234, 0x5678)
    assert not is_esp32_compatible(None, 0xEA60)


def test_serial_port_metadata_sorts_compatible_ports_first() -> None:
    ports = [
        SerialPortInfo(name="/dev/ttyS0", manufacturer="System"),
        SerialPortInfo(name="/dev/cu.usbserial-2", manufacturer="USB Serial"),
        SerialPortInfo(name="/dev/ttyUSB0", vid=0x1A86, pid=0x7523, manufacturer="QinHeng"),
    ]

    sorted_ports = sort_serial_ports(ports)

    assert [port.name for port in sorted_ports] == [
        "/dev/cu.usbserial-2",
        "/dev/ttyUSB0",
        "/dev/ttyS0",
    ]
    assert sorted_ports[0].is_esp32_compatible
    assert sorted_ports[2].is_esp32_compatible is False
    assert sorted_ports[2].vid_hex is None
    assert sorted_ports[1].vid_hex == "0x1A86"
    assert sorted_ports[1].pid_hex == "0x7523"


def test_fallback_serial_port_filtering_uses_provided_names_only() -> None:
    mac_ports = fallback_serial_port_candidates(
        ["cu.Bluetooth-Incoming-Port", "cu.usbserial-1410", "/dev/tty.usbmodem1101"],
        platform="darwin",
    )
    linux_ports = fallback_serial_port_candidates(
        ["tty0", "ttyUSB0", "ttyACM1", "cu.usbserial-1410"],
        platform="linux",
    )

    assert [port.name for port in mac_ports] == ["/dev/cu.usbserial-1410", "/dev/tty.usbmodem1101"]
    assert [port.name for port in linux_ports] == ["/dev/ttyACM1", "/dev/ttyUSB0"]
    assert all(port.is_esp32_compatible for port in [*mac_ports, *linux_ports])


def test_wifi_command_plan_keeps_password_out_of_repr() -> None:
    plan = build_wifi_serial_command_plan("/dev/ttyUSB0", "lab-net", "secret-pass")

    assert plan.commands == (
        "wifi_config lab-net secret-pass\r\n",
        "wifi lab-net secret-pass\r\n",
        "set ssid lab-net\r\n",
        "set password secret-pass\r\n",
        "reboot\r\n",
    )
    assert plan.redacted_commands[0] == "wifi_config lab-net <redacted>\r\n"
    assert "secret-pass" not in repr(plan)
    assert "<redacted>" in repr(plan)

    with pytest.raises(ValueError):
        build_wifi_serial_command_plan("/dev/ttyUSB0", "lab\nnet", "secret-pass")


def test_nvs_header_checksum_and_chunks_match_desktop_protocol() -> None:
    config = {
        "wifi_ssid": "Lab",
        "wifi_password": "secret",
        "target_port": 5005,
        "node_id": 7,
        "tdm_slot": 1,
        "tdm_total": 4,
        "channel_list": [1, 6, 11],
        "wasm_verify": True,
    }

    plan = build_nvs_provisioning_plan(config, chunk_size=10)
    expected_checksum = hashlib.sha256(plan.nvs_data).digest()[:8].hex()

    assert plan.header == PROVISION_MAGIC + b"\x01" + len(plan.nvs_data).to_bytes(4, "little")
    assert plan.checksum == expected_checksum
    assert plan.checksum_line == f"{expected_checksum}\n".encode("ascii")
    assert b"wifi_ssid" in plan.nvs_data
    assert b"wifi_pass" in plan.nvs_data
    assert plan.nvs_data.endswith(b"\x00")
    assert b"channels" in plan.nvs_data
    assert plan.chunks == tuple(plan.nvs_data[index : index + 10] for index in range(0, len(plan.nvs_data), 10))
    assert "secret" not in repr(plan)


def test_websocket_url_builder_normalizes_browser_csi_urls() -> None:
    assert build_websocket_url("localhost", 3030) == "ws://localhost:3030/ws/csi"
    assert (
        build_websocket_url("https://example.test", None, "/ws/csi", query={"limit": 2})
        == "wss://example.test/ws/csi?limit=2"
    )
    assert build_websocket_url("::1", 3030, "ws/csi") == "ws://[::1]:3030/ws/csi"


def test_attention_fusion_and_quality_gating_weights() -> None:
    visual = np.array([1.0, 10.0, 100.0])
    csi = np.array([3.0, 30.0, 300.0])
    weights = np.array([0.0, 0.5, 1.0])

    fused = fuse_embeddings(visual, csi, weights)

    np.testing.assert_allclose(fused, np.array([3.0, 20.0, 100.0]))

    gated = quality_gated_attention_weights(
        0.5,
        shape=3,
        quality=ModalityQuality(visual=0.25, csi=0.75),
    )
    np.testing.assert_allclose(gated, np.full(3, 0.25))

    csi_only = quality_gated_attention_weights(
        [0.2, 0.8],
        shape=2,
        quality=ModalityQuality(visual_available=False, csi_available=True),
    )
    np.testing.assert_allclose(csi_only, np.zeros(2))


def test_browser_pose_payload_shape_and_visible_edges() -> None:
    keypoints = [(index / 16.0, index / 32.0, 0.9) for index in range(17)]
    payload = build_canvas_pose_payload(keypoints, width=640, height=480, min_confidence=0.5)

    assert payload["schema"] == "ruview.browser_pose.canvas.v1"
    assert payload["width"] == 640
    assert payload["height"] == 480
    assert len(payload["keypoints"]) == 17
    assert len(payload["edges"]) == len(COCO_SKELETON_EDGES)
    assert payload["keypoints"][0] == {
        "index": 0,
        "name": "nose",
        "x": 0.0,
        "y": 0.0,
        "confidence": 0.9,
        "visible": True,
    }
    assert payload["keypoints"][16]["x"] == pytest.approx(640.0)
    assert payload["keypoints"][16]["y"] == pytest.approx(240.0)
    assert all(edge["visible"] for edge in payload["edges"])

    low_confidence = list(keypoints)
    low_confidence[7] = (0.5, 0.5, 0.1)
    payload = build_canvas_pose_payload(low_confidence, width=640, height=480, min_confidence=0.5)
    hidden_edges = [edge for edge in payload["edges"] if edge["start"] == 5 and edge["end"] == 7]
    assert hidden_edges[0]["visible"] is False
