from __future__ import annotations

import socket
import sys
import types

import pytest

from ruview.hardware import UdpReceiver, iter_packet_bytes, replay_packets


def test_udp_receiver_receives_and_parses_datagram() -> None:
    calls: list[bytes] = []

    def parser(payload: bytes) -> str:
        calls.append(payload)
        return payload.decode("ascii")

    with UdpReceiver("127.0.0.1", 0, parser=parser, timeout=0.5) as receiver:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(b"frame-1", receiver.address)

        packet = receiver.receive()

    assert packet is not None
    assert packet.raw == b"frame-1"
    assert packet.parsed == "frame-1"
    assert packet.address[0] == "127.0.0.1"
    assert calls == [b"frame-1"]


def test_udp_receiver_returns_none_on_timeout() -> None:
    with UdpReceiver("127.0.0.1", 0, parser=lambda payload: payload, timeout=0.01) as receiver:
        assert receiver.receive() is None


def test_udp_receiver_close_is_idempotent_and_rejects_receive() -> None:
    receiver: UdpReceiver[bytes] = UdpReceiver(
        "127.0.0.1",
        0,
        parser=lambda payload: payload,
        timeout=0.01,
    )

    receiver.close()
    receiver.close()

    assert receiver.closed
    with pytest.raises(RuntimeError, match="closed"):
        receiver.receive()


def test_udp_receiver_context_manager_closes_socket() -> None:
    with UdpReceiver("127.0.0.1", 0, parser=lambda payload: payload) as receiver:
        assert not receiver.closed

    assert receiver.closed


def test_replay_packets_reads_paths_and_in_memory_packets(tmp_path) -> None:
    first = tmp_path / "one.bin"
    second = tmp_path / "two.bin"
    first.write_bytes(b"\x01\x02")
    second.write_bytes(b"\x03\x04")

    def parser(payload: bytes) -> int:
        return sum(payload)

    packets = list(replay_packets([first, b"\x05", second], parser=parser))

    assert [packet.raw for packet in packets] == [b"\x01\x02", b"\x05", b"\x03\x04"]
    assert [packet.parsed for packet in packets] == [3, 5, 7]
    assert [packet.index for packet in packets] == [0, 1, 2]
    assert packets[0].path == first
    assert packets[1].path is None
    assert packets[2].path == second
    assert list(iter_packet_bytes(first)) == [b"\x01\x02"]


def test_replay_packets_resolves_default_parser_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("ruview.protocols.esp32")
    module.parse_packet = lambda payload: {"size": len(payload)}
    monkeypatch.setitem(sys.modules, "ruview.protocols.esp32", module)

    packets = list(replay_packets([b"abc"]))

    assert packets[0].raw == b"abc"
    assert packets[0].parsed == {"size": 3}
