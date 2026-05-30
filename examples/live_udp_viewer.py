"""Print parsed ESP32 UDP packets as they arrive."""

from __future__ import annotations

import argparse
import sys

from ruview.hardware import UdpPacket, UdpReceiver


def _describe(parsed: object) -> str:
    metadata = getattr(parsed, "metadata", None)
    parts: list[str] = []

    if metadata is not None:
        device = getattr(metadata, "device_id", None) or getattr(metadata, "node_id", None)
        sequence = getattr(metadata, "sequence_number", None)
        if sequence is None:
            sequence = getattr(metadata, "sequence", None)
        rssi = getattr(metadata, "rssi_dbm", None)
        if rssi is None:
            rssi = getattr(metadata, "rssi", None)

        if device is not None:
            parts.append(f"device={device}")
        if sequence is not None:
            parts.append(f"seq={sequence}")
        if rssi is not None:
            parts.append(f"rssi={rssi}")

    num_subcarriers = getattr(parsed, "num_subcarriers", None)
    if callable(num_subcarriers):
        parts.append(f"subcarriers={num_subcarriers()}")
    mean_amplitude = getattr(parsed, "mean_amplitude", None)
    if callable(mean_amplitude):
        parts.append(f"mean_amp={mean_amplitude():.3f}")

    return " ".join(parts) if parts else repr(parsed)


def _print_packet(packet: UdpPacket[object]) -> None:
    print(
        f"{packet.address[0]}:{packet.address[1]} "
        f"bytes={len(packet.raw)} {_describe(packet.parsed)}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="UDP bind host")
    parser.add_argument("--port", default=5005, type=int, help="UDP bind port")
    parser.add_argument("--timeout", default=1.0, type=float, help="Receive timeout in seconds")
    parser.add_argument("--limit", type=int, help="Stop after this many parsed packets")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seen = 0

    try:
        with UdpReceiver(args.host, args.port, timeout=args.timeout) as receiver:
            print(f"Listening on {receiver.address[0]}:{receiver.address[1]}", file=sys.stderr)
            while args.limit is None or seen < args.limit:
                packet = receiver.receive()
                if packet is None:
                    continue
                _print_packet(packet)
                seen += 1
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
    except RuntimeError as exc:
        print(f"Cannot parse UDP packets: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
