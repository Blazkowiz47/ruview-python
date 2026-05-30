"""Synchronous UDP capture and replay helpers for ESP32 packet streams."""

from __future__ import annotations

import importlib
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Generic, Iterable, Iterator, TypeAlias, TypeVar

ParsedPacket = TypeVar("ParsedPacket")
PacketParser: TypeAlias = Callable[[bytes], ParsedPacket]
SocketAddress: TypeAlias = tuple[str, int]
PacketBytesSource: TypeAlias = str | bytes | bytearray | memoryview | Path
ReplaySource: TypeAlias = PacketBytesSource | Iterable[PacketBytesSource]

_DEFAULT_BUFFER_SIZE = 65_535
_TIMEOUT_UNSET = object()


@dataclass(frozen=True)
class UdpPacket(Generic[ParsedPacket]):
    """One UDP datagram plus the object produced by the packet parser."""

    raw: bytes
    parsed: ParsedPacket
    address: SocketAddress
    received_at: float


@dataclass(frozen=True)
class ReplayPacket(Generic[ParsedPacket]):
    """One replayed datagram plus the object produced by the packet parser."""

    raw: bytes
    parsed: ParsedPacket
    index: int
    path: Path | None = None


def _load_default_parser() -> PacketParser[Any]:
    """Resolve the ESP32 parser lazily so this module imports before Milestone 2 lands."""

    try:
        module = importlib.import_module("ruview.protocols.esp32")
    except ModuleNotFoundError as exc:
        if exc.name == "ruview.protocols.esp32":
            raise RuntimeError(
                "No packet parser was supplied and ruview.protocols.esp32.parse_packet "
                "is not available yet."
            ) from exc
        raise

    parser = getattr(module, "parse_packet", None)
    if not callable(parser):
        raise RuntimeError("ruview.protocols.esp32.parse_packet is not callable.")
    return parser


def _resolve_parser(parser: PacketParser[ParsedPacket] | None) -> PacketParser[ParsedPacket]:
    if parser is not None:
        return parser
    return _load_default_parser()


class UdpReceiver(Generic[ParsedPacket]):
    """Small blocking UDP receiver for host-side CSI research capture."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5005,
        *,
        parser: PacketParser[ParsedPacket] | None = None,
        timeout: float | None = None,
        max_datagram_size: int = _DEFAULT_BUFFER_SIZE,
        reuse_address: bool = False,
    ) -> None:
        if max_datagram_size <= 0:
            raise ValueError("max_datagram_size must be positive")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            if reuse_address:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.settimeout(timeout)
        except Exception:
            sock.close()
            raise

        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_datagram_size = max_datagram_size
        self._parser = parser
        self._socket: socket.socket | None = sock
        self._address = sock.getsockname()

    @property
    def address(self) -> SocketAddress:
        """Return the actual bound address, including an ephemeral port if requested."""

        host, port = self._address[:2]
        return str(host), int(port)

    @property
    def closed(self) -> bool:
        return self._socket is None

    def close(self) -> None:
        """Close the UDP socket. Safe to call more than once."""

        sock = self._socket
        if sock is not None:
            sock.close()
            self._socket = None

    def __enter__(self) -> "UdpReceiver[ParsedPacket]":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def receive(self, *, timeout: float | None | object = _TIMEOUT_UNSET) -> UdpPacket[ParsedPacket] | None:
        """Receive and parse one datagram.

        Returns ``None`` on timeout. Parser exceptions are allowed to propagate so
        research callers can decide whether to skip, log, or fail on bad packets.
        """

        sock = self._socket
        if sock is None:
            raise RuntimeError("UDP receiver is closed")

        socket_timeout = self.timeout if timeout is _TIMEOUT_UNSET else timeout
        sock.settimeout(socket_timeout)

        try:
            raw, address = sock.recvfrom(self.max_datagram_size)
        except TimeoutError:
            return None

        parser = _resolve_parser(self._parser)
        host, port = address[:2]
        return UdpPacket(
            raw=bytes(raw),
            parsed=parser(bytes(raw)),
            address=(str(host), int(port)),
            received_at=time.time(),
        )


def _packet_bytes_from_item(item: PacketBytesSource) -> tuple[bytes, Path | None]:
    if isinstance(item, bytes):
        return item, None
    if isinstance(item, (bytearray, memoryview)):
        return bytes(item), None

    path = Path(item)
    return path.read_bytes(), path


def iter_packet_bytes(source: ReplaySource) -> Iterator[bytes]:
    """Yield datagram bytes from a path, bytes object, or iterable of either."""

    if isinstance(source, (str, bytes, bytearray, memoryview, Path)):
        yield _packet_bytes_from_item(source)[0]
        return

    for item in source:
        yield _packet_bytes_from_item(item)[0]


def replay_packets(
    source: ReplaySource,
    *,
    parser: PacketParser[ParsedPacket] | None = None,
) -> Iterator[ReplayPacket[ParsedPacket]]:
    """Parse packet bytes from disk or an in-memory iterable without opening a socket."""

    resolved_parser = _resolve_parser(parser)

    if isinstance(source, (str, bytes, bytearray, memoryview, Path)):
        raw, path = _packet_bytes_from_item(source)
        yield ReplayPacket(raw=raw, parsed=resolved_parser(raw), index=0, path=path)
        return

    for index, item in enumerate(source):
        raw, path = _packet_bytes_from_item(item)
        yield ReplayPacket(raw=raw, parsed=resolved_parser(raw), index=index, path=path)
