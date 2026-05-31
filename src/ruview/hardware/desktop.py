"""Pure desktop helper models for RuView hardware research tools.

This module intentionally does not enumerate devices, open serial ports, spawn
sidecars, flash firmware, or perform network I/O.  It mirrors the deterministic
planning pieces from the Tauri desktop helpers so unit tests and notebooks can
inspect what would be sent to the desktop layer.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

PROVISION_BAUD = 115_200
SERIAL_TIMEOUT_MS = 5_000
PROVISION_MAGIC = b"RUVIEW_NVS"
PROVISION_VERSION = 1
PROVISION_HEADER_SIZE = 15
PROVISION_CHUNK_SIZE = 256

DEFAULT_SERVER_BIN = "sensing-server"
DEFAULT_SERVER_SOURCE = "simulate"
WASM_PORT = 8_033
WASM_TIMEOUT_SECS = 30

_ESP32_VID_PID: Mapping[int, frozenset[int] | None] = {
    0x10C4: frozenset({0xEA60, 0xEA70}),  # Silicon Labs CP210x
    0x1A86: frozenset({0x7523, 0x5523}),  # QinHeng CH340/CH341
    0x0403: frozenset({0x6001, 0x6010, 0x6011, 0x6014, 0x6015}),  # FTDI
    0x303A: None,  # Espressif ESP32-S2/S3/C3/C6 native USB VID
}

_NVS_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("wifi_ssid", "wifi_ssid", "str"),
    ("wifi_password", "wifi_pass", "str"),
    ("target_ip", "target_ip", "str"),
    ("target_port", "target_port", "u16"),
    ("node_id", "node_id", "u8"),
    ("tdm_slot", "tdm_slot", "u8"),
    ("tdm_total", "tdm_total", "u8"),
    ("edge_tier", "edge_tier", "u8"),
    ("presence_thresh", "presence_th", "u16"),
    ("fall_thresh", "fall_th", "u16"),
    ("vital_window", "vital_win", "u16"),
    ("vital_interval_ms", "vital_int", "u16"),
    ("top_k_count", "top_k", "u8"),
    ("hop_count", "hop_count", "u8"),
    ("channel_list", "channels", "channels"),
    ("power_duty", "power_duty", "u8"),
    ("wasm_max_modules", "wasm_max", "u8"),
    ("wasm_verify", "wasm_verify", "bool_u8"),
    ("ota_psk", "ota_psk", "str"),
)


def _coerce_u16(value: int | str) -> int:
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


def is_esp32_compatible(vid: int | str | None, pid: int | str | None) -> bool:
    """Return whether a USB VID/PID is a known ESP32-compatible adapter."""

    if vid is None or pid is None:
        return False
    normalized_vid = _coerce_u16(vid)
    normalized_pid = _coerce_u16(pid)
    known_pids = _ESP32_VID_PID.get(normalized_vid)
    return normalized_vid in _ESP32_VID_PID if known_pids is None else normalized_pid in known_pids


def _looks_like_usb_serial_path(name: str) -> bool:
    lowered = PurePosixPath(name).name.lower()
    return "usb" in lowered or lowered.startswith("ttyacm")


@dataclass(frozen=True)
class SerialPortInfo:
    """Metadata for one local serial port candidate."""

    name: str
    vid: int | None = None
    pid: int | None = None
    manufacturer: str | None = None
    serial_number: str | None = None
    is_esp32_compatible: bool | None = None

    def __post_init__(self) -> None:
        compatible = self.is_esp32_compatible
        if compatible is None:
            compatible = is_esp32_compatible(self.vid, self.pid)
            if not compatible and self.vid is None and self.pid is None:
                compatible = _looks_like_usb_serial_path(self.name)
        object.__setattr__(self, "is_esp32_compatible", bool(compatible))

    @property
    def vid_hex(self) -> str | None:
        return None if self.vid is None else f"0x{self.vid:04X}"

    @property
    def pid_hex(self) -> str | None:
        return None if self.pid is None else f"0x{self.pid:04X}"


def sort_serial_ports(ports: Iterable[SerialPortInfo]) -> list[SerialPortInfo]:
    """Return serial ports with ESP32-compatible candidates first."""

    return sorted(ports, key=lambda port: (not bool(port.is_esp32_compatible), port.name))


def _platform_key(platform: str | None) -> str:
    key = (platform or sys.platform).lower()
    if key.startswith("darwin") or key.startswith("mac"):
        return "macos"
    if key.startswith("linux"):
        return "linux"
    return key


def fallback_serial_port_candidates(
    names: Iterable[str],
    *,
    platform: str | None = None,
) -> list[SerialPortInfo]:
    """Filter supplied ``/dev`` entry names into fallback serial candidates.

    The caller supplies names from a fixture or from an explicit directory read.
    This helper never reads ``/dev`` itself.
    """

    platform_key = _platform_key(platform)
    candidates: list[SerialPortInfo] = []
    for item in names:
        name = str(item)
        basename = PurePosixPath(name).name
        if platform_key == "macos":
            matched = basename.startswith(("cu.usb", "tty.usb"))
        elif platform_key == "linux":
            matched = basename.startswith(("ttyUSB", "ttyACM"))
        else:
            matched = False
        if not matched:
            continue
        path = name if name.startswith("/dev/") else f"/dev/{basename}"
        candidates.append(
            SerialPortInfo(
                name=path,
                manufacturer="USB Serial",
                is_esp32_compatible=True,
            )
        )
    return sort_serial_ports(candidates)


def fallback_serial_port_names(names: Iterable[str], *, platform: str | None = None) -> list[str]:
    """Return only the fallback path strings for supplied device entry names."""

    return [candidate.name for candidate in fallback_serial_port_candidates(names, platform=platform)]


@dataclass(frozen=True, repr=False)
class WifiSerialCommandPlan:
    """Serial command plan for configuring WiFi without opening the port."""

    port: str
    ssid: str
    password: str = field(repr=False)
    baud: int = PROVISION_BAUD
    ready_delay_ms: int = 500
    response_wait_ms: int = 500
    commands: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        commands = self.commands or (
            f"wifi_config {self.ssid} {self.password}\r\n",
            f"wifi {self.ssid} {self.password}\r\n",
            f"set ssid {self.ssid}\r\n",
            f"set password {self.password}\r\n",
            "reboot\r\n",
        )
        object.__setattr__(self, "commands", tuple(commands))

    @property
    def redacted_commands(self) -> tuple[str, ...]:
        return tuple(command.replace(self.password, "<redacted>") for command in self.commands)

    def __repr__(self) -> str:
        return (
            "WifiSerialCommandPlan("
            f"port={self.port!r}, ssid={self.ssid!r}, password='<redacted>', "
            f"baud={self.baud!r}, commands={self.redacted_commands!r})"
        )


def build_wifi_serial_command_plan(
    port: str,
    ssid: str,
    password: str,
    *,
    baud: int = PROVISION_BAUD,
) -> WifiSerialCommandPlan:
    """Build the serial command strings the desktop app would try."""

    if "\n" in ssid or "\r" in ssid or "\n" in password or "\r" in password:
        raise ValueError("ssid and password must not contain newline characters")
    return WifiSerialCommandPlan(port=port, ssid=ssid, password=password, baud=baud)


@dataclass(frozen=True)
class ProvisioningConfig:
    """Subset of NVS provisioning fields mirrored from the desktop command."""

    wifi_ssid: str | None = None
    wifi_password: str | None = field(default=None, repr=False)
    target_ip: str | None = None
    target_port: int | None = None
    node_id: int | None = None
    tdm_slot: int | None = None
    tdm_total: int | None = None
    edge_tier: int | None = None
    presence_thresh: int | None = None
    fall_thresh: int | None = None
    vital_window: int | None = None
    vital_interval_ms: int | None = None
    top_k_count: int | None = None
    hop_count: int | None = None
    channel_list: Sequence[int] | None = None
    power_duty: int | None = None
    wasm_max_modules: int | None = None
    wasm_verify: bool | None = None
    ota_psk: str | None = field(default=None, repr=False)

    def value_for(self, field_name: str) -> Any:
        return getattr(self, field_name)


def _config_value(config: ProvisioningConfig | Mapping[str, Any], field_name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(field_name)
    return config.value_for(field_name)


def _write_nvs_field(data: bytearray, key: str, value_bytes: bytes) -> None:
    key_bytes = key.encode("utf-8")
    if len(key_bytes) > 255:
        raise ValueError(f"NVS key is too long: {key!r}")
    if len(value_bytes) > 65_535:
        raise ValueError(f"NVS value for {key!r} is too large")
    data.append(len(key_bytes))
    data.extend(key_bytes)
    data.extend(len(value_bytes).to_bytes(2, "little"))
    data.extend(value_bytes)


def _encode_nvs_value(key: str, value: Any, kind: str) -> bytes:
    if kind == "str":
        return str(value).encode("utf-8")
    if kind == "u8":
        integer = int(value)
        if not 0 <= integer <= 0xFF:
            raise ValueError(f"{key} must fit in u8")
        return bytes([integer])
    if kind == "u16":
        integer = int(value)
        if not 0 <= integer <= 0xFFFF:
            raise ValueError(f"{key} must fit in u16")
        return integer.to_bytes(2, "little")
    if kind == "bool_u8":
        return bytes([1 if bool(value) else 0])
    if kind == "channels":
        return ",".join(str(int(channel)) for channel in value).encode("utf-8")
    raise ValueError(f"unsupported NVS field kind: {kind}")


def serialize_nvs_config(config: ProvisioningConfig | Mapping[str, Any]) -> bytes:
    """Serialize provisioning fields to the length-prefixed NVS blob format."""

    data = bytearray()
    for field_name, nvs_key, kind in _NVS_FIELDS:
        value = _config_value(config, field_name)
        if value is None:
            continue
        _write_nvs_field(data, nvs_key, _encode_nvs_value(nvs_key, value, kind))
    data.append(0)
    return bytes(data)


def build_provision_header(size: int, *, version: int = PROVISION_VERSION) -> bytes:
    """Build the packed ``RUVIEW_NVS`` provisioning header."""

    if size < 0:
        raise ValueError("size must be non-negative")
    if not 0 <= version <= 0xFF:
        raise ValueError("version must fit in u8")
    return PROVISION_MAGIC + bytes([version]) + int(size).to_bytes(4, "little")


def nvs_checksum(nvs_data: bytes) -> str:
    """Return the desktop protocol checksum: first 8 SHA-256 bytes as hex."""

    return hashlib.sha256(nvs_data).digest()[:8].hex()


def chunk_bytes(data: bytes, chunk_size: int = PROVISION_CHUNK_SIZE) -> tuple[bytes, ...]:
    """Split bytes into serial provisioning chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return tuple(data[index : index + chunk_size] for index in range(0, len(data), chunk_size))


@dataclass(frozen=True, repr=False)
class NvsProvisioningPlan:
    """Header, chunks, and checksum for a serial NVS provisioning session."""

    nvs_data: bytes = field(repr=False)
    header: bytes
    checksum: str
    chunks: tuple[bytes, ...] = field(repr=False)
    chunk_size: int = PROVISION_CHUNK_SIZE
    version: int = PROVISION_VERSION

    @property
    def size(self) -> int:
        return len(self.nvs_data)

    @property
    def checksum_line(self) -> bytes:
        return f"{self.checksum}\n".encode("ascii")

    def __repr__(self) -> str:
        return (
            "NvsProvisioningPlan("
            f"size={self.size!r}, checksum={self.checksum!r}, "
            f"chunk_count={len(self.chunks)!r}, chunk_size={self.chunk_size!r})"
        )


def build_nvs_provisioning_plan(
    config: ProvisioningConfig | Mapping[str, Any],
    *,
    chunk_size: int = PROVISION_CHUNK_SIZE,
) -> NvsProvisioningPlan:
    """Build a deterministic NVS provisioning plan without serial I/O."""

    nvs_data = serialize_nvs_config(config)
    return NvsProvisioningPlan(
        nvs_data=nvs_data,
        header=build_provision_header(len(nvs_data)),
        checksum=nvs_checksum(nvs_data),
        chunks=chunk_bytes(nvs_data, chunk_size),
        chunk_size=chunk_size,
    )


@dataclass(frozen=True)
class ServerConfig:
    """Configuration for a sensing-server sidecar command descriptor."""

    http_port: int | None = None
    ws_port: int | None = None
    udp_port: int | None = None
    log_level: str | None = None
    bind_address: str | None = None
    server_path: str | None = None
    source: str | None = None

    def args(self) -> tuple[str, ...]:
        args: list[str] = []
        if self.http_port is not None:
            args.extend(("--http-port", str(self.http_port)))
        if self.ws_port is not None:
            args.extend(("--ws-port", str(self.ws_port)))
        if self.udp_port is not None:
            args.extend(("--udp-port", str(self.udp_port)))
        if self.bind_address is not None:
            args.extend(("--bind", self.bind_address))
        if self.log_level is not None:
            args.extend(("--log-level", self.log_level))
        args.extend(("--source", self.source or DEFAULT_SERVER_SOURCE))
        return tuple(args)


@dataclass(frozen=True)
class ServerSidecarDescriptor:
    """Pure description of the sensing server sidecar invocation."""

    binary: str
    args: tuple[str, ...]
    http_port: int | None = None
    ws_port: int | None = None
    udp_port: int | None = None
    source: str = DEFAULT_SERVER_SOURCE

    @property
    def argv(self) -> tuple[str, ...]:
        return (self.binary, *self.args)


def build_server_sidecar_descriptor(
    config: ServerConfig | None = None,
    *,
    binary: str | None = None,
) -> ServerSidecarDescriptor:
    """Build a command descriptor without checking paths or spawning a process."""

    config = config or ServerConfig()
    executable = binary or config.server_path or DEFAULT_SERVER_BIN
    return ServerSidecarDescriptor(
        binary=executable,
        args=config.args(),
        http_port=config.http_port,
        ws_port=config.ws_port,
        udp_port=config.udp_port,
        source=config.source or DEFAULT_SERVER_SOURCE,
    )


@dataclass(frozen=True)
class WasmModuleInfo:
    id: str
    name: str
    size_bytes: int
    status: str
    sha256: str | None = None
    loaded_at: str | None = None
    memory_used_kb: int | None = None
    cpu_usage_pct: float | None = None
    exec_count: int | None = None


@dataclass(frozen=True)
class WasmModuleDetail:
    id: str
    name: str
    size_bytes: int
    status: str
    sha256: str
    loaded_at: str
    memory_used_kb: int
    exports: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    execution_count: int = 0
    last_error: str | None = None


@dataclass(frozen=True)
class WasmUploadResult:
    success: bool
    module_id: str
    message: str
    sha256: str | None = None


@dataclass(frozen=True)
class WasmControlResult:
    success: bool
    module_id: str
    action: str
    message: str


@dataclass(frozen=True)
class WasmRuntimeStats:
    total_modules: int
    running_modules: int
    memory_used_kb: int
    memory_limit_kb: int
    total_executions: int
    errors: int


@dataclass(frozen=True)
class WasmSupportInfo:
    supported: bool
    max_modules: int | None = None
    memory_limit_kb: int | None = None
    verify_signatures: bool = False


@dataclass(frozen=True)
class WasmNodeEndpoint:
    """Pure URL descriptor for a node's WASM management endpoints."""

    node_ip: str
    port: int = WASM_PORT
    scheme: str = "http"

    def url(self, path: str) -> str:
        normalized_path = path if path.startswith("/") else f"/{path}"
        return f"{self.scheme}://{self.node_ip}:{self.port}{normalized_path}"

    @property
    def list_url(self) -> str:
        return self.url("/wasm/list")

    @property
    def upload_url(self) -> str:
        return self.url("/wasm/upload")

    def module_url(self, module_id: str, action: str | None = None) -> str:
        suffix = f"/wasm/{module_id}" if action is None else f"/wasm/{module_id}/{action}"
        return self.url(suffix)

    @property
    def stats_url(self) -> str:
        return self.url("/wasm/stats")

    @property
    def info_url(self) -> str:
        return self.url("/wasm/info")


def wasm_sha256(data: bytes) -> str:
    """Return SHA-256 for a WASM byte payload."""

    return hashlib.sha256(data).hexdigest()


def has_wasm_magic(data: bytes) -> bool:
    """Return whether bytes start with the WebAssembly magic header."""

    return len(data) >= 4 and data[:4] == b"\0asm"


__all__ = [
    "DEFAULT_SERVER_BIN",
    "DEFAULT_SERVER_SOURCE",
    "NvsProvisioningPlan",
    "PROVISION_BAUD",
    "PROVISION_CHUNK_SIZE",
    "PROVISION_HEADER_SIZE",
    "PROVISION_MAGIC",
    "PROVISION_VERSION",
    "SERIAL_TIMEOUT_MS",
    "ServerConfig",
    "ServerSidecarDescriptor",
    "SerialPortInfo",
    "WASM_PORT",
    "WASM_TIMEOUT_SECS",
    "WasmControlResult",
    "WasmModuleDetail",
    "WasmModuleInfo",
    "WasmNodeEndpoint",
    "WasmRuntimeStats",
    "WasmSupportInfo",
    "WasmUploadResult",
    "WifiSerialCommandPlan",
    "build_nvs_provisioning_plan",
    "build_provision_header",
    "build_server_sidecar_descriptor",
    "build_wifi_serial_command_plan",
    "chunk_bytes",
    "fallback_serial_port_candidates",
    "fallback_serial_port_names",
    "has_wasm_magic",
    "is_esp32_compatible",
    "nvs_checksum",
    "serialize_nvs_config",
    "sort_serial_ports",
    "wasm_sha256",
]
