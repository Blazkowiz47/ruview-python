# ESP32 Protocol Source Map

Milestone 2 tracks the Python port of the ESP32 UDP packet parser. This file records
the current wire formats from the RuView reference tree so parser work can stay
byte-for-byte aligned with firmware.

All multi-byte numeric fields below are little-endian. The ESP32 firmware writes
fixed-width C integers with `memcpy` on a little-endian target; packed structs are
also little-endian on the current ESP32 targets. I/Q bytes are signed int8 pairs
ordered as `I, Q`.

## Authoritative Sources

| Source | Why it matters |
|---|---|
| `/Users/sushrutpatwardhan/1Projects/RuView/firmware/esp32-csi-node/main/csi_collector.h:14` | `CSI_MAGIC`, `CSI_HEADER_SIZE`, and max CSI frame size. |
| `/Users/sushrutpatwardhan/1Projects/RuView/firmware/esp32-csi-node/main/csi_collector.c:107` | Authoritative raw CSI header layout and serializer. |
| `/Users/sushrutpatwardhan/1Projects/RuView/firmware/esp32-csi-node/main/csi_collector.c:306` | Authoritative ADR-110 sync-packet construction. |
| `/Users/sushrutpatwardhan/1Projects/RuView/firmware/esp32-csi-node/main/edge_processing.h:94` | Packed edge vitals, feature-vector, and fused-vitals structs. |
| `/Users/sushrutpatwardhan/1Projects/RuView/firmware/esp32-csi-node/main/edge_processing.c:556` | Edge vitals/fused packet population and send path. |
| `/Users/sushrutpatwardhan/1Projects/RuView/firmware/esp32-csi-node/main/wasm_runtime.h:46` | WASM output magic and packed event packet structs. |
| `/Users/sushrutpatwardhan/1Projects/RuView/firmware/esp32-csi-node/main/wasm_runtime.c:346` | WASM event dead-band filtering and variable-length send size. |
| `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-hardware/src/esp32_parser.rs:39` | Rust ADR-018 parser and sibling-packet magic registry. |
| `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-hardware/src/sync_packet.rs:10` | Rust ADR-110 sync-packet decoder and canonical behavior. |
| `/Users/sushrutpatwardhan/1Projects/RuView/archive/v1/tests/unit/test_esp32_binary_parser.py:26` | Archived Python parser tests with ADR-110 byte 18/19 and sync parity vectors. |

Some older host paths are not authoritative for the current wire format. In
particular, `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-sensing-server/src/main.rs:1167`
documents the current 20-byte header but its parsing code reads several fields at
stale offsets. Prefer the firmware plus `wifi-densepose-hardware` parser as the
source of truth.

## UDP Magic Dispatch

The ESP32 firmware multiplexes several packet families on the same UDP socket.
Dispatch should start with the first four bytes as a little-endian `u32`.

| Magic | Packet | Current status |
|---:|---|---|
| `0xC5110001` | Raw ADR-018 CSI | Known, exact layout below. |
| `0xC511A110` | ADR-110 sync packet | Known, exact layout below. |
| `0xC5110002` | Edge vitals | Known packed 32-byte layout below. |
| `0xC5110003` | Edge feature vector | Known packed 48-byte sibling packet; not Milestone 2 parser priority unless routing siblings. |
| `0xC5110004` | Fused vitals or WASM output | Ambiguous collision in current firmware; see TODO below. |
| `0xC5110005` | Compressed CSI | Known header, delta/RLE payload; parser can route/skip initially. |
| `0xC5110006` | Feature state | Separate feature-state protocol, not covered here. |
| `0xC5110007` | Temporal classification | Sibling packet, not covered here. |

## Raw CSI Packet

Magic: `0xC5110001`. Wire bytes start `01 00 11 c5`.

| Offset | Size | Type | Field | Notes |
|---:|---:|---|---|---|
| 0 | 4 | `u32` | `magic` | Must equal `0xC5110001`. |
| 4 | 1 | `u8` | `node_id` | Defensive copy from NVS via `csi_collector_get_node_id()`. |
| 5 | 1 | `u8` | `n_antennas` | Firmware currently writes `1`; format allows more. |
| 6 | 2 | `u16` | `n_subcarriers` | Computed as `info->len / (2 * n_antennas)`. |
| 8 | 4 | `u32` | `freq_mhz` | Derived from WiFi channel: ch 1-13 => `2412 + 5*(ch-1)`, ch 14 => `2484`, ch 36-177 => `5000 + 5*ch`, else `0`. |
| 12 | 4 | `u32` | `sequence` | `s_sequence++` at serialization time. |
| 16 | 1 | `i8` | `rssi_dbm` | `info->rx_ctrl.rssi`. |
| 17 | 1 | `i8` | `noise_floor_dbm` | `info->rx_ctrl.noise_floor`. |
| 18 | 1 | `u8` | `ppdu_type` | ADR-110 extension when `CONFIG_CSI_FRAME_HE_TAGGING`; otherwise `0`. |
| 19 | 1 | `u8` | `flags` | ADR-110 extension when `CONFIG_CSI_FRAME_HE_TAGGING`; otherwise `0`. |
| 20 | `N` | `i8` pairs | `iq_data` | Raw `wifi_csi_info_t.buf`; length is `n_antennas * n_subcarriers * 2`. |

`ppdu_type` values:

| Value | Meaning |
|---:|---|
| `0` | HT/legacy bucket. Firmware also collapses non-HT, HT, VHT, and HE-ER-SU fallback here as documented in source. |
| `1` | HE-SU. |
| `2` | HE-MU. |
| `3` | HE-TB. |
| `0xff` | Unknown. |

`flags` bits:

| Bit | Mask | Meaning | Firmware population |
|---:|---:|---|---|
| 0 | `0x01` | 40 MHz bandwidth observed | Set from C6 `rx_ctrl.second != 0` or legacy `rx_ctrl.cwb`. |
| 1 | `0x02` | Reserved for future wide-band encoding | Not populated by firmware. |
| 2 | `0x04` | STBC | Set on legacy targets from `rx_ctrl.stbc`; not populated on the C6 branch today. |
| 3 | `0x08` | LDPC | Host structs reserve it; firmware does not populate it today. |
| 4 | `0x10` | Cross-node sync valid | Set when either `c6_timesync_is_valid()` or `c6_sync_espnow_is_valid()` is true. |
| 5-7 | `0xe0` | Reserved | Must be ignored by parsers. |

Parser notes:

- Reject packets shorter than 20 bytes.
- Validate `len >= 20 + n_antennas * n_subcarriers * 2`.
- Interpret payload bytes as signed int8 values, then compute amplitude and phase
  as `sqrt(I*I + Q*Q)` and `atan2(Q, I)` when needed.
- There is no per-frame local timestamp in ADR-018 v1. Mesh-aligned timing is
  recovered from sync packets keyed by `(node_id, sequence)`.

## Sync Packet

Magic: `0xC511A110`. Wire bytes start `10 a1 11 c5`.

Firmware emits this packet every `CONFIG_C6_SYNC_EVERY_N_FRAMES` CSI callbacks
after the raw CSI serialization path. The default is 20 callbacks.

| Offset | Size | Type | Field | Notes |
|---:|---:|---|---|---|
| 0 | 4 | `u32` | `magic` | Must equal `0xC511A110`. |
| 4 | 1 | `u8` | `node_id` | Same node identifier as raw CSI. |
| 5 | 1 | `u8` | `proto_ver` | Currently `0x01`. |
| 6 | 1 | `u8` | `flags` | See below. |
| 7 | 1 | `u8` | reserved | Firmware writes zero. |
| 8 | 8 | `u64` | `local_us` | `esp_timer_get_time()` at emission time. |
| 16 | 8 | `u64` | `epoch_us` | `c6_sync_espnow_get_epoch_us()`. |
| 24 | 4 | `u32` | `sequence` | Current `s_sequence` high-water mark for host pairing. |
| 28 | 4 | `u32` | reserved | Firmware writes zero; comment reserves space for leader-id low32. |

`flags` bits:

| Bit | Mask | Meaning |
|---:|---:|---|
| 0 | `0x01` | Node is sync leader. |
| 1 | `0x02` | Sync is valid/fresh. |
| 2 | `0x04` | Smoothed offset value is nonzero in firmware. Host references call this `smoothed_used`. |
| 3-7 | `0xf8` | Reserved. |

Known canonical sync bytes from the Rust/Python parity tests:

```text
10a111c509010600f26db70100000000c5aca501000000001400000000000000
```

Decoded:

- `node_id=9`, `proto_ver=1`, `flags=0x06`
- `local_us=28_798_450`, `epoch_us=27_634_885`
- `sequence=20`, `local_us - epoch_us = 1_163_565`

## Edge Vitals Packet

Magic: `0xC5110002`. Fixed size: 32 bytes. Source struct:
`edge_vitals_pkt_t`.

| Offset | Size | Type | Field | Notes |
|---:|---:|---|---|---|
| 0 | 4 | `u32` | `magic` | `0xC5110002`. |
| 4 | 1 | `u8` | `node_id` | Runtime node id. |
| 5 | 1 | `u8` | `flags` | Bit0 presence, bit1 fall, bit2 motion. |
| 6 | 2 | `u16` | `breathing_rate` | BPM * 100. |
| 8 | 4 | `u32` | `heartrate` | BPM * 10000. |
| 12 | 1 | `i8` | `rssi` | Latest CSI RSSI. |
| 13 | 1 | `u8` | `n_persons` | Count of active person groups. |
| 14 | 2 | `u8[2]` | reserved | Zeroed by `memset`. |
| 16 | 4 | `f32` | `motion_energy` | ESP32 IEEE-754 little-endian float in practice. |
| 20 | 4 | `f32` | `presence_score` | ESP32 IEEE-754 little-endian float in practice. |
| 24 | 4 | `u32` | `timestamp_ms` | `esp_timer_get_time() / 1000`. |
| 28 | 4 | `u32` | reserved2 | Zeroed by `memset`. |

Edge vitals are sent at `s_cfg.vital_interval_ms`. At the same cadence the
firmware also sends the ADR-069 feature vector (`0xC5110003`).

## WASM Output Packet

Magic: `0xC5110004`. Variable size: `8 + event_count * 5` bytes. Source struct:
`wasm_output_pkt_t`.

| Offset | Size | Type | Field | Notes |
|---:|---:|---|---|---|
| 0 | 4 | `u32` | `magic` | `0xC5110004`. |
| 4 | 1 | `u8` | `node_id` | Runtime node id. |
| 5 | 1 | `u8` | `module_id` | WASM module slot id. |
| 6 | 2 | `u16` | `event_count` | Number of following events after dead-band filtering. |
| 8 | 5 each | packed event | `events` | Repeated `event_count` times. |

Each event is:

| Event offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 1 | `u8` | `event_type` |
| 1 | 4 | `f32` | `value` |

The firmware filters repeated values with a 5 percent dead-band before sending.
It sends only the used prefix of the struct, not the full `WASM_MAX_EVENTS`
array. WASM output is compiled only when `CONFIG_WASM_ENABLE` and
`WASM3_AVAILABLE` are both defined.

## Known Ambiguities And TODOs

- `0xC5110004` currently collides: `edge_processing.h` assigns it to fused
  vitals, while `wasm_runtime.h` assigns it to WASM output. Their layouts differ.
  Parser TODO: dispatch cannot rely on magic alone for this value. Check packet
  length and fields conservatively, or ask firmware worker to resolve the magic.
- Edge/fused/WASM float fields are written as C `float`. ESP32 targets are
  little-endian IEEE-754, so Python should use `struct.unpack_from("<f", ...)`;
  keep a test fixture to guard this assumption.
- Raw CSI byte 19 bit 4 says a sync source is valid, but ADR-018 raw frames still
  carry no local timestamp. Host-side mesh alignment must use the sibling sync
  packet plus sequence interpolation.
- Older host parsers in the sensing server and pointcloud crate have stale field
  offsets or alternate v6 assumptions. The Python port should mirror the firmware
  and `wifi-densepose-hardware` parser, then optionally add compatibility shims
  only with explicit tests.
