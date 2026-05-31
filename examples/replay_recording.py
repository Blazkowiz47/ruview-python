"""Replay JSONL sensing recordings and print compact update summaries.

The example accepts two lightweight line-oriented shapes:

* full sensing updates with ``type/msg_type``, ``source``, ``tick``, ``nodes``,
  ``features``, ``classification``, ``signal_field``, ``vital_signs``, and
  ``estimated_persons`` fields;
* recorded-frame rows with fields such as ``timestamp``, ``subcarriers``,
  ``rssi``, ``noise_floor``, and ``features``.

When ``ruview.server.sources.ReplaySensingSource`` is available, the CLI tries
that source first. Until the Milestone 7 server package lands, it falls back to
the local JSONL reader below.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


UPDATE_KEYS = {
    "type",
    "msg_type",
    "source",
    "tick",
    "nodes",
    "features",
    "classification",
    "signal_field",
    "vital_signs",
    "estimated_persons",
}


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    return {}


def _flatten_numbers(value: Any) -> list[float]:
    numbers: list[float] = []

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            numbers.append(float(value))
        return numbers

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        for item in value:
            numbers.extend(_flatten_numbers(item))

    return numbers


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "present"}:
            return True
        if lowered in {"0", "false", "no", "absent"}:
            return False
    return None


def _looks_like_update(row: Mapping[str, Any]) -> bool:
    msg_type = row.get("type", row.get("msg_type"))
    if msg_type == "sensing_update":
        return True
    return len(UPDATE_KEYS.intersection(row)) >= 3


def _classification_from_record(record: Mapping[str, Any], features: Mapping[str, Any]) -> dict[str, Any]:
    presence = _first(record, "presence", "person_present")
    presence_bool = _as_bool(presence)
    if presence is None:
        presence_score = _as_float(_first(features, "presence_score", "presence"))
        presence_bool = None if presence_score is None else presence_score >= 0.5

    confidence = _first(record, "confidence")
    if confidence is None:
        confidence = _first(features, "confidence", "presence_score")

    motion_level = _first(record, "motion_level", "motion")
    if motion_level is None:
        motion_energy = _as_float(_first(features, "motion_energy", "motion_score"))
        if motion_energy is not None:
            if motion_energy >= 0.65:
                motion_level = "active"
            elif motion_energy >= 0.25:
                motion_level = "moving"
            elif presence_bool:
                motion_level = "still"
            else:
                motion_level = "absent"
        else:
            motion_level = "unknown"

    return {
        "motion_level": motion_level,
        "presence": False if presence_bool is None else presence_bool,
        "confidence": 0.0 if confidence is None else confidence,
    }


def _vitals_from_record(record: Mapping[str, Any], features: Mapping[str, Any]) -> dict[str, Any]:
    vital_signs = _to_mapping(_first(record, "vital_signs", "vitals"))
    breathing = _first(record, "breathing_rate_bpm", "breathing_bpm")
    heart = _first(record, "heart_rate_bpm", "heartrate_bpm", "hr_bpm")
    if breathing is None:
        breathing = _first(features, "breathing_rate_bpm", "breathing_bpm")
    if heart is None:
        heart = _first(features, "heart_rate_bpm", "heartrate_bpm", "hr_bpm")

    if breathing is not None:
        vital_signs.setdefault("breathing_rate_bpm", breathing)
    if heart is not None:
        vital_signs.setdefault("heart_rate_bpm", heart)
    return vital_signs


def _record_to_update(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    features = _to_mapping(record.get("features"))
    subcarriers = _flatten_numbers(_first(record, "subcarriers", "amplitudes", "amplitude"))
    tick = _first(record, "tick", "sequence", "sequence_number")

    if subcarriers:
        mean_amplitude = sum(abs(value) for value in subcarriers) / len(subcarriers)
        variance = sum((value - mean_amplitude) ** 2 for value in subcarriers) / len(subcarriers)
        features.setdefault("mean_amplitude", mean_amplitude)
        features.setdefault("variance", variance)

    node_id = _first(record, "node_id", "device_id")
    node: dict[str, Any] = {
        "node_id": 0 if node_id is None else node_id,
        "rssi_dbm": _first(record, "rssi_dbm", "rssi"),
        "noise_floor_dbm": _first(record, "noise_floor_dbm", "noise_floor"),
        "subcarrier_count": len(subcarriers),
    }
    if 0 < len(subcarriers) <= 8:
        node["amplitude"] = subcarriers

    classification = _classification_from_record(record, features)
    estimated_persons = _first(record, "estimated_persons", "n_persons", "person_count")
    if estimated_persons is None and classification["presence"]:
        estimated_persons = 1
    if estimated_persons is None:
        estimated_persons = 0

    return {
        "type": "sensing_update",
        "timestamp": _first(record, "timestamp", "ts", "time"),
        "source": _first(record, "source") or "replay",
        "tick": index + 1 if tick is None else tick,
        "nodes": [node],
        "features": features,
        "classification": classification,
        "signal_field": _to_mapping(record.get("signal_field")),
        "vital_signs": _vitals_from_record(record, features),
        "estimated_persons": estimated_persons,
    }


def _normalise_row(row: Any, index: int) -> dict[str, Any]:
    mapping = _to_mapping(row)
    if not mapping:
        return {"type": "record", "source": "replay", "tick": index, "value": repr(row)}

    if "update" in mapping:
        return _normalise_row(mapping["update"], index)
    if "record" in mapping:
        return _record_to_update(_to_mapping(mapping["record"]), index)
    if _looks_like_update(mapping):
        update = dict(mapping)
        if "type" not in update and update.get("msg_type") == "sensing_update":
            update["type"] = update["msg_type"]
        update.setdefault("source", "replay")
        update.setdefault("tick", index + 1)
        return update
    return _record_to_update(mapping, index)


def _open_jsonl(path_text: str):
    if path_text == "-":
        return sys.stdin
    return Path(path_text).open("r", encoding="utf-8")


def _iter_jsonl(path_text: str, *, skip_errors: bool) -> Iterable[dict[str, Any]]:
    with _open_jsonl(path_text) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                message = f"{path_text}:{line_number}: invalid JSON: {exc.msg}"
                if skip_errors:
                    print(message, file=sys.stderr)
                    continue
                raise SystemExit(message) from exc


def _load_replay_source_class() -> type[Any] | None:
    try:
        from ruview.server.sources import ReplaySensingSource
    except (ImportError, AttributeError):
        return None
    return ReplaySensingSource


def _construct_replay_source(source_class: type[Any], path: Path, limit: int | None) -> Any:
    constructor_attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
        ((path,), {}),
        ((), {"path": path}),
        ((), {"recording_path": path}),
        ((), {"file_path": path}),
        ((), {"source": path}),
        ((), {"path": path, "limit": limit}),
    ]

    last_error: TypeError | None = None
    for args, kwargs in constructor_attempts:
        try:
            return source_class(*args, **{key: value for key, value in kwargs.items() if value is not None})
        except TypeError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    return source_class(path)


async def _collect_async(async_iterable: Any, limit: int | None) -> list[Any]:
    items: list[Any] = []
    async for item in async_iterable:
        items.append(item)
        if limit is not None and len(items) >= limit:
            break
    return items


def _materialise_candidate(candidate: Any, limit: int | None) -> Iterable[Any] | None:
    if inspect.isawaitable(candidate):
        candidate = asyncio.run(candidate)
    if hasattr(candidate, "__aiter__"):
        return asyncio.run(_collect_async(candidate, limit))
    if isinstance(candidate, Iterable) and not isinstance(candidate, (str, bytes, bytearray, Mapping)):
        return candidate
    return None


def _iter_server_source(path_text: str, *, limit: int | None) -> Iterable[Any] | None:
    source_class = _load_replay_source_class()
    if source_class is None:
        return None

    source = _construct_replay_source(source_class, Path(path_text), limit)
    method_names = ("iter_updates", "updates", "records", "replay", "read", "iter")
    for method_name in method_names:
        method = getattr(source, method_name, None)
        if callable(method):
            iterable = _materialise_candidate(method(), limit)
            if iterable is not None:
                return iterable

    return _materialise_candidate(source, limit)


def _fmt_value(value: Any, digits: int = 2) -> str:
    number = _as_float(value)
    if number is None:
        return str(value)
    return f"{number:.{digits}f}"


def _summarise_update(update: Mapping[str, Any], index: int) -> str:
    features = _to_mapping(update.get("features"))
    classification = _to_mapping(update.get("classification"))
    vital_signs = _to_mapping(update.get("vital_signs"))
    nodes = update.get("nodes") if isinstance(update.get("nodes"), list) else []

    parts = [
        f"#{index}",
        f"tick={update.get('tick', index)}",
        f"source={update.get('source', 'replay')}",
    ]

    timestamp = _first(update, "timestamp", "ts", "time")
    if timestamp is not None:
        parts.append(f"t={_fmt_value(timestamp, 3)}")

    if nodes:
        parts.append(f"nodes={len(nodes)}")
        first_node = _to_mapping(nodes[0])
        node_id = _first(first_node, "node_id", "device_id")
        rssi = _first(first_node, "rssi_dbm", "rssi")
        subcarriers = _first(first_node, "subcarrier_count", "num_subcarriers")
        if node_id is not None:
            parts.append(f"node={node_id}")
        if rssi is not None:
            parts.append(f"rssi={_fmt_value(rssi, 1)}")
        if subcarriers is not None:
            parts.append(f"subcarriers={subcarriers}")

    presence = classification.get("presence")
    if presence is not None:
        presence_bool = _as_bool(presence)
        presence_text = str(presence if presence_bool is None else presence_bool).lower()
        parts.append(f"presence={presence_text}")

    motion_level = classification.get("motion_level")
    if motion_level is not None:
        parts.append(f"motion={motion_level}")

    confidence = classification.get("confidence")
    if confidence is not None:
        parts.append(f"conf={_fmt_value(confidence)}")

    estimated_persons = update.get("estimated_persons")
    if estimated_persons is not None:
        parts.append(f"persons={estimated_persons}")

    mean_amplitude = _first(features, "mean_amplitude", "mean_amp")
    if mean_amplitude is not None:
        parts.append(f"mean_amp={_fmt_value(mean_amplitude, 3)}")

    breathing = _first(vital_signs, "breathing_rate_bpm", "breathing_bpm")
    heart = _first(vital_signs, "heart_rate_bpm", "heartrate_bpm", "hr_bpm")
    if breathing is not None:
        parts.append(f"br={_fmt_value(breathing, 1)}bpm")
    if heart is not None:
        parts.append(f"hr={_fmt_value(heart, 1)}bpm")

    return " ".join(parts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", help="JSONL recording path, or '-' for stdin")
    parser.add_argument("--limit", type=int, help="Stop after this many rows")
    parser.add_argument("--delay", type=float, default=0.0, help="Sleep this many seconds between rows")
    parser.add_argument("--json", action="store_true", help="Print normalised compact JSON updates")
    parser.add_argument("--skip-errors", action="store_true", help="Skip invalid JSONL rows")
    parser.add_argument(
        "--source-mode",
        choices=("auto", "server", "local"),
        default="auto",
        help="Use ReplaySensingSource when available, require it, or force the local JSONL parser",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.delay < 0:
        raise SystemExit("--delay must be non-negative")
    if args.recording != "-" and not Path(args.recording).is_file():
        raise SystemExit(f"Recording not found: {args.recording}")

    rows: Iterable[Any] | None = None
    if args.source_mode in {"auto", "server"}:
        try:
            rows = _iter_server_source(args.recording, limit=args.limit)
        except Exception as exc:
            if args.source_mode == "server":
                raise SystemExit(f"ReplaySensingSource failed: {exc}") from exc
            print(f"ReplaySensingSource unavailable, using local JSONL parser: {exc}", file=sys.stderr)

    if rows is None:
        if args.source_mode == "server":
            raise SystemExit("ruview.server.sources.ReplaySensingSource is not available")
        rows = _iter_jsonl(args.recording, skip_errors=args.skip_errors)

    printed = 0
    for index, row in enumerate(rows):
        if args.limit is not None and printed >= args.limit:
            break
        update = _normalise_row(row, index)
        if args.json:
            print(json.dumps(update, separators=(",", ":"), sort_keys=True))
        else:
            print(_summarise_update(update, index))
        printed += 1
        if args.delay > 0 and (args.limit is None or printed < args.limit):
            time.sleep(args.delay)

    print(f"replayed {printed} row(s)", file=sys.stderr)


if __name__ == "__main__":
    main()
