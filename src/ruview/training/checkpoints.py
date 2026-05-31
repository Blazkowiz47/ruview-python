"""JSON checkpoint manifests for lightweight training experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


CHECKPOINT_FORMAT = "ruview.training.checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointError(ValueError):
    """Raised when a checkpoint manifest is invalid."""


@dataclass(frozen=True)
class CheckpointManifest:
    """Portable JSON manifest describing one saved training checkpoint."""

    epoch: int
    metrics: Mapping[str, float]
    config: Mapping[str, Any] = field(default_factory=dict)
    weights_path: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)
    format: str = CHECKPOINT_FORMAT
    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    manifest_path: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if int(self.epoch) < 0:
            raise CheckpointError("epoch must be non-negative")
        metrics = {str(key): float(value) for key, value in self.metrics.items()}
        if any(not np.isfinite(value) for value in metrics.values()):
            raise CheckpointError("metrics must be finite")
        object.__setattr__(self, "epoch", int(self.epoch))
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "config", _json_ready(self.config))
        object.__setattr__(self, "provenance", _json_ready(self.provenance))
        object.__setattr__(self, "extra", _json_ready(self.extra))
        if self.weights_path is not None:
            object.__setattr__(self, "weights_path", str(self.weights_path))
        if self.format != CHECKPOINT_FORMAT:
            raise CheckpointError(f"unsupported checkpoint format {self.format!r}")
        if int(self.schema_version) != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError(f"unsupported checkpoint schema {self.schema_version!r}")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format,
            "schema_version": self.schema_version,
            "epoch": self.epoch,
            "config": dict(self.config),
            "metrics": dict(self.metrics),
            "weights_path": self.weights_path,
            "provenance": dict(self.provenance),
            "extra": dict(self.extra),
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, manifest_path: str | None = None) -> "CheckpointManifest":
        return cls(
            epoch=int(payload.get("epoch", 0)),
            metrics=payload.get("metrics", {}),
            config=payload.get("config", {}),
            weights_path=payload.get("weights_path"),
            provenance=payload.get("provenance", {}),
            extra=payload.get("extra", {}),
            format=str(payload.get("format", CHECKPOINT_FORMAT)),
            schema_version=int(payload.get("schema_version", CHECKPOINT_SCHEMA_VERSION)),
            manifest_path=manifest_path,
        )

    def metric(self, name: str, default: float | None = None) -> float | None:
        if name not in self.metrics:
            return default
        return float(self.metrics[name])

    def resolved_weights_path(self) -> Path | None:
        if self.weights_path is None:
            return None
        weights = Path(self.weights_path)
        if weights.is_absolute() or self.manifest_path is None:
            return weights
        return Path(self.manifest_path).parent / weights


def save_checkpoint_manifest(
    path: str | Path,
    manifest: CheckpointManifest | None = None,
    *,
    epoch: int | None = None,
    metrics: Mapping[str, float] | None = None,
    config: Mapping[str, Any] | None = None,
    weights_path: str | Path | None = None,
    provenance: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> CheckpointManifest:
    """Write a checkpoint manifest as deterministic JSON."""

    output_path = Path(path)
    if manifest is None:
        if epoch is None or metrics is None:
            raise CheckpointError("epoch and metrics are required when manifest is not provided")
        manifest = CheckpointManifest(
            epoch=epoch,
            metrics=metrics,
            config=config or {},
            weights_path=None if weights_path is None else str(weights_path),
            provenance=provenance or {},
            extra=extra or {},
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return replace(manifest, manifest_path=str(output_path))


def load_checkpoint_manifest(path: str | Path) -> CheckpointManifest:
    """Load a JSON checkpoint manifest."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint manifest must be a JSON object")
    return CheckpointManifest.from_dict(payload, manifest_path=str(manifest_path))


def select_best_checkpoints(
    checkpoints: Iterable[CheckpointManifest | str | Path],
    *,
    metric: str = "val_pck",
    mode: str = "max",
    top_k: int = 1,
) -> tuple[CheckpointManifest, ...]:
    """Rank checkpoint manifests by a metric and return the top-k."""

    if top_k <= 0:
        return ()
    if mode not in {"max", "min"}:
        raise CheckpointError("mode must be 'max' or 'min'")

    loaded = [_load_if_path(item) for item in checkpoints]
    scored = [item for item in loaded if item.metric(metric) is not None]
    reverse_score = mode == "max"

    def sort_key(item: CheckpointManifest) -> tuple[float, int, str]:
        score = float(item.metric(metric, 0.0))
        path_key = item.manifest_path or ""
        if reverse_score:
            return (-score, -item.epoch, path_key)
        return (score, -item.epoch, path_key)

    return tuple(sorted(scored, key=sort_key)[:top_k])


def best_checkpoint(
    checkpoints: Iterable[CheckpointManifest | str | Path],
    *,
    metric: str = "val_pck",
    mode: str = "max",
) -> CheckpointManifest | None:
    """Return the single best checkpoint, or None when no metric is present."""

    selected = select_best_checkpoints(checkpoints, metric=metric, mode=mode, top_k=1)
    return selected[0] if selected else None


def _load_if_path(item: CheckpointManifest | str | Path) -> CheckpointManifest:
    if isinstance(item, CheckpointManifest):
        return item
    return load_checkpoint_manifest(item)


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
