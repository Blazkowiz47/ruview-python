"""JSON/RVF-like model export manifests for research checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from blake3 import blake3
import numpy as np


EXPORT_FORMAT = "ruview.training.model_export"
EXPORT_SCHEMA_VERSION = 1


class ModelExportError(ValueError):
    """Raised when a model export manifest is invalid."""


@dataclass(frozen=True)
class TensorSpec:
    """Tensor name, shape, dtype, and optional layout metadata."""

    name: str
    shape: tuple[int | str | None, ...]
    dtype: str = "float32"
    layout: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ModelExportError("tensor name must be non-empty")
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "dtype", str(self.dtype))
        if self.layout is not None:
            object.__setattr__(self, "layout", str(self.layout))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "layout": self.layout,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TensorSpec":
        return cls(
            name=str(payload["name"]),
            shape=tuple(payload.get("shape", ())),
            dtype=str(payload.get("dtype", "float32")),
            layout=payload.get("layout"),
        )


@dataclass(frozen=True)
class ModelExportManifest:
    """Portable JSON manifest for a model artifact and its tensor contract."""

    model_name: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    backend: str = "json"
    config: Mapping[str, Any] = field(default_factory=dict)
    weights_path: str | None = None
    weights_hash: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    rvf_compatible: bool = True
    format: str = EXPORT_FORMAT
    schema_version: int = EXPORT_SCHEMA_VERSION
    manifest_path: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ModelExportError("model_name must be non-empty")
        if not self.inputs:
            raise ModelExportError("at least one input tensor is required")
        if not self.outputs:
            raise ModelExportError("at least one output tensor is required")
        object.__setattr__(self, "inputs", tuple(_coerce_tensor_spec(item) for item in self.inputs))
        object.__setattr__(self, "outputs", tuple(_coerce_tensor_spec(item) for item in self.outputs))
        object.__setattr__(self, "config", _json_ready(self.config))
        object.__setattr__(self, "provenance", _json_ready(self.provenance))
        if self.weights_path is not None:
            object.__setattr__(self, "weights_path", str(self.weights_path))
        if self.format != EXPORT_FORMAT:
            raise ModelExportError(f"unsupported export format {self.format!r}")
        if int(self.schema_version) != EXPORT_SCHEMA_VERSION:
            raise ModelExportError(f"unsupported export schema {self.schema_version!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "rvf_compatible": bool(self.rvf_compatible),
            "model": {
                "name": self.model_name,
                "backend": self.backend,
            },
            "inputs": [item.to_dict() for item in self.inputs],
            "outputs": [item.to_dict() for item in self.outputs],
            "config": dict(self.config),
            "weights": {
                "path": self.weights_path,
                "blake3": self.weights_hash,
            },
            "provenance": dict(self.provenance),
            "provenance_hash": self.provenance_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, manifest_path: str | None = None) -> "ModelExportManifest":
        model = payload.get("model", {})
        weights = payload.get("weights", {})
        if not isinstance(model, Mapping) or not isinstance(weights, Mapping):
            raise ModelExportError("export manifest has invalid model or weights fields")
        return cls(
            model_name=str(model.get("name", "")),
            backend=str(model.get("backend", "json")),
            inputs=tuple(TensorSpec.from_dict(item) for item in payload.get("inputs", ())),
            outputs=tuple(TensorSpec.from_dict(item) for item in payload.get("outputs", ())),
            config=payload.get("config", {}),
            weights_path=weights.get("path"),
            weights_hash=weights.get("blake3"),
            provenance=payload.get("provenance", {}),
            provenance_hash=str(payload.get("provenance_hash", "")),
            rvf_compatible=bool(payload.get("rvf_compatible", True)),
            format=str(payload.get("format", EXPORT_FORMAT)),
            schema_version=int(payload.get("schema_version", EXPORT_SCHEMA_VERSION)),
            manifest_path=manifest_path,
        )


def build_model_export_manifest(
    *,
    model_name: str,
    inputs: Sequence[TensorSpec | Mapping[str, Any]],
    outputs: Sequence[TensorSpec | Mapping[str, Any]],
    backend: str = "json",
    config: Mapping[str, Any] | None = None,
    weights_path: str | Path | None = None,
    provenance: Mapping[str, Any] | None = None,
    rvf_compatible: bool = True,
) -> ModelExportManifest:
    """Build a deterministic export manifest with BLAKE3 provenance."""

    weights_hash = hash_file(weights_path) if weights_path is not None and Path(weights_path).exists() else None
    manifest = ModelExportManifest(
        model_name=model_name,
        backend=backend,
        inputs=tuple(_coerce_tensor_spec(item) for item in inputs),
        outputs=tuple(_coerce_tensor_spec(item) for item in outputs),
        config=config or {},
        weights_path=None if weights_path is None else str(weights_path),
        weights_hash=weights_hash,
        provenance=provenance or {},
        rvf_compatible=rvf_compatible,
    )
    return replace(manifest, provenance_hash=_provenance_hash(manifest))


def save_model_export_manifest(
    path: str | Path,
    manifest: ModelExportManifest | None = None,
    **kwargs: Any,
) -> ModelExportManifest:
    """Write a model export manifest as deterministic JSON."""

    output_path = Path(path)
    if manifest is None:
        manifest = build_model_export_manifest(**kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return replace(manifest, manifest_path=str(output_path))


def load_model_export_manifest(path: str | Path) -> ModelExportManifest:
    """Load a model export manifest from JSON."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ModelExportError("model export manifest must be a JSON object")
    return ModelExportManifest.from_dict(payload, manifest_path=str(manifest_path))


def hash_file(path: str | Path) -> str:
    """Return the BLAKE3 hash of a file."""

    hasher = blake3()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def shape_metadata(manifest: ModelExportManifest) -> dict[str, list[dict[str, Any]]]:
    """Return RVF-like shape metadata grouped by input and output tensors."""

    return {
        "inputs": [item.to_dict() for item in manifest.inputs],
        "outputs": [item.to_dict() for item in manifest.outputs],
    }


def _coerce_tensor_spec(value: TensorSpec | Mapping[str, Any]) -> TensorSpec:
    if isinstance(value, TensorSpec):
        return value
    return TensorSpec.from_dict(value)


def _provenance_hash(manifest: ModelExportManifest) -> str:
    payload = manifest.to_dict()
    payload.pop("provenance_hash", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return blake3(canonical).hexdigest()


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
