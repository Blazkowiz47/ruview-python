"""Browser visualization helpers for local pose-fusion demos.

The functions here are pure data builders.  They do not open WebSockets, touch
browser APIs, render Canvas content, or execute WASM; they only produce URLs,
arrays, and JSON-compatible payloads that browser code can consume.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urlsplit

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_CSI_WS_HOST = "localhost"
DEFAULT_CSI_WS_PORT = 3_030
DEFAULT_CSI_WS_PATH = "/ws/csi"

COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

COCO_SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)


def build_websocket_url(
    host: str = DEFAULT_CSI_WS_HOST,
    port: int | None = DEFAULT_CSI_WS_PORT,
    path: str = DEFAULT_CSI_WS_PATH,
    *,
    secure: bool = False,
    query: Mapping[str, object] | Sequence[tuple[str, object]] | None = None,
) -> str:
    """Build a browser WebSocket URL for the CSI stream."""

    parsed = urlsplit(host)
    if parsed.scheme in {"ws", "wss", "http", "https"}:
        secure = parsed.scheme in {"wss", "https"}
        host_part = parsed.hostname or parsed.netloc
        port = parsed.port if port is None else port
        if path == DEFAULT_CSI_WS_PATH and parsed.path:
            path = parsed.path
    else:
        host_part = host

    scheme = "wss" if secure else "ws"
    normalized_path = path if path.startswith("/") else f"/{path}"
    normalized_host = _format_host(host_part)
    port_part = "" if port is None else f":{int(port)}"
    query_part = "" if not query else f"?{urlencode(query, doseq=True)}"
    return f"{scheme}://{normalized_host}{port_part}{normalized_path}{query_part}"


def _format_host(host: str) -> str:
    stripped = host.strip().strip("/")
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped
    if ":" in stripped and stripped.count(":") > 1:
        return f"[{stripped}]"
    return stripped


@dataclass(frozen=True)
class ModalityQuality:
    """Availability and confidence estimates for visual/CSI fusion."""

    visual: float = 1.0
    csi: float = 1.0
    visual_available: bool = True
    csi_available: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "visual", _clamp01(self.visual))
        object.__setattr__(self, "csi", _clamp01(self.csi))


def _clamp01(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("quality values must be finite")
    return min(1.0, max(0.0, value))


def _as_1d_array(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} values must be finite")
    return array


def _as_weight_array(weights: ArrayLike | float, shape: tuple[int, ...]) -> NDArray[np.float64]:
    array = np.asarray(weights, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(shape, float(array), dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"attention_weights shape {array.shape} does not match embedding shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("attention weights must be finite")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("attention weights must be in [0, 1]")
    return array.astype(np.float64, copy=False)


def quality_gated_attention_weights(
    attention_weights: ArrayLike | float,
    *,
    shape: int | tuple[int, ...],
    quality: ModalityQuality | None = None,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Adjust visual attention weights using video and CSI quality estimates."""

    target_shape = (shape,) if isinstance(shape, int) else tuple(shape)
    base = _as_weight_array(attention_weights, target_shape)
    quality = quality or ModalityQuality()

    if not quality.visual_available and not quality.csi_available:
        raise ValueError("at least one modality must be available")
    if not quality.visual_available:
        return np.zeros(target_shape, dtype=np.float64)
    if not quality.csi_available:
        return np.ones(target_shape, dtype=np.float64)

    visual_score = base * quality.visual
    csi_score = (1.0 - base) * quality.csi
    denom = visual_score + csi_score
    return np.divide(
        visual_score,
        denom + eps,
        out=np.full(target_shape, 0.5, dtype=np.float64),
        where=denom > eps,
    )


def fuse_embeddings(
    visual_embedding: ArrayLike,
    csi_embedding: ArrayLike,
    attention_weights: ArrayLike | float,
    *,
    quality: ModalityQuality | None = None,
) -> NDArray[np.float64]:
    """Fuse visual and CSI embeddings with visual attention weights."""

    visual = _as_1d_array(visual_embedding, name="visual_embedding")
    csi = _as_1d_array(csi_embedding, name="csi_embedding")
    if visual.shape != csi.shape:
        raise ValueError(f"embedding shapes differ: {visual.shape} vs {csi.shape}")
    weights = quality_gated_attention_weights(attention_weights, shape=visual.shape, quality=quality)
    return weights * visual + (1.0 - weights) * csi


@dataclass(frozen=True)
class PoseKeypoint:
    """One COCO keypoint for browser canvas payloads."""

    index: int
    x: float
    y: float
    confidence: float
    name: str | None = None

    def to_canvas_dict(self, *, min_confidence: float) -> dict[str, object]:
        name = self.name or COCO_KEYPOINT_NAMES[self.index]
        confidence = float(self.confidence)
        return {
            "index": int(self.index),
            "name": name,
            "x": float(self.x),
            "y": float(self.y),
            "confidence": confidence,
            "visible": confidence >= min_confidence,
        }


def _coerce_keypoint(
    raw: PoseKeypoint | Mapping[str, Any] | Sequence[float],
    index: int,
    *,
    width: float,
    height: float,
    normalized: bool,
) -> PoseKeypoint:
    if isinstance(raw, PoseKeypoint):
        keypoint = raw
    elif isinstance(raw, Mapping):
        keypoint = PoseKeypoint(
            index=int(raw.get("index", index)),
            x=float(raw["x"]),
            y=float(raw["y"]),
            confidence=float(raw.get("confidence", raw.get("score", 1.0))),
            name=raw.get("name"),
        )
    else:
        if len(raw) < 3:
            raise ValueError("keypoint sequences must contain x, y, confidence")
        keypoint = PoseKeypoint(index=index, x=float(raw[0]), y=float(raw[1]), confidence=float(raw[2]))

    x = keypoint.x * width if normalized else keypoint.x
    y = keypoint.y * height if normalized else keypoint.y
    return PoseKeypoint(
        index=keypoint.index,
        x=x,
        y=y,
        confidence=keypoint.confidence,
        name=keypoint.name,
    )


def build_canvas_pose_payload(
    keypoints: Sequence[PoseKeypoint | Mapping[str, Any] | Sequence[float]],
    *,
    width: int,
    height: int,
    min_confidence: float = 0.2,
    normalized: bool = True,
    edges: Sequence[tuple[int, int]] = COCO_SKELETON_EDGES,
) -> dict[str, object]:
    """Build a JSON-compatible Canvas skeleton payload."""

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if len(keypoints) != len(COCO_KEYPOINT_NAMES):
        raise ValueError(f"expected {len(COCO_KEYPOINT_NAMES)} COCO keypoints")

    canvas_keypoints = [
        _coerce_keypoint(raw, index, width=float(width), height=float(height), normalized=normalized)
        for index, raw in enumerate(keypoints)
    ]
    keypoint_payload = [
        keypoint.to_canvas_dict(min_confidence=min_confidence) for keypoint in canvas_keypoints
    ]

    edge_payload = []
    for start, end in edges:
        start_point = keypoint_payload[start]
        end_point = keypoint_payload[end]
        edge_payload.append(
            {
                "start": int(start),
                "end": int(end),
                "start_name": start_point["name"],
                "end_name": end_point["name"],
                "visible": bool(start_point["visible"] and end_point["visible"]),
            }
        )

    return {
        "schema": "ruview.browser_pose.canvas.v1",
        "width": int(width),
        "height": int(height),
        "keypoints": keypoint_payload,
        "edges": edge_payload,
    }


__all__ = [
    "COCO_KEYPOINT_NAMES",
    "COCO_SKELETON_EDGES",
    "DEFAULT_CSI_WS_HOST",
    "DEFAULT_CSI_WS_PATH",
    "DEFAULT_CSI_WS_PORT",
    "ModalityQuality",
    "PoseKeypoint",
    "build_canvas_pose_payload",
    "build_websocket_url",
    "fuse_embeddings",
    "quality_gated_attention_weights",
]
