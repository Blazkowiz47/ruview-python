"""Neural prediction heads for RF embeddings and DensePose-style maps."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

try:  # pragma: no cover - exercised by torch-enabled tests when available.
    import torch
    from torch import nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - default lightweight installs.
    torch = None
    nn = None
    F = None


class TorchUnavailableError(RuntimeError):
    """Raised when an optional PyTorch head is constructed without torch."""


class TaskKind(str, Enum):
    """Multi-task heads used by the RF embedding experiments."""

    POSE = "pose"
    PRESENCE = "presence"
    COUNT = "count"
    ACTIVITY = "activity"
    VITALS = "vitals"
    GAIT = "gait"
    IDENTITY_EMBEDDING = "identity_embedding"

    @classmethod
    def all(cls) -> tuple["TaskKind", ...]:
        return tuple(cls)


DEFAULT_TASK_DIMS: dict[TaskKind, int] = {
    TaskKind.POSE: 51,
    TaskKind.PRESENCE: 1,
    TaskKind.COUNT: 1,
    TaskKind.ACTIVITY: 6,
    TaskKind.VITALS: 2,
    TaskKind.GAIT: 32,
    TaskKind.IDENTITY_EMBEDDING: 128,
}


@dataclass(frozen=True)
class DensePoseOutput:
    """DensePose-style output maps produced from a feature map."""

    keypoint_heatmaps: Any
    part_logits: Any
    uv_coordinates: Any
    confidence: Any | None = None

    @property
    def uv(self) -> Any:
        """Alias for callers that use the shorter Rust training name."""

        return self.uv_coordinates


@dataclass(frozen=True)
class TaskHeadOutput:
    """Values and predictive uncertainty for one embedding task head."""

    task: TaskKind
    values: Any
    uncertainty: Any

    @property
    def confidence(self) -> Any:
        return 1.0 / (1.0 + self.uncertainty)


if torch is not None:

    class ProjectionHead(nn.Module):
        """Small MLP projection head for contrastive RF embeddings."""

        def __init__(
            self,
            input_dim: int = 256,
            projection_dim: int = 128,
            *,
            hidden_dim: int | None = None,
            dropout: float = 0.0,
            normalize: bool = True,
        ) -> None:
            super().__init__()
            if input_dim <= 0 or projection_dim <= 0:
                raise ValueError("input_dim and projection_dim must be positive")
            if hidden_dim is not None and hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout must be in [0, 1)")

            self.input_dim = input_dim
            self.projection_dim = projection_dim
            self.normalize = normalize
            mid = hidden_dim or input_dim
            self.net = nn.Sequential(
                nn.Linear(input_dim, mid),
                nn.LayerNorm(mid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mid, projection_dim),
            )

        def forward(self, embedding: Any) -> Any:
            _validate_embedding(embedding, self.input_dim, "embedding")
            projection = self.net(embedding)
            if self.normalize:
                projection = F.normalize(projection, p=2, dim=-1)
            return projection


    class DensePoseHead(nn.Module):
        """Predict keypoint heatmaps, body-part logits, UV maps, and confidence."""

        def __init__(
            self,
            input_channels: int = 256,
            *,
            hidden_channels: tuple[int, ...] = (256, 256),
            num_keypoints: int = 17,
            num_body_parts: int = 24,
            output_size: int | tuple[int, int] | None = None,
            include_confidence: bool = True,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            if input_channels <= 0:
                raise ValueError("input_channels must be positive")
            if not hidden_channels or any(ch <= 0 for ch in hidden_channels):
                raise ValueError("hidden_channels must contain positive channel counts")
            if num_keypoints <= 0:
                raise ValueError("num_keypoints must be positive")
            if num_body_parts <= 0:
                raise ValueError("num_body_parts must be positive")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout must be in [0, 1)")

            self.input_channels = input_channels
            self.num_keypoints = num_keypoints
            self.num_body_parts = num_body_parts
            self.output_size = output_size
            self.include_confidence = include_confidence

            blocks: list[nn.Module] = []
            in_channels = input_channels
            for out_channels in hidden_channels:
                blocks.append(_conv_block(in_channels, out_channels, dropout))
                in_channels = out_channels
            self.shared = nn.Sequential(*blocks)
            self.keypoint_head = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)
            self.part_head = nn.Conv2d(in_channels, num_body_parts + 1, kernel_size=1)
            self.uv_head = nn.Conv2d(in_channels, num_body_parts * 2, kernel_size=1)
            self.confidence_head = (
                nn.Conv2d(in_channels, 1, kernel_size=1) if include_confidence else None
            )

        def forward(self, features: Any, *, output_size: int | tuple[int, int] | None = None) -> DensePoseOutput:
            _validate_feature_map(features, self.input_channels, "features")
            shared = self.shared(features)
            target_size = _as_hw(output_size if output_size is not None else self.output_size)
            if target_size is not None:
                shared = F.interpolate(shared, size=target_size, mode="bilinear", align_corners=False)

            confidence = None
            if self.confidence_head is not None:
                confidence = torch.sigmoid(self.confidence_head(shared))

            return DensePoseOutput(
                keypoint_heatmaps=self.keypoint_head(shared),
                part_logits=self.part_head(shared),
                uv_coordinates=torch.sigmoid(self.uv_head(shared)),
                confidence=confidence,
            )


    class LinearTaskHead(nn.Module):
        """Linear value head with a learned softplus uncertainty scalar."""

        def __init__(self, task: TaskKind | str, input_dim: int, output_dim: int) -> None:
            super().__init__()
            task_kind = _task_kind(task)
            if input_dim <= 0 or output_dim <= 0:
                raise ValueError("input_dim and output_dim must be positive")
            self.task = task_kind
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.value = nn.Linear(input_dim, output_dim)
            self.log_variance = nn.Linear(input_dim, 1)

        def forward(self, embedding: Any) -> TaskHeadOutput:
            _validate_embedding(embedding, self.input_dim, "embedding")
            uncertainty = F.softplus(self.log_variance(embedding))
            return TaskHeadOutput(
                task=self.task,
                values=self.value(embedding),
                uncertainty=uncertainty,
            )


    class MultiTaskHeads(nn.Module):
        """Ablatable lightweight heads over a shared RF embedding."""

        def __init__(
            self,
            input_dim: int = 256,
            *,
            task_dims: Mapping[TaskKind | str, int] | None = None,
            enabled_tasks: tuple[TaskKind | str, ...] | None = None,
        ) -> None:
            super().__init__()
            if input_dim <= 0:
                raise ValueError("input_dim must be positive")

            dims = dict(task_dims or DEFAULT_TASK_DIMS)
            enabled = tuple(_task_kind(task) for task in enabled_tasks) if enabled_tasks is not None else None
            heads: dict[str, LinearTaskHead] = {}
            for task, output_dim in dims.items():
                task_kind = _task_kind(task)
                if enabled is not None and task_kind not in enabled:
                    continue
                heads[task_kind.value] = LinearTaskHead(task_kind, input_dim, int(output_dim))
            if not heads:
                raise ValueError("at least one task head must be configured")

            self.input_dim = input_dim
            self.heads = nn.ModuleDict(heads)

        def forward(
            self,
            embedding: Any,
            *,
            enabled_tasks: tuple[TaskKind | str, ...] | None = None,
        ) -> dict[TaskKind, TaskHeadOutput]:
            _validate_embedding(embedding, self.input_dim, "embedding")
            enabled = { _task_kind(task) for task in enabled_tasks } if enabled_tasks is not None else None
            outputs: dict[TaskKind, TaskHeadOutput] = {}
            for head in self.heads.values():
                if enabled is None or head.task in enabled:
                    outputs[head.task] = head(embedding)
            return outputs

        def forward_subset(
            self,
            embedding: Any,
            enabled_tasks: tuple[TaskKind | str, ...],
        ) -> dict[TaskKind, TaskHeadOutput]:
            return self.forward(embedding, enabled_tasks=enabled_tasks)


else:

    class ProjectionHead:  # pragma: no cover - simple optional dependency guard.
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise TorchUnavailableError("ProjectionHead requires torch; install with `uv sync --extra nn`")


    class DensePoseHead:  # pragma: no cover - simple optional dependency guard.
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise TorchUnavailableError("DensePoseHead requires torch; install with `uv sync --extra nn`")


    class LinearTaskHead:  # pragma: no cover - simple optional dependency guard.
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise TorchUnavailableError("LinearTaskHead requires torch; install with `uv sync --extra nn`")


    class MultiTaskHeads:  # pragma: no cover - simple optional dependency guard.
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise TorchUnavailableError("MultiTaskHeads requires torch; install with `uv sync --extra nn`")


def _task_kind(task: TaskKind | str) -> TaskKind:
    if isinstance(task, TaskKind):
        return task
    return TaskKind(str(task))


def _validate_embedding(value: Any, input_dim: int, name: str) -> None:
    if torch is None or not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must be rank-2 [batch, dim], got shape {tuple(value.shape)}")
    if value.shape[-1] != input_dim:
        raise ValueError(f"{name} last dimension must be {input_dim}, got {value.shape[-1]}")


def _validate_feature_map(value: Any, channels: int, name: str) -> None:
    if torch is None or not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if value.ndim != 4:
        raise ValueError(f"{name} must be rank-4 [batch, channels, height, width]")
    if value.shape[1] != channels:
        raise ValueError(f"{name} channel dimension must be {channels}, got {value.shape[1]}")


def _conv_block(in_channels: int, out_channels: int, dropout: float) -> Any:
    layers: list[Any] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.GELU(),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _as_hw(value: int | tuple[int, int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("output_size must be positive")
        return (value, value)
    if len(value) != 2:
        raise ValueError("output_size tuple must have length 2")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError("output_size dimensions must be positive")
    return (height, width)
