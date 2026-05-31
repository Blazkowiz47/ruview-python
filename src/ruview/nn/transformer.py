"""CSI-to-pose transformer research modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - exercised by torch-enabled tests when available.
    import torch
    from torch import nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - default lightweight installs.
    torch = None
    nn = None
    F = None

from .encoders import phase_sanitize
from .heads import DensePoseHead, DensePoseOutput


class TorchUnavailableError(RuntimeError):
    """Raised when an optional PyTorch transformer is constructed without torch."""


@dataclass(frozen=True)
class CsiToPoseConfig:
    """Configuration for the CSI-to-pose transformer."""

    num_subcarriers: int
    num_paths: int | None = None
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    embedding_dim: int = 256
    feature_channels: int = 64
    feature_map_size: int = 8
    heatmap_size: int = 16
    num_keypoints: int = 17
    num_body_parts: int = 24
    sanitize_phase: bool = True

    def __post_init__(self) -> None:
        if self.num_subcarriers <= 0:
            raise ValueError("num_subcarriers must be positive")
        if self.num_paths is not None and self.num_paths <= 0:
            raise ValueError("num_paths must be positive when provided")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.feature_channels <= 0:
            raise ValueError("feature_channels must be positive")
        if self.feature_map_size <= 0:
            raise ValueError("feature_map_size must be positive")
        if self.heatmap_size <= 0:
            raise ValueError("heatmap_size must be positive")
        if self.num_keypoints <= 0:
            raise ValueError("num_keypoints must be positive")
        if self.num_body_parts <= 0:
            raise ValueError("num_body_parts must be positive")


@dataclass(frozen=True)
class CsiToPoseOutput:
    """End-to-end transformer output."""

    embedding: Any
    features: Any
    keypoint_heatmaps: Any
    part_logits: Any
    uv_coordinates: Any
    confidence: Any | None = None

    @classmethod
    def from_densepose(cls, *, embedding: Any, features: Any, densepose: DensePoseOutput) -> "CsiToPoseOutput":
        return cls(
            embedding=embedding,
            features=features,
            keypoint_heatmaps=densepose.keypoint_heatmaps,
            part_logits=densepose.part_logits,
            uv_coordinates=densepose.uv_coordinates,
            confidence=densepose.confidence,
        )


if torch is not None:

    class CsiTransformerEncoder(nn.Module):
        """Transformer encoder over antenna-path CSI tokens."""

        def __init__(self, config: CsiToPoseConfig) -> None:
            super().__init__()
            self.config = config
            self.input_projection = nn.Linear(config.num_subcarriers * 2, config.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
            self.output_norm = nn.LayerNorm(config.d_model)

        def forward(self, amplitude: Any, phase: Any) -> Any:
            tokens = self.tokenize(amplitude, phase)
            positions = sinusoidal_positions(
                tokens.shape[1],
                self.config.d_model,
                device=tokens.device,
                dtype=tokens.dtype,
            )
            encoded = self.encoder(tokens + positions)
            return self.output_norm(encoded)

        def tokenize(self, amplitude: Any, phase: Any) -> Any:
            _validate_csi_pair(amplitude, phase, self.config)
            phase_input = phase_sanitize(phase) if self.config.sanitize_phase else phase
            pair = torch.cat((amplitude, phase_input), dim=-1)
            return self.input_projection(pair)


    class CsiToPoseTransformer(nn.Module):
        """End-to-end CSI amplitude/phase transformer with DensePose-style heads."""

        def __init__(self, config: CsiToPoseConfig) -> None:
            super().__init__()
            self.config = config
            self.encoder = CsiTransformerEncoder(config)
            self.embedding_head = nn.Linear(config.d_model, config.embedding_dim)
            self.feature_head = nn.Sequential(
                nn.Linear(
                    config.d_model,
                    config.feature_channels * config.feature_map_size * config.feature_map_size,
                ),
                nn.GELU(),
            )
            self.pose_head = DensePoseHead(
                input_channels=config.feature_channels,
                hidden_channels=(config.feature_channels, config.feature_channels),
                num_keypoints=config.num_keypoints,
                num_body_parts=config.num_body_parts,
                output_size=config.heatmap_size,
                dropout=config.dropout,
            )

        def forward(self, amplitude: Any, phase: Any) -> CsiToPoseOutput:
            encoded = self.encoder(amplitude, phase)
            summary = encoded.mean(dim=1)
            embedding = F.normalize(self.embedding_head(summary), p=2, dim=-1)
            features = self.feature_head(summary).view(
                amplitude.shape[0],
                self.config.feature_channels,
                self.config.feature_map_size,
                self.config.feature_map_size,
            )
            densepose = self.pose_head(features)
            return CsiToPoseOutput.from_densepose(
                embedding=embedding,
                features=features,
                densepose=densepose,
            )


else:

    class CsiTransformerEncoder:  # pragma: no cover - simple optional dependency guard.
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise TorchUnavailableError("CsiTransformerEncoder requires torch; install with `uv sync --extra nn`")


    class CsiToPoseTransformer:  # pragma: no cover - simple optional dependency guard.
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise TorchUnavailableError("CsiToPoseTransformer requires torch; install with `uv sync --extra nn`")


def sinusoidal_positions(length: int, d_model: int, *, device: Any = None, dtype: Any = None) -> Any:
    """Create rank-3 sinusoidal positional encodings ``[1, length, d_model]``."""

    if torch is None:
        raise TorchUnavailableError("sinusoidal_positions requires torch")
    if length <= 0 or d_model <= 0:
        raise ValueError("length and d_model must be positive")

    positions = torch.arange(length, device=device, dtype=dtype or torch.float32).unsqueeze(1)
    even_dims = torch.arange(0, d_model, 2, device=device, dtype=dtype or torch.float32)
    div = torch.exp(even_dims * (-math.log(10000.0) / d_model))
    pe = torch.zeros((length, d_model), device=device, dtype=dtype or torch.float32)
    pe[:, 0::2] = torch.sin(positions * div)
    pe[:, 1::2] = torch.cos(positions * div[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


def _validate_csi_pair(amplitude: Any, phase: Any, config: CsiToPoseConfig) -> None:
    if torch is None or not isinstance(amplitude, torch.Tensor) or not isinstance(phase, torch.Tensor):
        raise TypeError("amplitude and phase must be torch tensors")
    if amplitude.ndim != 3 or phase.ndim != 3:
        raise ValueError("amplitude and phase must be rank-3 [batch, paths, subcarriers]")
    if tuple(amplitude.shape) != tuple(phase.shape):
        raise ValueError("amplitude and phase must have the same shape")
    if amplitude.shape[-1] != config.num_subcarriers:
        raise ValueError(
            f"expected {config.num_subcarriers} subcarriers, got {amplitude.shape[-1]}"
        )
    if config.num_paths is not None and amplitude.shape[1] != config.num_paths:
        raise ValueError(f"expected {config.num_paths} antenna paths, got {amplitude.shape[1]}")
