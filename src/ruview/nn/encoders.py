"""RF encoders and contrastive embedding helpers.

The PyTorch classes in this module are intentionally optional: importing the
module only requires NumPy, while constructing the neural modules requires the
``nn`` extra that provides torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:  # pragma: no cover - exercised by torch-enabled tests when available.
    import torch
    from torch import nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - default lightweight installs.
    torch = None
    nn = None
    F = None


EMBEDDING_DIM = 256
FloatArray = NDArray[np.float64]
Reduction = Literal["mean", "sum", "none"]


class TorchUnavailableError(RuntimeError):
    """Raised when an optional PyTorch model is constructed without torch."""


@dataclass(frozen=True)
class RFEncoderConfig:
    """Configuration for :class:`RFEncoder`.

    Inputs are CSI amplitude and phase tensors with shape
    ``[batch, antenna_paths, subcarriers]``.  The encoder stacks them into
    channels, applies compact 2D convolutions, optionally mixes spatial tokens
    with a transformer encoder, and returns an L2-normalized RF embedding.
    """

    input_channels: int = 2
    embedding_dim: int = EMBEDDING_DIM
    conv_channels: tuple[int, ...] = (32, 64)
    hidden_dim: int = 256
    dropout: float = 0.1
    use_transformer: bool = False
    transformer_layers: int = 1
    transformer_heads: int = 4
    sanitize_phase: bool = True
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not self.conv_channels or any(ch <= 0 for ch in self.conv_channels):
            raise ValueError("conv_channels must contain positive channel counts")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.transformer_layers <= 0:
            raise ValueError("transformer_layers must be positive")
        if self.transformer_heads <= 0:
            raise ValueError("transformer_heads must be positive")
        if self.use_transformer and self.conv_channels[-1] % self.transformer_heads != 0:
            raise ValueError("last conv channel count must be divisible by transformer_heads")


def phase_sanitize(phase: ArrayLike | Any) -> FloatArray | Any:
    """Return first-order phase differences with a zero-padded first column.

    This mirrors the Rust training model's differentiable subcarrier-difference
    sanitizer and works for both NumPy arrays and torch tensors.
    """

    if _is_torch_tensor(phase):
        if phase.ndim < 1:
            raise ValueError("phase must have at least one dimension")
        if phase.shape[-1] <= 1:
            return torch.zeros_like(phase)
        zeros = torch.zeros_like(phase[..., :1])
        return torch.cat((zeros, phase[..., 1:] - phase[..., :-1]), dim=-1)

    values = np.asarray(phase, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("phase must have at least one dimension")
    if values.shape[-1] <= 1:
        return np.zeros_like(values)
    zeros = np.zeros(values.shape[:-1] + (1,), dtype=values.dtype)
    return np.concatenate((zeros, np.diff(values, axis=-1)), axis=-1)


def l2_normalize(values: ArrayLike | Any, *, axis: int = -1, eps: float = 1e-12) -> FloatArray | Any:
    """Normalize embeddings along ``axis`` for NumPy arrays or torch tensors."""

    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if _is_torch_tensor(values):
        return F.normalize(values, p=2, dim=axis, eps=eps)

    arr = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    return arr / np.maximum(norm, eps)


def cosine_similarity_matrix(embeddings: ArrayLike | Any, *, temperature: float = 1.0) -> FloatArray | Any:
    """Return pairwise cosine logits for a batch of embeddings."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    normalized = l2_normalize(embeddings, axis=-1)
    if _is_torch_tensor(normalized):
        _validate_rank(normalized, 2, "embeddings")
        return normalized @ normalized.transpose(0, 1) / temperature

    arr = np.asarray(normalized, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("embeddings must be rank-2 [batch, dim]")
    return (arr @ arr.T) / temperature


def calibration_robustness_loss(
    under_cal_a: ArrayLike | Any,
    under_cal_b: ArrayLike | Any,
    *,
    reduction: Reduction = "mean",
) -> float | FloatArray | Any:
    """Mean squared embedding drift between two calibration baselines."""

    if _is_torch_tensor(under_cal_a) or _is_torch_tensor(under_cal_b):
        if not (_is_torch_tensor(under_cal_a) and _is_torch_tensor(under_cal_b)):
            raise TypeError("both calibration inputs must be torch tensors or neither must be")
        _validate_same_shape(under_cal_a, under_cal_b, "under_cal_a", "under_cal_b")
        per_sample = (under_cal_a - under_cal_b).pow(2).mean(dim=-1)
        return _reduce_torch(per_sample, reduction)

    a = np.asarray(under_cal_a, dtype=np.float64)
    b = np.asarray(under_cal_b, dtype=np.float64)
    _validate_numpy_same_shape(a, b, "under_cal_a", "under_cal_b")
    per_sample = np.mean((a - b) ** 2, axis=-1)
    return _reduce_numpy(per_sample, reduction)


def triplet_loss(
    anchor: ArrayLike | Any,
    positive: ArrayLike | Any,
    negative: ArrayLike | Any,
    *,
    margin: float = 0.2,
    reduction: Reduction = "mean",
) -> float | FloatArray | Any:
    """Triplet contrastive loss over RF embeddings.

    Computes ``max(0, d(anchor, positive) - d(anchor, negative) + margin)``
    with squared Euclidean distances, matching the Rust research helper.
    """

    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    if _is_torch_tensor(anchor) or _is_torch_tensor(positive) or _is_torch_tensor(negative):
        if not all(_is_torch_tensor(x) for x in (anchor, positive, negative)):
            raise TypeError("anchor, positive, and negative must all be torch tensors or all NumPy-like")
        _validate_same_shape(anchor, positive, "anchor", "positive")
        _validate_same_shape(anchor, negative, "anchor", "negative")
        ap = (anchor - positive).pow(2).sum(dim=-1)
        an = (anchor - negative).pow(2).sum(dim=-1)
        return _reduce_torch(F.relu(ap - an + margin), reduction)

    a = np.asarray(anchor, dtype=np.float64)
    p = np.asarray(positive, dtype=np.float64)
    n = np.asarray(negative, dtype=np.float64)
    _validate_numpy_same_shape(a, p, "anchor", "positive")
    _validate_numpy_same_shape(a, n, "anchor", "negative")
    ap = np.sum((a - p) ** 2, axis=-1)
    an = np.sum((a - n) ** 2, axis=-1)
    return _reduce_numpy(np.maximum(0.0, ap - an + margin), reduction)


@dataclass(frozen=True)
class Triplet:
    """Indices for one anchor/positive/negative contrastive sample."""

    anchor: int
    positive: int
    negative: int


@dataclass(frozen=True)
class ContrastiveBatcher:
    """Deterministic triplet sampler for cross-environment positives."""

    state_labels: tuple[Hashable, ...]
    environment_labels: tuple[Hashable, ...]

    def __init__(self, state_labels: Any, environment_labels: Any) -> None:
        states = tuple(state_labels)
        envs = tuple(environment_labels)
        if len(states) != len(envs):
            raise ValueError("state_labels and environment_labels must have the same length")
        object.__setattr__(self, "state_labels", states)
        object.__setattr__(self, "environment_labels", envs)

    def triplets(self) -> tuple[Triplet, ...]:
        """Enumerate lowest-index valid triplets for reproducible tests."""

        out: list[Triplet] = []
        n = len(self.state_labels)
        for anchor in range(n):
            positive = next(
                (
                    idx
                    for idx in range(n)
                    if idx != anchor
                    and self.state_labels[idx] == self.state_labels[anchor]
                    and self.environment_labels[idx] != self.environment_labels[anchor]
                ),
                None,
            )
            negative = next(
                (idx for idx in range(n) if self.state_labels[idx] != self.state_labels[anchor]),
                None,
            )
            if positive is not None and negative is not None:
                out.append(Triplet(anchor=anchor, positive=positive, negative=negative))
        return tuple(out)


if torch is not None:

    class RFEncoder(nn.Module):
        """Compact PyTorch RF encoder for CSI amplitude/phase windows."""

        def __init__(self, config: RFEncoderConfig | None = None) -> None:
            super().__init__()
            self.config = config or RFEncoderConfig()

            layers: list[nn.Module] = []
            in_channels = self.config.input_channels
            for out_channels in self.config.conv_channels:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
                layers.append(nn.GroupNorm(_group_count(out_channels), out_channels))
                layers.append(nn.GELU())
                if self.config.dropout > 0.0:
                    layers.append(nn.Dropout2d(self.config.dropout))
                in_channels = out_channels
            self.features = nn.Sequential(*layers)

            feature_dim = self.config.conv_channels[-1]
            if self.config.use_transformer:
                layer = nn.TransformerEncoderLayer(
                    d_model=feature_dim,
                    nhead=self.config.transformer_heads,
                    dim_feedforward=self.config.hidden_dim,
                    dropout=self.config.dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                self.token_mixer: nn.Module | None = nn.TransformerEncoder(
                    layer,
                    num_layers=self.config.transformer_layers,
                )
            else:
                self.token_mixer = None

            self.projection = nn.Sequential(
                nn.Linear(feature_dim, self.config.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim, self.config.embedding_dim),
            )

        def forward(self, amplitude: Any, phase: Any | None = None) -> Any:
            """Encode amplitude/phase CSI into ``[batch, embedding_dim]``."""

            x = self._coerce_input(amplitude, phase)
            features = self.features(x)
            if self.token_mixer is None:
                pooled = features.mean(dim=(-2, -1))
            else:
                tokens = features.flatten(2).transpose(1, 2)
                tokens = self.token_mixer(tokens)
                pooled = tokens.mean(dim=1)
            embedding = self.projection(pooled)
            if self.config.normalize:
                embedding = F.normalize(embedding, p=2, dim=-1)
            return embedding

        encode = forward

        def _coerce_input(self, amplitude: Any, phase: Any | None) -> Any:
            if not _is_torch_tensor(amplitude):
                raise TypeError("amplitude must be a torch tensor")
            if phase is None:
                if amplitude.ndim == 4:
                    if amplitude.shape[1] != self.config.input_channels:
                        raise ValueError(
                            f"expected {self.config.input_channels} channels, got {amplitude.shape[1]}"
                        )
                    return amplitude
                if amplitude.ndim == 3 and self.config.input_channels == 1:
                    return amplitude.unsqueeze(1)
                raise ValueError(
                    "pass amplitude and phase as rank-3 tensors, or a rank-4 channel tensor"
                )

            if not _is_torch_tensor(phase):
                raise TypeError("phase must be a torch tensor")
            _validate_rank(amplitude, 3, "amplitude")
            _validate_rank(phase, 3, "phase")
            _validate_same_shape(amplitude, phase, "amplitude", "phase")
            if self.config.input_channels != 2:
                raise ValueError("amplitude+phase input requires input_channels=2")
            phase_input = phase_sanitize(phase) if self.config.sanitize_phase else phase
            return torch.stack((amplitude, phase_input), dim=1)


else:

    class RFEncoder:  # pragma: no cover - simple optional dependency guard.
        """Placeholder that explains how to enable the PyTorch RF encoder."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise TorchUnavailableError("RFEncoder requires torch; install with `uv sync --extra nn`")


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _validate_rank(value: Any, rank: int, name: str) -> None:
    if value.ndim != rank:
        raise ValueError(f"{name} must be rank-{rank}, got shape {tuple(value.shape)}")


def _validate_same_shape(a: Any, b: Any, a_name: str, b_name: str) -> None:
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(f"{a_name} and {b_name} must have the same shape")


def _validate_numpy_same_shape(a: np.ndarray, b: np.ndarray, a_name: str, b_name: str) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{a_name} and {b_name} must have the same shape")


def _reduce_numpy(values: np.ndarray, reduction: Reduction) -> float | FloatArray:
    if reduction == "none":
        return values.astype(np.float64, copy=False)
    if reduction == "mean":
        return float(np.mean(values))
    if reduction == "sum":
        return float(np.sum(values))
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")


def _reduce_torch(values: Any, reduction: Reduction) -> Any:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
