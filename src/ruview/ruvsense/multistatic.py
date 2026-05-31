"""Attention-style multistatic fusion for node and link observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
NumericArray = NDArray[np.float64] | NDArray[np.complex128]


@dataclass(frozen=True)
class MultistaticFusionConfig:
    """Configuration for quality/coherence/distance attention weights."""

    quality_power: float = 1.0
    coherence_power: float = 1.0
    distance_power: float = 1.0
    distance_reference_m: float = 1.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.quality_power < 0.0:
            raise ValueError("quality_power must be non-negative")
        if self.coherence_power < 0.0:
            raise ValueError("coherence_power must be non-negative")
        if self.distance_power < 0.0:
            raise ValueError("distance_power must be non-negative")
        if self.distance_reference_m <= 0.0:
            raise ValueError("distance_reference_m must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")


@dataclass(frozen=True)
class MultistaticObservation:
    """Feature or CSI-summary observation from one node/link."""

    features: ArrayLike
    node_id: str = ""
    quality: float = 1.0
    coherence: float = 1.0
    distance_m: float | None = None


@dataclass(frozen=True)
class MultistaticContribution:
    """One node/link contribution to the fused field."""

    node_id: str
    weight: float
    quality: float
    coherence: float
    distance_m: float | None
    contribution_field: NumericArray

    @property
    def contribution(self) -> NumericArray:
        """Alias for the weighted contribution field."""

        return self.contribution_field


@dataclass(frozen=True)
class MultistaticFusionResult:
    """Fused field/features and the attention weights used to produce them."""

    fused_field: NumericArray
    weights: FloatArray
    contributions: tuple[MultistaticContribution, ...]
    fused_quality: float
    weight_entropy: float

    @property
    def fused(self) -> NumericArray:
        """Alias for concise result access."""

        return self.fused_field

    @property
    def fused_features(self) -> NumericArray:
        """Alias for feature-oriented callers."""

        return self.fused_field


LinkObservation = MultistaticObservation
MultistaticConfig = MultistaticFusionConfig
MultistaticResult = MultistaticFusionResult
NodeObservation = MultistaticObservation
NodeContribution = MultistaticContribution


def compute_attention_weights(
    observations: Sequence[MultistaticObservation | ArrayLike],
    *,
    qualities: Sequence[float] | None = None,
    coherences: Sequence[float] | None = None,
    distances_m: Sequence[float | None] | None = None,
    node_ids: Sequence[str] | None = None,
    config: MultistaticFusionConfig | None = None,
) -> FloatArray:
    """Compute attention weights from quality, coherence, and distance priors."""

    cfg = config or MultistaticFusionConfig()
    obs = _coerce_observations(observations, qualities, coherences, distances_m, node_ids)
    if not obs:
        raise ValueError("multistatic fusion requires at least one observation")

    scores = np.asarray([_attention_score(item, cfg) for item in obs], dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("attention scores must be finite")
    if np.all(scores <= 0.0):
        return np.full(scores.shape, 1.0 / scores.size, dtype=np.float64)

    sharpened = np.where(scores > 0.0, scores ** (1.0 / cfg.temperature), 0.0)
    total = float(np.sum(sharpened))
    if total <= 0.0:
        return np.full(scores.shape, 1.0 / scores.size, dtype=np.float64)
    return (sharpened / total).astype(np.float64, copy=False)


def attention_weights(
    observations: Sequence[MultistaticObservation | ArrayLike],
    *,
    qualities: Sequence[float] | None = None,
    coherences: Sequence[float] | None = None,
    distances_m: Sequence[float | None] | None = None,
    node_ids: Sequence[str] | None = None,
    config: MultistaticFusionConfig | None = None,
) -> FloatArray:
    """Compatibility alias for :func:`compute_attention_weights`."""

    return compute_attention_weights(
        observations,
        qualities=qualities,
        coherences=coherences,
        distances_m=distances_m,
        node_ids=node_ids,
        config=config,
    )


def fuse_multistatic(
    observations: Sequence[MultistaticObservation | ArrayLike],
    *,
    qualities: Sequence[float] | None = None,
    coherences: Sequence[float] | None = None,
    distances_m: Sequence[float | None] | None = None,
    node_ids: Sequence[str] | None = None,
    config: MultistaticFusionConfig | None = None,
) -> MultistaticFusionResult:
    """Fuse node/link feature fields using quality-aware attention weights."""

    cfg = config or MultistaticFusionConfig()
    obs = _coerce_observations(observations, qualities, coherences, distances_m, node_ids)
    if not obs:
        raise ValueError("multistatic fusion requires at least one observation")

    arrays = [_as_numeric_array(item.features) for item in obs]
    first_shape = arrays[0].shape
    for index, array in enumerate(arrays):
        if array.shape != first_shape:
            raise ValueError(
                "all multistatic feature fields must have matching shapes; "
                f"observation 0 has {first_shape}, observation {index} has {array.shape}"
            )

    weights = compute_attention_weights(obs, config=cfg)
    fused = np.zeros(first_shape, dtype=np.result_type(*arrays, np.float64))
    contributions: list[MultistaticContribution] = []
    for index, (item, array, weight) in enumerate(zip(obs, arrays, weights, strict=True)):
        contribution = weight * array
        fused += contribution
        contributions.append(
            MultistaticContribution(
                node_id=item.node_id or f"node-{index}",
                weight=float(weight),
                quality=float(item.quality),
                coherence=float(item.coherence),
                distance_m=item.distance_m,
                contribution_field=contribution,
            )
        )

    quality_terms = np.asarray([item.quality * item.coherence for item in obs], dtype=np.float64)
    fused_quality = float(np.sum(weights * quality_terms))
    return MultistaticFusionResult(
        fused_field=fused,
        weights=weights,
        contributions=tuple(contributions),
        fused_quality=fused_quality,
        weight_entropy=normalized_weight_entropy(weights),
    )


def normalized_weight_entropy(weights: ArrayLike) -> float:
    """Return 1 for uniform weights and 0 when a single link dominates."""

    weight_array = np.asarray(weights, dtype=np.float64).ravel()
    if weight_array.size <= 1:
        return 1.0
    valid = weight_array[weight_array > 0.0]
    if valid.size == 0:
        return 0.0
    entropy = float(np.sum(-valid * np.log(valid)))
    max_entropy = float(np.log(weight_array.size))
    if max_entropy <= 0.0:
        return 1.0
    return min(1.0, max(0.0, entropy / max_entropy))


def _coerce_observations(
    observations: Sequence[MultistaticObservation | ArrayLike],
    qualities: Sequence[float] | None,
    coherences: Sequence[float] | None,
    distances_m: Sequence[float | None] | None,
    node_ids: Sequence[str] | None,
) -> list[MultistaticObservation]:
    obs_list = list(observations)
    count = len(obs_list)
    _validate_optional_length(qualities, count, "qualities")
    _validate_optional_length(coherences, count, "coherences")
    _validate_optional_length(distances_m, count, "distances_m")
    _validate_optional_length(node_ids, count, "node_ids")

    coerced: list[MultistaticObservation] = []
    for index, item in enumerate(obs_list):
        if isinstance(item, MultistaticObservation):
            features = item.features
            quality = item.quality if qualities is None else float(qualities[index])
            coherence = item.coherence if coherences is None else float(coherences[index])
            distance = item.distance_m if distances_m is None else distances_m[index]
            node_id = item.node_id if node_ids is None else str(node_ids[index])
        else:
            features = item
            quality = 1.0 if qualities is None else float(qualities[index])
            coherence = 1.0 if coherences is None else float(coherences[index])
            distance = None if distances_m is None else distances_m[index]
            node_id = f"node-{index}" if node_ids is None else str(node_ids[index])

        _validate_metric(quality, "quality")
        _validate_metric(coherence, "coherence")
        if distance is not None:
            distance = float(distance)
            if not np.isfinite(distance) or distance < 0.0:
                raise ValueError("distance_m values must be finite and non-negative")

        coerced.append(
            MultistaticObservation(
                features=features,
                node_id=node_id,
                quality=float(quality),
                coherence=float(coherence),
                distance_m=distance,
            )
        )
    return coerced


def _validate_optional_length(values: Sequence[object] | None, count: int, name: str) -> None:
    if values is not None and len(values) != count:
        raise ValueError(f"{name} length must match the number of observations")


def _validate_metric(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} values must be finite and non-negative")


def _attention_score(observation: MultistaticObservation, config: MultistaticFusionConfig) -> float:
    quality = float(observation.quality) ** config.quality_power
    coherence = float(observation.coherence) ** config.coherence_power
    distance_factor = 1.0
    if observation.distance_m is not None and config.distance_power > 0.0:
        distance_ratio = float(observation.distance_m) / config.distance_reference_m
        distance_factor = 1.0 / ((1.0 + distance_ratio) ** config.distance_power)
    return float(quality * coherence * distance_factor)


def _as_numeric_array(values: ArrayLike) -> NumericArray:
    array = np.asarray(values)
    if array.size == 0:
        raise ValueError("multistatic feature fields must not be empty")
    if np.iscomplexobj(array):
        converted = array.astype(np.complex128, copy=False)
        finite = np.isfinite(converted.real) & np.isfinite(converted.imag)
    else:
        converted = array.astype(np.float64, copy=False)
        finite = np.isfinite(converted)
    if not bool(np.all(finite)):
        raise ValueError("multistatic feature fields must contain only finite values")
    return converted


__all__ = [
    "LinkObservation",
    "MultistaticConfig",
    "MultistaticContribution",
    "MultistaticFusionConfig",
    "MultistaticFusionResult",
    "MultistaticObservation",
    "MultistaticResult",
    "NodeContribution",
    "NodeObservation",
    "attention_weights",
    "compute_attention_weights",
    "fuse_multistatic",
    "normalized_weight_entropy",
]
