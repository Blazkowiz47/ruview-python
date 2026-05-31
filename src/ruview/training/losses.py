"""Lightweight training losses for WiFi-DensePose research utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class TrainingLossError(ValueError):
    """Raised when loss inputs are malformed."""


@dataclass(frozen=True)
class LossWeights:
    """Scalar weights for the combined pose-training objective."""

    lambda_kp: float = 0.3
    lambda_dp: float = 0.6
    lambda_tr: float = 0.1

    def __post_init__(self) -> None:
        for name in ("lambda_kp", "lambda_dp", "lambda_tr"):
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise TrainingLossError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class LossComponents:
    """Scalar loss breakdown plus an optional backend scalar for autograd."""

    total: float
    keypoint: float
    densepose_parts: float | None = None
    densepose_uv: float | None = None
    transfer: float | None = None
    details: Mapping[str, float] = field(default_factory=dict)
    backend_loss: Any | None = field(default=None, repr=False, compare=False)


def generate_gaussian_heatmap(
    kp_x: float,
    kp_y: float,
    heatmap_size: int,
    sigma: float = 2.0,
    *,
    clip_radius_sigma: float = 3.0,
) -> FloatArray:
    """Generate one clipped Gaussian heatmap for a normalized keypoint."""

    heatmap_size = int(heatmap_size)
    sigma = float(sigma)
    if heatmap_size <= 0:
        raise TrainingLossError("heatmap_size must be positive")
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise TrainingLossError("sigma must be positive and finite")
    if not np.isfinite(kp_x) or not np.isfinite(kp_y):
        raise TrainingLossError("keypoint coordinates must be finite")
    if clip_radius_sigma < 0.0:
        raise TrainingLossError("clip_radius_sigma must be non-negative")

    scale = float(max(heatmap_size - 1, 1))
    center_x = float(kp_x) * scale
    center_y = float(kp_y) * scale
    rows, cols = np.indices((heatmap_size, heatmap_size), dtype=np.float64)
    dist_sq = (cols - center_x) ** 2 + (rows - center_y) ** 2
    heatmap = np.exp(-dist_sq / (2.0 * sigma * sigma))
    if clip_radius_sigma > 0.0:
        heatmap = np.where(dist_sq <= (clip_radius_sigma * sigma) ** 2, heatmap, 0.0)
    return heatmap.astype(np.float64, copy=False)


def generate_target_heatmaps(
    keypoints: ArrayLike,
    visibility: ArrayLike,
    heatmap_size: int,
    sigma: float = 2.0,
) -> FloatArray:
    """Generate Gaussian target heatmaps, leaving invisible joints as zero."""

    keypoints_arr = np.asarray(keypoints, dtype=np.float64)
    visibility_arr = np.asarray(visibility, dtype=np.float64)
    single_sample = False

    if keypoints_arr.ndim == 2:
        keypoints_arr = keypoints_arr[np.newaxis, ...]
        visibility_arr = visibility_arr[np.newaxis, ...]
        single_sample = True
    if keypoints_arr.ndim != 3 or keypoints_arr.shape[-1] != 2:
        raise TrainingLossError("keypoints must have shape [B, J, 2] or [J, 2]")
    if visibility_arr.shape != keypoints_arr.shape[:2]:
        raise TrainingLossError("visibility must have shape [B, J] matching keypoints")

    batch, joints = keypoints_arr.shape[:2]
    heatmaps = np.zeros((batch, joints, int(heatmap_size), int(heatmap_size)), dtype=np.float64)
    for batch_index in range(batch):
        for joint_index in range(joints):
            if visibility_arr[batch_index, joint_index] < 0.5:
                continue
            x, y = keypoints_arr[batch_index, joint_index]
            heatmaps[batch_index, joint_index] = generate_gaussian_heatmap(
                float(x),
                float(y),
                heatmap_size,
                sigma,
            )
    return heatmaps[0] if single_sample else heatmaps


def keypoint_heatmap_loss(
    pred_heatmaps: ArrayLike,
    target_heatmaps: ArrayLike,
    visibility: ArrayLike | None = None,
) -> float | Any:
    """Compute MSE over heatmaps, optionally normalized by visible joints."""

    like = _first_torch_tensor(pred_heatmaps, target_heatmaps, visibility)
    if like is not None:
        pred = _to_torch_float(pred_heatmaps, like)
        target = _to_torch_float(target_heatmaps, like)
        if tuple(pred.shape) != tuple(target.shape):
            raise TrainingLossError("pred_heatmaps and target_heatmaps must have the same shape")
        sq_err = (pred - target) ** 2
        if visibility is None:
            return sq_err.mean()
        if pred.ndim < 3:
            raise TrainingLossError("visibility masking requires heatmaps with joint dimensions")
        vis = _to_torch_float(visibility, pred)
        spatial_dims = tuple(range(2, pred.ndim))
        per_joint = sq_err.mean(dim=spatial_dims)
        if tuple(vis.shape) != tuple(per_joint.shape):
            raise TrainingLossError("visibility must match heatmap batch and joint dimensions")
        return (per_joint * vis).sum() / vis.sum().clamp_min(1.0)

    pred = np.asarray(pred_heatmaps, dtype=np.float64)
    target = np.asarray(target_heatmaps, dtype=np.float64)
    if pred.shape != target.shape:
        raise TrainingLossError("pred_heatmaps and target_heatmaps must have the same shape")
    sq_err = (pred - target) ** 2
    if visibility is None:
        return float(np.mean(sq_err))
    if pred.ndim < 3:
        raise TrainingLossError("visibility masking requires heatmaps with joint dimensions")
    vis = np.asarray(visibility, dtype=np.float64)
    spatial_axes = tuple(range(2, pred.ndim))
    per_joint = np.mean(sq_err, axis=spatial_axes)
    if vis.shape != per_joint.shape:
        raise TrainingLossError("visibility must match heatmap batch and joint dimensions")
    return float(np.sum(per_joint * vis) / max(float(np.sum(vis)), 1.0))


def keypoint_coordinate_mse(
    pred_keypoints: ArrayLike,
    target_keypoints: ArrayLike,
    visibility: ArrayLike | None = None,
) -> float | Any:
    """Visibility-masked coordinate MSE for precomputed keypoint predictions."""

    like = _first_torch_tensor(pred_keypoints, target_keypoints, visibility)
    if like is not None:
        pred = _to_torch_float(pred_keypoints, like)
        target = _to_torch_float(target_keypoints, like)
        if tuple(pred.shape) != tuple(target.shape) or pred.shape[-1] != 2:
            raise TrainingLossError("keypoint arrays must have matching shape [..., J, 2]")
        per_joint = ((pred - target) ** 2).mean(dim=-1)
        if visibility is None:
            return per_joint.mean()
        vis = _to_torch_float(visibility, pred)
        if tuple(vis.shape) != tuple(per_joint.shape):
            raise TrainingLossError("visibility must match keypoint batch and joint dimensions")
        return (per_joint * vis).sum() / vis.sum().clamp_min(1.0)

    pred = np.asarray(pred_keypoints, dtype=np.float64)
    target = np.asarray(target_keypoints, dtype=np.float64)
    if pred.shape != target.shape or pred.shape[-1] != 2:
        raise TrainingLossError("keypoint arrays must have matching shape [..., J, 2]")
    per_joint = np.mean((pred - target) ** 2, axis=-1)
    if visibility is None:
        return float(np.mean(per_joint))
    vis = np.asarray(visibility, dtype=np.float64)
    if vis.shape != per_joint.shape:
        raise TrainingLossError("visibility must match keypoint batch and joint dimensions")
    return float(np.sum(per_joint * vis) / max(float(np.sum(vis)), 1.0))


def densepose_part_loss(
    pred_part_logits: ArrayLike,
    target_part_labels: ArrayLike,
    *,
    ignore_index: int = -100,
) -> float | Any:
    """Mean cross-entropy over DensePose part logits."""

    like = _first_torch_tensor(pred_part_logits, target_part_labels)
    if like is not None:
        torch = _torch_module()
        functional = torch.nn.functional
        logits = _to_torch_float(pred_part_logits, like)
        labels = _to_torch_long(target_part_labels, logits)
        return functional.cross_entropy(logits, labels, ignore_index=int(ignore_index))

    logits = np.asarray(pred_part_logits, dtype=np.float64)
    labels = np.asarray(target_part_labels, dtype=np.int64)
    if logits.ndim != 4:
        raise TrainingLossError("pred_part_logits must have shape [B, C, H, W]")
    if labels.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
        raise TrainingLossError("target_part_labels must have shape [B, H, W]")

    valid = labels != int(ignore_index)
    if not np.any(valid):
        return 0.0
    classes = logits.shape[1]
    targets = labels[valid]
    if np.any((targets < 0) | (targets >= classes)):
        raise TrainingLossError("target part labels must be in class range or ignore_index")

    moved = np.moveaxis(logits, 1, -1)
    selected = moved[valid]
    selected = selected - np.max(selected, axis=1, keepdims=True)
    log_probs = selected - np.log(np.sum(np.exp(selected), axis=1, keepdims=True))
    return float(-np.mean(log_probs[np.arange(targets.size), targets]))


def densepose_uv_loss(
    pred_uv: ArrayLike,
    target_uv: ArrayLike,
    target_part_labels: ArrayLike,
    *,
    beta: float = 1.0,
    background_index: int = 0,
) -> float | Any:
    """Foreground-masked Smooth-L1 loss for DensePose UV coordinates."""

    beta = float(beta)
    if beta < 0.0 or not np.isfinite(beta):
        raise TrainingLossError("beta must be finite and non-negative")

    like = _first_torch_tensor(pred_uv, target_uv, target_part_labels)
    if like is not None:
        pred = _to_torch_float(pred_uv, like)
        target = _to_torch_float(target_uv, pred)
        labels = _to_torch_long(target_part_labels, pred)
        if tuple(pred.shape) != tuple(target.shape):
            raise TrainingLossError("pred_uv and target_uv must have the same shape")
        foreground = (labels != int(background_index)) & (labels >= 0)
        mask = foreground.unsqueeze(1).expand_as(pred).to(dtype=pred.dtype)
        denom = mask.sum().clamp_min(1.0)
        diff = (pred - target) * mask
        abs_diff = diff.abs()
        if beta == 0.0:
            smooth = abs_diff
        else:
            smooth = _torch_module().where(abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)
        return smooth.sum() / denom

    pred = np.asarray(pred_uv, dtype=np.float64)
    target = np.asarray(target_uv, dtype=np.float64)
    labels = np.asarray(target_part_labels, dtype=np.int64)
    if pred.shape != target.shape:
        raise TrainingLossError("pred_uv and target_uv must have the same shape")
    if pred.ndim != 4 or labels.shape != (pred.shape[0], pred.shape[2], pred.shape[3]):
        raise TrainingLossError("UV tensors must be [B, C, H, W] with labels [B, H, W]")
    foreground = (labels != int(background_index)) & (labels >= 0)
    mask = np.broadcast_to(foreground[:, np.newaxis, :, :], pred.shape).astype(np.float64)
    denom = max(float(np.sum(mask)), 1.0)
    diff = (pred - target) * mask
    abs_diff = np.abs(diff)
    if beta == 0.0:
        smooth = abs_diff
    else:
        smooth = np.where(abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)
    return float(np.sum(smooth) / denom)


def transfer_loss(student_features: ArrayLike, teacher_features: ArrayLike) -> float | Any:
    """Mean-squared feature distillation loss."""

    like = _first_torch_tensor(student_features, teacher_features)
    if like is not None:
        student = _to_torch_float(student_features, like)
        teacher = _to_torch_float(teacher_features, student)
        if tuple(student.shape) != tuple(teacher.shape):
            raise TrainingLossError("student_features and teacher_features must have the same shape")
        return ((student - teacher) ** 2).mean()

    student = np.asarray(student_features, dtype=np.float64)
    teacher = np.asarray(teacher_features, dtype=np.float64)
    if student.shape != teacher.shape:
        raise TrainingLossError("student_features and teacher_features must have the same shape")
    return float(np.mean((student - teacher) ** 2))


def compute_losses(
    pred_kpt_heatmaps: ArrayLike,
    gt_kpt_heatmaps: ArrayLike,
    visibility: ArrayLike | None = None,
    *,
    pred_part_logits: ArrayLike | None = None,
    gt_part_labels: ArrayLike | None = None,
    pred_uv: ArrayLike | None = None,
    gt_uv: ArrayLike | None = None,
    student_features: ArrayLike | None = None,
    teacher_features: ArrayLike | None = None,
    weights: LossWeights | None = None,
) -> LossComponents:
    """Compute the weighted WiFi-DensePose-style loss breakdown."""

    loss_weights = weights if weights is not None else LossWeights()
    kp_loss = keypoint_heatmap_loss(pred_kpt_heatmaps, gt_kpt_heatmaps, visibility)
    kp_value = _scalar_value(kp_loss)
    total_value = loss_weights.lambda_kp * kp_value
    backend_total = kp_loss * loss_weights.lambda_kp if _is_torch_tensor(kp_loss) else None
    details: dict[str, float] = {"kp_mse": kp_value}

    part_value: float | None = None
    uv_value: float | None = None
    transfer_value: float | None = None

    if pred_part_logits is not None and gt_part_labels is not None:
        part_loss = densepose_part_loss(pred_part_logits, gt_part_labels)
        part_value = _scalar_value(part_loss)
        total_value += loss_weights.lambda_dp * part_value
        backend_total = _add_backend_loss(backend_total, part_loss, loss_weights.lambda_dp)
        details["dp_part_ce"] = part_value

    if pred_uv is not None and gt_uv is not None and gt_part_labels is not None:
        uv_loss = densepose_uv_loss(pred_uv, gt_uv, gt_part_labels)
        uv_value = _scalar_value(uv_loss)
        total_value += loss_weights.lambda_dp * uv_value
        backend_total = _add_backend_loss(backend_total, uv_loss, loss_weights.lambda_dp)
        details["dp_uv_smooth_l1"] = uv_value

    if student_features is not None and teacher_features is not None:
        tr_loss = transfer_loss(student_features, teacher_features)
        transfer_value = _scalar_value(tr_loss)
        total_value += loss_weights.lambda_tr * transfer_value
        backend_total = _add_backend_loss(backend_total, tr_loss, loss_weights.lambda_tr)
        details["transfer_mse"] = transfer_value

    return LossComponents(
        total=float(total_value),
        keypoint=float(kp_value),
        densepose_parts=part_value,
        densepose_uv=uv_value,
        transfer=transfer_value,
        details=details,
        backend_loss=backend_total,
    )


class WiFiDensePoseLoss:
    """Small callable wrapper mirroring the Rust training loss surface."""

    def __init__(self, weights: LossWeights | None = None) -> None:
        self.weights = weights if weights is not None else LossWeights()

    def keypoint_loss(
        self,
        pred_heatmaps: ArrayLike,
        target_heatmaps: ArrayLike,
        visibility: ArrayLike | None = None,
    ) -> float | Any:
        return keypoint_heatmap_loss(pred_heatmaps, target_heatmaps, visibility)

    def densepose_loss(
        self,
        pred_parts: ArrayLike,
        target_parts: ArrayLike,
        pred_uv: ArrayLike,
        target_uv: ArrayLike,
    ) -> float | Any:
        return densepose_part_loss(pred_parts, target_parts) + densepose_uv_loss(
            pred_uv,
            target_uv,
            target_parts,
        )

    def transfer_loss(self, student_features: ArrayLike, teacher_features: ArrayLike) -> float | Any:
        return transfer_loss(student_features, teacher_features)

    def forward(
        self,
        pred_keypoints: ArrayLike,
        target_keypoints: ArrayLike,
        visibility: ArrayLike | None = None,
        **kwargs: Any,
    ) -> LossComponents:
        return compute_losses(
            pred_keypoints,
            target_keypoints,
            visibility,
            weights=self.weights,
            **kwargs,
        )


def _is_torch_tensor(value: Any) -> bool:
    return value is not None and type(value).__module__.split(".", maxsplit=1)[0] == "torch"


def _first_torch_tensor(*values: Any) -> Any | None:
    for value in values:
        if _is_torch_tensor(value):
            return value
    return None


def _torch_module() -> Any:
    import torch

    return torch


def _to_torch_float(value: Any, like: Any) -> Any:
    torch = _torch_module()
    if _is_torch_tensor(value):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _to_torch_long(value: Any, like: Any) -> Any:
    torch = _torch_module()
    if _is_torch_tensor(value):
        return value.to(device=like.device, dtype=torch.long)
    return torch.as_tensor(value, device=like.device, dtype=torch.long)


def _scalar_value(value: Any) -> float:
    if _is_torch_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _add_backend_loss(total: Any | None, loss: Any, weight: float) -> Any | None:
    if not _is_torch_tensor(loss):
        return total
    weighted = loss * float(weight)
    return weighted if total is None else total + weighted
