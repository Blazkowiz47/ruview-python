"""PCK and OKS metrics for pose-training research experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]

NUM_COCO_KEYPOINTS = 17
COCO_KEYPOINT_SIGMAS: tuple[float, ...] = (
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
)


class TrainingMetricError(ValueError):
    """Raised when metric inputs are malformed."""


@dataclass(frozen=True)
class MetricsResult:
    """Aggregated pose metric result."""

    pck: float = 0.0
    oks: float = 0.0
    num_keypoints: int = 0
    num_samples: int = 0

    def is_better_than(self, other: "MetricsResult") -> bool:
        return self.pck > other.pck

    def summary(self) -> str:
        return (
            f"PCK@0.2={self.pck:.4f} OKS={self.oks:.4f} "
            f"(n_samples={self.num_samples} n_kp={self.num_keypoints})"
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "pck": float(self.pck),
            "oks": float(self.oks),
            "num_keypoints": int(self.num_keypoints),
            "num_samples": int(self.num_samples),
        }


@dataclass(frozen=True)
class AggregatedMetrics:
    """PCK and OKS report over a sequence of pose frames."""

    pck_02: float = 0.0
    pck_05: float = 0.0
    per_joint_pck: tuple[float, ...] = field(default_factory=lambda: (0.0,) * NUM_COCO_KEYPOINTS)
    oks: float = 0.0
    oks_values: tuple[float, ...] = ()
    frames_evaluated: int = 0
    keypoints_evaluated: int = 0


class MetricsAccumulator:
    """Running accumulator for PCK@threshold and OKS."""

    def __init__(self, pck_threshold: float = 0.2, *, normalizer: str | float = "bbox") -> None:
        if pck_threshold < 0.0 or not np.isfinite(pck_threshold):
            raise TrainingMetricError("pck_threshold must be finite and non-negative")
        self.pck_threshold = float(pck_threshold)
        self.normalizer = normalizer
        self.reset()

    def update(self, pred_kp: ArrayLike, gt_kp: ArrayLike, visibility: ArrayLike) -> None:
        correct, total, pck = compute_pck(
            pred_kp,
            gt_kp,
            visibility,
            threshold=self.pck_threshold,
            normalizer=self.normalizer,
        )
        oks = compute_oks(pred_kp, gt_kp, visibility)
        self._pck_sum += pck
        self._oks_sum += oks
        self._num_keypoints += total
        self._num_samples += 1
        self._num_correct += correct

    def finalize(self) -> MetricsResult | None:
        if self._num_samples == 0:
            return None
        return MetricsResult(
            pck=float(self._pck_sum / self._num_samples),
            oks=float(self._oks_sum / self._num_samples),
            num_keypoints=int(self._num_keypoints),
            num_samples=int(self._num_samples),
        )

    @property
    def num_samples(self) -> int:
        return int(self._num_samples)

    @property
    def num_keypoints(self) -> int:
        return int(self._num_keypoints)

    def reset(self) -> None:
        self._pck_sum = 0.0
        self._oks_sum = 0.0
        self._num_keypoints = 0
        self._num_samples = 0
        self._num_correct = 0


def bounding_box_diagonal(keypoints: ArrayLike, visibility: ArrayLike) -> float:
    """Return the visible-keypoint bounding-box diagonal."""

    kp = _coerce_keypoints(keypoints)
    vis = _coerce_visibility(visibility, kp.shape[0])
    visible = vis >= 0.5
    if not np.any(visible):
        return 0.0
    coords = kp[visible]
    width = float(np.max(coords[:, 0]) - np.min(coords[:, 0]))
    height = float(np.max(coords[:, 1]) - np.min(coords[:, 1]))
    return float(np.hypot(max(width, 0.0), max(height, 0.0)))


def torso_diameter(keypoints: ArrayLike, visibility: ArrayLike) -> float:
    """Return the COCO left-hip to right-shoulder distance when visible."""

    kp = _coerce_keypoints(keypoints)
    vis = _coerce_visibility(visibility, kp.shape[0])
    left_hip = 11
    right_shoulder = 6
    if kp.shape[0] <= max(left_hip, right_shoulder):
        return 0.0
    if vis[left_hip] < 0.5 or vis[right_shoulder] < 0.5:
        return 0.0
    return float(np.linalg.norm(kp[left_hip] - kp[right_shoulder]))


def compute_pck(
    pred_kpts: ArrayLike,
    gt_kpts: ArrayLike,
    visibility: ArrayLike,
    threshold: float = 0.2,
    *,
    normalizer: str | float = "bbox",
) -> tuple[int, int, float]:
    """Compute single-frame Percentage of Correct Keypoints."""

    pred, gt, vis = _aligned_pose_arrays(pred_kpts, gt_kpts, visibility)
    dist_threshold = float(threshold) * _normalizer_value(gt, vis, normalizer)
    correct = 0
    total = 0
    for index in range(gt.shape[0]):
        if vis[index] < 0.5:
            continue
        total += 1
        distance = float(np.linalg.norm(pred[index] - gt[index]))
        if distance <= dist_threshold:
            correct += 1
    pck = float(correct / total) if total else 0.0
    return correct, total, pck


def compute_per_joint_pck(
    pred_batch: Sequence[ArrayLike] | ArrayLike,
    gt_batch: Sequence[ArrayLike] | ArrayLike,
    vis_batch: Sequence[ArrayLike] | ArrayLike,
    threshold: float = 0.2,
    *,
    normalizer: str | float = "bbox",
) -> tuple[float, ...]:
    """Compute per-joint PCK over a batch of frames."""

    pred_frames = _as_pose_batch(pred_batch)
    gt_frames = _as_pose_batch(gt_batch)
    vis_frames = _as_visibility_batch(vis_batch)
    if len(pred_frames) != len(gt_frames) or len(pred_frames) != len(vis_frames):
        raise TrainingMetricError("pred, gt, and visibility batches must have the same length")

    joint_count = max((frame.shape[0] for frame in gt_frames), default=NUM_COCO_KEYPOINTS)
    correct = np.zeros(joint_count, dtype=np.float64)
    total = np.zeros(joint_count, dtype=np.float64)
    for pred, gt, vis in zip(pred_frames, gt_frames, vis_frames, strict=True):
        pred_arr, gt_arr, vis_arr = _aligned_pose_arrays(pred, gt, vis)
        dist_threshold = float(threshold) * _normalizer_value(gt_arr, vis_arr, normalizer)
        for joint in range(gt_arr.shape[0]):
            if vis_arr[joint] < 0.5:
                continue
            total[joint] += 1.0
            if np.linalg.norm(pred_arr[joint] - gt_arr[joint]) <= dist_threshold:
                correct[joint] += 1.0
    result = np.divide(correct, total, out=np.zeros_like(correct), where=total > 0.0)
    return tuple(float(value) for value in result)


def compute_oks(
    pred_kpts: ArrayLike,
    gt_kpts: ArrayLike,
    visibility: ArrayLike,
    object_scale: float | None = None,
) -> float:
    """Compute COCO-style Object Keypoint Similarity for one person."""

    pred, gt, vis = _aligned_pose_arrays(pred_kpts, gt_kpts, visibility)
    if object_scale is None:
        scale = max(bounding_box_diagonal(gt, vis), 1e-3)
    else:
        scale = max(float(object_scale), 1e-3)

    numerator = 0.0
    denominator = 0
    sigmas = np.asarray(COCO_KEYPOINT_SIGMAS, dtype=np.float64)
    for joint in range(gt.shape[0]):
        if vis[joint] < 0.5:
            continue
        sigma = float(sigmas[joint]) if joint < sigmas.size else 0.07
        dist_sq = float(np.sum((pred[joint] - gt[joint]) ** 2))
        numerator += float(np.exp(-dist_sq / (2.0 * scale * scale * sigma * sigma + 1e-12)))
        denominator += 1
    return float(numerator / denominator) if denominator else 0.0


def aggregate_metrics(
    pred_kpts: Sequence[ArrayLike] | ArrayLike,
    gt_kpts: Sequence[ArrayLike] | ArrayLike,
    visibility: Sequence[ArrayLike] | ArrayLike,
) -> AggregatedMetrics:
    """Aggregate PCK@0.2, PCK@0.5, per-joint PCK, and OKS."""

    pred_frames = _as_pose_batch(pred_kpts)
    gt_frames = _as_pose_batch(gt_kpts)
    vis_frames = _as_visibility_batch(visibility)
    if len(pred_frames) != len(gt_frames) or len(pred_frames) != len(vis_frames):
        raise TrainingMetricError("pred, gt, and visibility batches must have the same length")
    if not pred_frames:
        return AggregatedMetrics()

    pck02 = []
    pck05 = []
    oks_values = []
    total_keypoints = 0
    for pred, gt, vis in zip(pred_frames, gt_frames, vis_frames, strict=True):
        _, total, value02 = compute_pck(pred, gt, vis, threshold=0.2)
        _, _, value05 = compute_pck(pred, gt, vis, threshold=0.5)
        pck02.append(value02)
        pck05.append(value05)
        oks_values.append(compute_oks(pred, gt, vis))
        total_keypoints += total

    return AggregatedMetrics(
        pck_02=float(np.mean(pck02)),
        pck_05=float(np.mean(pck05)),
        per_joint_pck=compute_per_joint_pck(pred_frames, gt_frames, vis_frames, threshold=0.2),
        oks=float(np.mean(oks_values)),
        oks_values=tuple(float(value) for value in oks_values),
        frames_evaluated=len(pred_frames),
        keypoints_evaluated=int(total_keypoints),
    )


def heatmap_to_keypoints(heatmaps: ArrayLike) -> FloatArray:
    """Convert heatmaps to normalized keypoints by spatial argmax."""

    maps = np.asarray(heatmaps, dtype=np.float64)
    single_sample = False
    if maps.ndim == 3:
        maps = maps[np.newaxis, ...]
        single_sample = True
    if maps.ndim != 4:
        raise TrainingMetricError("heatmaps must have shape [B, J, H, W] or [J, H, W]")
    batch, joints, height, width = maps.shape
    flat = maps.reshape(batch, joints, height * width)
    argmax = np.argmax(flat, axis=-1)
    rows = argmax // width
    cols = argmax % width
    x = cols.astype(np.float64) / float(max(width - 1, 1))
    y = rows.astype(np.float64) / float(max(height - 1, 1))
    keypoints = np.stack([x, y], axis=-1)
    return keypoints[0] if single_sample else keypoints


def evaluate_keypoints(
    pred_kpts: Sequence[ArrayLike] | ArrayLike,
    gt_kpts: Sequence[ArrayLike] | ArrayLike,
    visibility: Sequence[ArrayLike] | ArrayLike,
    *,
    pck_threshold: float = 0.2,
) -> MetricsResult:
    """Evaluate a batch of predicted keypoints."""

    pred_frames = _as_pose_batch(pred_kpts)
    gt_frames = _as_pose_batch(gt_kpts)
    vis_frames = _as_visibility_batch(visibility)
    if len(pred_frames) != len(gt_frames) or len(pred_frames) != len(vis_frames):
        raise TrainingMetricError("pred, gt, and visibility batches must have the same length")
    acc = MetricsAccumulator(pck_threshold)
    for pred, gt, vis in zip(pred_frames, gt_frames, vis_frames, strict=True):
        acc.update(pred, gt, vis)
    return acc.finalize() or MetricsResult()


def _coerce_keypoints(values: ArrayLike) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise TrainingMetricError("keypoints must have shape [J, 2]")
    if not np.all(np.isfinite(arr)):
        raise TrainingMetricError("keypoints must be finite")
    return arr


def _coerce_visibility(values: ArrayLike, joints: int) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < joints:
        raise TrainingMetricError("visibility must have at least one value per keypoint")
    return arr[:joints]


def _aligned_pose_arrays(
    pred_kpts: ArrayLike,
    gt_kpts: ArrayLike,
    visibility: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    pred = _coerce_keypoints(pred_kpts)
    gt = _coerce_keypoints(gt_kpts)
    joints = min(pred.shape[0], gt.shape[0])
    vis = _coerce_visibility(visibility, joints)
    return pred[:joints], gt[:joints], vis


def _normalizer_value(gt: FloatArray, visibility: FloatArray, normalizer: str | float) -> float:
    if isinstance(normalizer, (int, float)):
        return max(float(normalizer), 1e-3)
    normalized = normalizer.lower().strip()
    if normalized == "bbox":
        return max(bounding_box_diagonal(gt, visibility), 1e-3)
    if normalized == "torso":
        return max(torso_diameter(gt, visibility), 1.0)
    if normalized == "unit":
        return 1.0
    raise TrainingMetricError("normalizer must be 'bbox', 'torso', 'unit', or a positive scale")


def _as_pose_batch(values: Sequence[ArrayLike] | ArrayLike) -> list[FloatArray]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return [arr[index] for index in range(arr.shape[0])]
    if arr.ndim == 2 and arr.shape[-1] == 2:
        return [arr]
    if isinstance(values, Iterable):
        return [_coerce_keypoints(item) for item in values]  # type: ignore[arg-type]
    raise TrainingMetricError("pose batch must be [B, J, 2] or an iterable of [J, 2] arrays")


def _as_visibility_batch(values: Sequence[ArrayLike] | ArrayLike) -> list[FloatArray]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 2:
        return [arr[index].reshape(-1) for index in range(arr.shape[0])]
    if arr.ndim == 1:
        return [arr.reshape(-1)]
    if isinstance(values, Iterable):
        return [np.asarray(item, dtype=np.float64).reshape(-1) for item in values]  # type: ignore[arg-type]
    raise TrainingMetricError("visibility batch must be [B, J] or an iterable of [J] arrays")
