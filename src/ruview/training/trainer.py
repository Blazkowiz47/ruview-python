"""Tiny deterministic trainer/evaluator for pose research loops."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ruview.training.losses import (
    TrainingLossError,
    generate_target_heatmaps,
    keypoint_coordinate_mse,
    keypoint_heatmap_loss,
)
from ruview.training.metrics import MetricsResult, evaluate_keypoints, heatmap_to_keypoints


FloatArray = NDArray[np.float64]


class TrainerError(ValueError):
    """Raised when the lightweight trainer cannot consume a batch."""


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for deterministic research-only training loops."""

    epochs: int = 1
    batch_size: int = 1
    heatmap_size: int = 32
    heatmap_sigma: float = 2.0
    pck_threshold: float = 0.2
    shuffle: bool = False
    seed: int = 0
    metric: str = "val_pck"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise TrainerError("epochs must be positive")
        if self.batch_size <= 0:
            raise TrainerError("batch_size must be positive")
        if self.heatmap_size <= 0:
            raise TrainerError("heatmap_size must be positive")
        if self.heatmap_sigma <= 0.0:
            raise TrainerError("heatmap_sigma must be positive")
        if self.pck_threshold < 0.0:
            raise TrainerError("pck_threshold must be non-negative")


@dataclass(frozen=True)
class TrainingExample:
    """One pose-training example with optional precomputed predictions."""

    keypoints: ArrayLike
    visibility: ArrayLike
    inputs: Any = None
    pred_keypoints: ArrayLike | None = None
    pred_heatmaps: ArrayLike | None = None
    target_heatmaps: ArrayLike | None = None


@dataclass(frozen=True)
class EpochRecord:
    """Deterministic per-epoch training/evaluation record."""

    epoch: int
    train_loss: float
    val_pck: float
    val_oks: float
    train_samples: int
    val_samples: int
    visible_keypoints: int
    details: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "train_loss": float(self.train_loss),
            "val_pck": float(self.val_pck),
            "val_oks": float(self.val_oks),
            "train_samples": int(self.train_samples),
            "val_samples": int(self.val_samples),
            "visible_keypoints": int(self.visible_keypoints),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class TrainingRun:
    """Summary of a completed lightweight training run."""

    config: TrainingConfig
    history: tuple[EpochRecord, ...]
    best_epoch: int
    best_metric: float
    final_metrics: MetricsResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "history": [item.to_dict() for item in self.history],
            "best_epoch": int(self.best_epoch),
            "best_metric": float(self.best_metric),
            "final_metrics": self.final_metrics.to_dict(),
        }


@dataclass
class Trainer:
    """Minimal trainer that can replay NumPy predictions or call model hooks."""

    config: TrainingConfig = field(default_factory=TrainingConfig)
    model: Any | None = None
    optimizer: Any | None = None

    def fit(
        self,
        train_data: Iterable[TrainingExample | Mapping[str, Any]],
        val_data: Iterable[TrainingExample | Mapping[str, Any]] | None = None,
        *,
        model: Any | None = None,
        optimizer: Any | None = None,
    ) -> TrainingRun:
        train_examples = _examples(train_data)
        if not train_examples:
            raise TrainerError("train_data must contain at least one example")
        val_examples = _examples(val_data) if val_data is not None else train_examples
        active_model = self.model if model is None else model
        active_optimizer = self.optimizer if optimizer is None else optimizer

        history: list[EpochRecord] = []
        best_epoch = 0
        best_metric = -float("inf")
        final_metrics = MetricsResult()

        for epoch in range(1, self.config.epochs + 1):
            losses = []
            for batch in self._iter_batches(train_examples, epoch=epoch, shuffle=self.config.shuffle):
                loss_value, backend_loss = self._training_batch_loss(
                    batch,
                    active_model,
                    epoch=epoch,
                    optimizer=active_optimizer,
                )
                losses.append(loss_value)
                if active_optimizer is not None and backend_loss is not None:
                    active_optimizer.zero_grad()
                    backend_loss.backward()
                    active_optimizer.step()

            final_metrics = self.evaluate(val_examples, model=active_model)
            metric_value = _metric_value(final_metrics, self.config.metric)
            if metric_value > best_metric:
                best_metric = metric_value
                best_epoch = epoch
            history.append(
                EpochRecord(
                    epoch=epoch,
                    train_loss=float(np.mean(losses)) if losses else 0.0,
                    val_pck=float(final_metrics.pck),
                    val_oks=float(final_metrics.oks),
                    train_samples=len(train_examples),
                    val_samples=len(val_examples),
                    visible_keypoints=int(final_metrics.num_keypoints),
                    details={"batches": float(len(losses))},
                )
            )

        return TrainingRun(
            config=self.config,
            history=tuple(history),
            best_epoch=best_epoch,
            best_metric=float(best_metric if np.isfinite(best_metric) else 0.0),
            final_metrics=final_metrics,
        )

    def evaluate(
        self,
        data: Iterable[TrainingExample | Mapping[str, Any]],
        *,
        model: Any | None = None,
    ) -> MetricsResult:
        examples = _examples(data)
        if not examples:
            return MetricsResult()
        active_model = self.model if model is None else model
        pred_frames: list[FloatArray] = []
        gt_frames: list[FloatArray] = []
        vis_frames: list[FloatArray] = []

        for batch in self._iter_batches(examples, epoch=1, shuffle=False):
            prediction = _normalize_prediction(
                _call_model(active_model, batch, epoch=1, training=False)
                if active_model is not None
                else {
                    "pred_keypoints": batch.get("pred_keypoints"),
                    "pred_heatmaps": batch.get("pred_heatmaps"),
                }
            )
            pred_keypoints = _prediction_keypoints(prediction)
            if pred_keypoints is None:
                raise TrainerError("evaluation requires pred_keypoints or pred_heatmaps")
            pred_np = _to_numpy(pred_keypoints)
            gt_np = _to_numpy(batch["keypoints"])
            vis_np = _to_numpy(batch["visibility"])
            for index in range(gt_np.shape[0]):
                pred_frames.append(pred_np[index])
                gt_frames.append(gt_np[index])
                vis_frames.append(vis_np[index])

        return evaluate_keypoints(
            np.stack(pred_frames),
            np.stack(gt_frames),
            np.stack(vis_frames),
            pck_threshold=self.config.pck_threshold,
        )

    def _iter_batches(
        self,
        examples: Sequence[TrainingExample | Mapping[str, Any]],
        *,
        epoch: int,
        shuffle: bool,
    ) -> Iterable[dict[str, Any]]:
        indices = np.arange(len(examples))
        if shuffle:
            rng = np.random.default_rng(int(self.config.seed) + int(epoch))
            rng.shuffle(indices)
        for start in range(0, len(indices), self.config.batch_size):
            yield _make_batch([examples[int(index)] for index in indices[start : start + self.config.batch_size]])

    def _training_batch_loss(
        self,
        batch: Mapping[str, Any],
        model: Any | None,
        *,
        epoch: int,
        optimizer: Any | None,
    ) -> tuple[float, Any | None]:
        prediction = _normalize_prediction(
            _call_model(model, batch, epoch=epoch, training=True)
            if model is not None
            else {
                "pred_keypoints": batch.get("pred_keypoints"),
                "pred_heatmaps": batch.get("pred_heatmaps"),
            }
        )
        if prediction.get("loss") is not None:
            loss = prediction["loss"]
            return _scalar(loss), loss if _is_torch_tensor(loss) else None

        pred_heatmaps = prediction.get("pred_heatmaps")
        if pred_heatmaps is not None:
            target_heatmaps = batch.get("target_heatmaps")
            if target_heatmaps is None:
                target_heatmaps = generate_target_heatmaps(
                    batch["keypoints"],
                    batch["visibility"],
                    self.config.heatmap_size,
                    self.config.heatmap_sigma,
                )
            loss = keypoint_heatmap_loss(pred_heatmaps, target_heatmaps, batch["visibility"])
            return _scalar(loss), loss if _is_torch_tensor(loss) and optimizer is not None else None

        pred_keypoints = prediction.get("pred_keypoints")
        if pred_keypoints is not None:
            loss = keypoint_coordinate_mse(pred_keypoints, batch["keypoints"], batch["visibility"])
            return _scalar(loss), loss if _is_torch_tensor(loss) and optimizer is not None else None

        raise TrainerError("training requires a loss, pred_keypoints, or pred_heatmaps")


def _examples(data: Iterable[TrainingExample | Mapping[str, Any]] | None) -> list[TrainingExample | Mapping[str, Any]]:
    if data is None:
        return []
    if isinstance(data, (TrainingExample, Mapping)):
        return [data]
    return list(data)


def _make_batch(examples: Sequence[TrainingExample | Mapping[str, Any]]) -> dict[str, Any]:
    keypoints = np.stack([_to_numpy(_get(item, "keypoints", "target_keypoints", "gt_keypoints")) for item in examples])
    visibility = np.stack([_to_numpy(_get(item, "visibility", "keypoint_visibility", "vis")) for item in examples])
    batch: dict[str, Any] = {
        "inputs": _stack_optional([_get(item, "inputs", "input", "x", default=None) for item in examples]),
        "keypoints": keypoints,
        "visibility": visibility,
    }
    for output_key, aliases in {
        "pred_keypoints": ("pred_keypoints", "prediction_keypoints", "pred_kpts"),
        "pred_heatmaps": ("pred_heatmaps", "pred_keypoint_heatmaps", "heatmaps"),
        "target_heatmaps": ("target_heatmaps", "gt_heatmaps"),
    }.items():
        values = [_get(item, *aliases, default=None) for item in examples]
        if all(value is not None for value in values):
            batch[output_key] = _stack_optional(values)
        else:
            batch[output_key] = None
    return batch


def _get(item: TrainingExample | Mapping[str, Any], *names: str, default: Any = ...) -> Any:
    for name in names:
        if isinstance(item, Mapping) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    if default is not ...:
        return default
    raise TrainerError(f"example is missing required field {names[0]!r}")


def _stack_optional(values: Sequence[Any]) -> Any:
    if all(value is None for value in values):
        return None
    if any(_is_torch_tensor(value) for value in values if value is not None):
        torch = _torch_module()
        tensors = [value if _is_torch_tensor(value) else torch.as_tensor(value) for value in values]
        return torch.stack(tensors)
    try:
        return np.stack([np.asarray(value) for value in values])
    except (TypeError, ValueError):
        return list(values)


def _call_model(model: Any, batch: Mapping[str, Any], *, epoch: int, training: bool) -> Any:
    if training and hasattr(model, "train_step"):
        return model.train_step(batch, epoch=epoch)
    if not training and hasattr(model, "predict"):
        return model.predict(batch["inputs"])
    if callable(model):
        try:
            return model(batch, epoch=epoch, training=training)
        except TypeError:
            try:
                return model(batch)
            except TypeError:
                return model(batch["inputs"])
    raise TrainerError("model must be callable or provide train_step/predict")


def _normalize_prediction(output: Any) -> dict[str, Any]:
    if isinstance(output, Mapping):
        return {
            "loss": _first_present(output, "loss", "train_loss"),
            "pred_keypoints": _first_present(output, "pred_keypoints", "keypoints", "pred_kpts"),
            "pred_heatmaps": _first_present(output, "pred_heatmaps", "heatmaps", "keypoint_heatmaps"),
        }
    shape = tuple(output.shape) if hasattr(output, "shape") else tuple(np.asarray(output).shape)
    if len(shape) >= 2 and shape[-1] == 2:
        return {"loss": None, "pred_keypoints": output, "pred_heatmaps": None}
    if len(shape) == 4:
        return {"loss": None, "pred_keypoints": None, "pred_heatmaps": output}
    raise TrainerError("model output must be a mapping, keypoint array, or heatmap array")


def _prediction_keypoints(prediction: Mapping[str, Any]) -> Any | None:
    if prediction.get("pred_keypoints") is not None:
        return prediction["pred_keypoints"]
    if prediction.get("pred_heatmaps") is not None:
        return heatmap_to_keypoints(_to_numpy(prediction["pred_heatmaps"]))
    return None


def _first_present(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _metric_value(metrics: MetricsResult, metric: str) -> float:
    aliases = {
        "pck": metrics.pck,
        "val_pck": metrics.pck,
        "oks": metrics.oks,
        "val_oks": metrics.oks,
    }
    if metric not in aliases:
        raise TrainerError(f"unsupported trainer metric {metric!r}")
    return float(aliases[metric])


def _is_torch_tensor(value: Any) -> bool:
    return value is not None and type(value).__module__.split(".", maxsplit=1)[0] == "torch"


def _torch_module() -> Any:
    import torch

    return torch


def _to_numpy(value: Any) -> FloatArray:
    if _is_torch_tensor(value):
        value = value.detach().cpu().numpy()
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TrainerError("value cannot be converted to a numeric numpy array") from exc


def _scalar(value: Any) -> float:
    if _is_torch_tensor(value):
        return float(value.detach().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingLossError("loss value must be scalar") from exc
