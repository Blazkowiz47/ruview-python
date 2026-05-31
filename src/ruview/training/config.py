"""Training configuration for WiFi-DensePose research experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self


@dataclass
class TrainingConfig:
    """Complete configuration for a CSI-to-pose training run."""

    num_subcarriers: int = 56
    native_subcarriers: int = 114
    num_antennas_tx: int = 3
    num_antennas_rx: int = 3
    window_frames: int = 100
    heatmap_size: int = 56
    num_keypoints: int = 17
    num_body_parts: int = 24
    backbone_channels: int = 256
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 50
    warmup_epochs: int = 5
    lr_milestones: list[int] = field(default_factory=lambda: [30, 45])
    lr_gamma: float = 0.1
    grad_clip_norm: float = 1.0
    lambda_kp: float = 0.3
    lambda_dp: float = 0.6
    lambda_tr: float = 0.1
    val_every_epochs: int = 1
    early_stopping_patience: int = 10
    checkpoint_dir: Path | str = Path("checkpoints")
    log_dir: Path | str = Path("logs")
    save_top_k: int = 3
    use_gpu: bool = False
    gpu_device_id: int = 0
    num_workers: int = 4
    seed: int = 42

    def __post_init__(self) -> None:
        self.num_subcarriers = int(self.num_subcarriers)
        self.native_subcarriers = int(self.native_subcarriers)
        self.num_antennas_tx = int(self.num_antennas_tx)
        self.num_antennas_rx = int(self.num_antennas_rx)
        self.window_frames = int(self.window_frames)
        self.heatmap_size = int(self.heatmap_size)
        self.num_keypoints = int(self.num_keypoints)
        self.num_body_parts = int(self.num_body_parts)
        self.backbone_channels = int(self.backbone_channels)
        self.batch_size = int(self.batch_size)
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)
        self.num_epochs = int(self.num_epochs)
        self.warmup_epochs = int(self.warmup_epochs)
        self.lr_milestones = [int(value) for value in self.lr_milestones]
        self.lr_gamma = float(self.lr_gamma)
        self.grad_clip_norm = float(self.grad_clip_norm)
        self.lambda_kp = float(self.lambda_kp)
        self.lambda_dp = float(self.lambda_dp)
        self.lambda_tr = float(self.lambda_tr)
        self.val_every_epochs = int(self.val_every_epochs)
        self.early_stopping_patience = int(self.early_stopping_patience)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)
        self.save_top_k = int(self.save_top_k)
        self.use_gpu = bool(self.use_gpu)
        self.gpu_device_id = int(self.gpu_device_id)
        self.num_workers = int(self.num_workers)
        self.seed = int(self.seed)

    @classmethod
    def for_subcarriers(cls, native: int, target: int) -> Self:
        """Build a config that resamples ``native`` subcarriers to ``target``."""

        return cls(native_subcarriers=native, num_subcarriers=target)

    @classmethod
    def mmfi(cls) -> Self:
        """MM-Fi preset: 114 native subcarriers to 56 model subcarriers."""

        return cls()

    @classmethod
    def ht40_192(cls) -> Self:
        """ESP32 HT40 preset: about 192 native subcarriers to 56."""

        return cls.for_subcarriers(192, 56)

    @classmethod
    def multiband_168(cls) -> Self:
        """Multi-band mesh preset: 168 native subcarriers to 56."""

        return cls.for_subcarriers(168, 56)

    def needs_subcarrier_interp(self) -> bool:
        """Return whether dataset subcarriers need interpolation."""

        return self.native_subcarriers != self.num_subcarriers

    def validate(self) -> Self:
        """Validate all fields and return ``self`` for fluent use."""

        _require_positive_int("num_subcarriers", self.num_subcarriers)
        _require_positive_int("native_subcarriers", self.native_subcarriers)
        _require_positive_int("num_antennas_tx", self.num_antennas_tx)
        _require_positive_int("num_antennas_rx", self.num_antennas_rx)
        _require_positive_int("window_frames", self.window_frames)
        _require_positive_int("heatmap_size", self.heatmap_size)
        _require_positive_int("num_keypoints", self.num_keypoints)
        _require_positive_int("num_body_parts", self.num_body_parts)
        _require_positive_int("backbone_channels", self.backbone_channels)
        _require_positive_int("batch_size", self.batch_size)
        _require_positive_int("num_epochs", self.num_epochs)
        _require_positive_int("val_every_epochs", self.val_every_epochs)
        _require_positive_int("early_stopping_patience", self.early_stopping_patience)
        _require_positive_int("save_top_k", self.save_top_k)
        _require_non_negative_int("gpu_device_id", self.gpu_device_id)
        _require_non_negative_int("num_workers", self.num_workers)
        _require_non_negative_int("seed", self.seed)

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0.0")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0.0")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be > 0.0")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if self.warmup_epochs >= self.num_epochs:
            raise ValueError("warmup_epochs must be < num_epochs")
        if self.lr_gamma <= 0.0 or self.lr_gamma >= 1.0:
            raise ValueError("lr_gamma must be in (0.0, 1.0)")

        previous = 0
        for milestone in self.lr_milestones:
            if milestone <= 0 or milestone > self.num_epochs:
                raise ValueError("lr_milestones must be within [1, num_epochs]")
            if milestone <= previous:
                raise ValueError("lr_milestones must be strictly increasing")
            previous = milestone

        for name, value in (
            ("lambda_kp", self.lambda_kp),
            ("lambda_dp", self.lambda_dp),
            ("lambda_tr", self.lambda_tr),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0.0")
        if self.lambda_kp + self.lambda_dp + self.lambda_tr <= 0.0:
            raise ValueError("at least one loss weight must be > 0.0")

        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        payload = asdict(self)
        payload["checkpoint_dir"] = str(self.checkpoint_dir)
        payload["log_dir"] = str(self.log_dir)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, validate: bool = True) -> Self:
        """Build a config from a mapping."""

        if not isinstance(payload, Mapping):
            raise TypeError("TrainingConfig.from_dict expects a mapping")
        cfg = cls(**dict(payload))
        if validate:
            cfg.validate()
        return cfg

    def to_json_string(self) -> str:
        """Serialize this config to pretty JSON."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json_string(cls, contents: str, *, validate: bool = True) -> Self:
        """Deserialize a config from JSON text."""

        payload = json.loads(contents)
        if not isinstance(payload, Mapping):
            raise ValueError("training config JSON must contain an object")
        return cls.from_dict(payload, validate=validate)

    def to_json(self, path: str | Path) -> Path:
        """Write this config to a JSON file and return the path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json_string() + "\n", encoding="utf-8")
        return output_path

    @classmethod
    def from_json(cls, path: str | Path, *, validate: bool = True) -> Self:
        """Load a config from a JSON file."""

        return cls.from_json_string(Path(path).read_text(encoding="utf-8"), validate=validate)


def _require_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


__all__ = ["TrainingConfig"]
