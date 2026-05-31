from __future__ import annotations

import json

import numpy as np
import pytest

from ruview.nn.tensor import Tensor, TensorError, TensorShape, TensorSpec, broadcast_shape
from ruview.training.config import TrainingConfig
from ruview.training.dataset import (
    CsiBatch,
    DataLoader,
    ReplayCsiDataset,
    SyntheticCsiConfig,
    SyntheticCsiDataset,
    resample_subcarriers,
)


def test_tensor_shape_spec_broadcast_and_numpy_wrapper() -> None:
    shape = TensorShape([1, 3, 1])

    assert shape.ndim == 3
    assert shape.numel == 3
    assert shape.dim(1) == 3
    assert shape.is_broadcast_compatible([2, 1, 4])
    assert broadcast_shape(shape, [2, 1, 4]).dims == (2, 3, 4)
    with pytest.raises(TensorError):
        broadcast_shape([2, 3], [3, 2])

    spec = TensorSpec([None, 3, 3, 8], name="csi")
    array = np.zeros((5, 3, 3, 8), dtype=np.float32)
    assert spec.validate(array) is array
    with pytest.raises(TensorError):
        spec.validate(array.astype(np.float64))

    tensor = Tensor(np.array([[-1.0, 1.0], [2.0, 3.0]], dtype=np.float32))
    np.testing.assert_allclose(tensor.relu().to_numpy(), [[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_allclose(tensor.softmax(axis=1).to_numpy().sum(axis=1), [1.0, 1.0])
    stacked = Tensor.stack([tensor, tensor])
    assert stacked.shape.dims == (2, 2, 2)


def test_training_config_defaults_validation_and_json_roundtrip(tmp_path) -> None:
    cfg = TrainingConfig()

    assert cfg.validate() is cfg
    assert cfg.num_subcarriers == 56
    assert cfg.native_subcarriers == 114
    assert cfg.num_antennas_tx == 3
    assert cfg.num_antennas_rx == 3
    assert cfg.window_frames == 100
    assert cfg.num_keypoints == 17
    assert cfg.num_body_parts == 24
    assert cfg.batch_size == 8
    assert cfg.needs_subcarrier_interp()
    assert TrainingConfig.ht40_192().native_subcarriers == 192
    assert not TrainingConfig.for_subcarriers(56, 56).needs_subcarrier_interp()

    path = tmp_path / "training.json"
    cfg.to_json(path)
    loaded = TrainingConfig.from_json(path)

    assert loaded == cfg
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["checkpoint_dir"] == "checkpoints"
    assert payload["log_dir"] == "logs"

    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(learning_rate=0.0).validate()
    with pytest.raises(ValueError, match="strictly increasing"):
        TrainingConfig(lr_milestones=[30, 30]).validate()
    with pytest.raises(ValueError, match="warmup_epochs"):
        TrainingConfig(num_epochs=5, warmup_epochs=5).validate()


def test_synthetic_dataset_is_deterministic_and_has_expected_shapes() -> None:
    cfg = SyntheticCsiConfig(
        num_subcarriers=8,
        num_antennas_tx=2,
        num_antennas_rx=2,
        window_frames=6,
        num_keypoints=17,
    )
    dataset = SyntheticCsiDataset(4, cfg, subject_id=2, action_id=3, start_frame_id=10)

    sample_a = dataset.get(2)
    sample_b = dataset.get(2)
    sample_c = dataset.get(3)

    assert sample_a.amplitude.shape == (6, 2, 2, 8)
    assert sample_a.phase.shape == (6, 2, 2, 8)
    assert sample_a.keypoints.shape == (17, 2)
    assert sample_a.visibility.shape == (17,)
    assert sample_a.subject_id == 2
    assert sample_a.action_id == 3
    assert sample_a.frame_id == 12
    np.testing.assert_allclose(sample_a.amplitude, sample_b.amplitude)
    np.testing.assert_allclose(sample_a.phase, sample_b.phase)
    np.testing.assert_allclose(sample_a.keypoints, sample_b.keypoints)
    assert not np.allclose(sample_a.amplitude, sample_c.amplitude)
    assert float(np.min(sample_a.amplitude)) >= 0.19
    assert float(np.max(sample_a.amplitude)) <= 0.81
    np.testing.assert_allclose(sample_a.visibility, np.full(17, 2.0, dtype=np.float32))

    with pytest.raises(IndexError):
        dataset.get(4)


def test_sample_signal_features_are_numpy_and_finite() -> None:
    dataset = SyntheticCsiDataset(
        1,
        SyntheticCsiConfig(num_subcarriers=8, num_antennas_tx=1, num_antennas_rx=1, window_frames=4),
    )
    sample = dataset.get(0)

    features = sample.signal_features()

    assert features.shape == (4,)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))
    assert features[0] == pytest.approx(float(np.mean(sample.amplitude)))
    assert features[3] > 0.0
    assert sample.keypoint_visibility is sample.visibility
    tensors = sample.as_tensors()
    assert tensors["amplitude"].shape.dims == sample.amplitude.shape


def test_dataloader_deterministic_shuffle_and_stacked_batches() -> None:
    dataset = SyntheticCsiDataset(
        10,
        SyntheticCsiConfig(num_subcarriers=6, num_antennas_tx=1, num_antennas_rx=1, window_frames=3),
    )

    loader_a = DataLoader(dataset, 4, shuffle=True, seed=99)
    loader_b = DataLoader(dataset, 4, shuffle=True, seed=99)
    loader_c = DataLoader(dataset, 10, shuffle=True, seed=1)
    loader_d = DataLoader(dataset, 10, shuffle=True, seed=2)

    ids_a = [sample.frame_id for batch in loader_a for sample in batch]
    ids_b = [sample.frame_id for batch in loader_b for sample in batch]
    ids_c = [sample.frame_id for batch in loader_c for sample in batch]
    ids_d = [sample.frame_id for batch in loader_d for sample in batch]

    assert loader_a.num_batches() == 3
    assert ids_a == ids_b
    assert sorted(ids_a) == list(range(10))
    assert ids_c != ids_d
    assert [sample.frame_id for sample in DataLoader(dataset, 3).iter_samples()] == list(range(10))

    stacked_loader = DataLoader(dataset, 4, shuffle=False, stack=True)
    first_batch = next(iter(stacked_loader))

    assert isinstance(first_batch, CsiBatch)
    assert len(first_batch) == 4
    assert first_batch.amplitude.shape == (4, 3, 1, 1, 6)
    assert first_batch.keypoints.shape == (4, 17, 2)
    np.testing.assert_array_equal(first_batch.frame_ids, [0, 1, 2, 3])


def test_replay_jsonl_dataset_reads_server_style_rows(tmp_path) -> None:
    path = tmp_path / "recording.jsonl"
    update_row = {
        "type": "sensing_update",
        "source": "fixture",
        "tick": 7,
        "nodes": [{"node_id": "node-1", "amplitude": [1.0, 2.0, 3.0, 4.0]}],
        "features": {"mean_amplitude": 2.5},
        "classification": {"presence": True},
    }
    sample_row = {
        "source": "sample-file",
        "frame_id": 8,
        "amplitude": [[1.0, 2.0, 1.0, 2.0]],
        "phase": [[0.0, 0.1, 0.2, 0.3]],
        "keypoints": [[0.2, 0.3, 2.0], [0.4, 0.5, 1.0]],
    }
    path.write_text("\n".join(json.dumps(row) for row in (update_row, sample_row)), encoding="utf-8")
    config = TrainingConfig(
        num_subcarriers=8,
        native_subcarriers=4,
        window_frames=3,
        num_antennas_tx=1,
        num_antennas_rx=1,
        num_epochs=50,
    )

    dataset = ReplayCsiDataset(path, config)
    first = dataset.get(0)
    second = dataset.get(1)

    assert len(dataset) == 2
    assert first.amplitude.shape == (3, 1, 1, 8)
    assert first.phase.shape == (3, 1, 1, 8)
    assert first.frame_id == 7
    assert first.recording_id == "fixture"
    np.testing.assert_allclose(first.amplitude[0, 0, 0, [0, -1]], [1.0, 4.0])
    np.testing.assert_allclose(first.phase, np.zeros_like(first.phase))
    assert second.frame_id == 8
    assert second.visibility[:2].tolist() == [2.0, 1.0]
    np.testing.assert_allclose(resample_subcarriers([1.0, 3.0], 3), [1.0, 2.0, 3.0])


def test_optional_torch_conversion_hooks() -> None:
    torch = pytest.importorskip("torch")
    dataset = SyntheticCsiDataset(
        2,
        SyntheticCsiConfig(num_subcarriers=4, num_antennas_tx=1, num_antennas_rx=1, window_frames=2),
    )

    sample_tensors = dataset.get(0).to_torch()
    batch_tensors = next(iter(DataLoader(dataset, 2, stack=True))).to_torch()
    tensor = Tensor(np.ones((2, 3), dtype=np.float32)).to_torch()

    assert isinstance(sample_tensors["amplitude"], torch.Tensor)
    assert sample_tensors["amplitude"].shape == (2, 1, 1, 4)
    assert batch_tensors["amplitude"].shape == (2, 2, 1, 1, 4)
    assert tensor.shape == (2, 3)
