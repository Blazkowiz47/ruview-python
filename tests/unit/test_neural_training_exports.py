from __future__ import annotations

import numpy as np

import ruview.nn as rvnn
import ruview.training as train


def test_milestone_10_public_nn_exports_are_available() -> None:
    tensor = rvnn.Tensor.ones((2, 3))
    clean_phase = rvnn.phase_sanitize(np.array([[0.0, 0.5, 1.5]]))

    assert tensor.shape.dims == (2, 3)
    assert rvnn.TensorShape([1, 3]).is_broadcast_compatible([2, 1])
    assert clean_phase.tolist() == [[0.0, 0.5, 1.0]]
    assert rvnn.triplet_loss(np.zeros(2), np.zeros(2), np.ones(2)) == 0.0
    assert rvnn.TaskKind.POSE.value == "pose"
    assert "CsiToPoseTransformer" in rvnn.__all__


def test_milestone_10_public_training_exports_are_available() -> None:
    cfg = train.TrainingConfig(num_epochs=2, warmup_epochs=1, lr_milestones=[2])
    dataset = train.SyntheticCsiDataset(
        2,
        train.SyntheticCsiConfig(
            num_subcarriers=4,
            num_antennas_tx=1,
            num_antennas_rx=1,
            window_frames=3,
        ),
    )
    sample = dataset.get(0)
    heatmaps = train.generate_target_heatmaps(
        np.expand_dims(sample.keypoints, axis=0),
        np.expand_dims(sample.visibility, axis=0),
        heatmap_size=8,
    )

    assert cfg.validate() is cfg
    assert sample.amplitude.shape == (3, 1, 1, 4)
    assert heatmaps.shape == (1, 17, 8, 8)
    assert train.compute_pck(sample.keypoints, sample.keypoints, sample.visibility)[2] == 1.0
    assert train.ExportTensorSpec("csi", (1, 2), "float32").name == "csi"
    assert "TrainerConfig" in train.__all__
