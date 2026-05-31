from __future__ import annotations

import numpy as np
import pytest

from ruview.nn.encoders import (
    ContrastiveBatcher,
    calibration_robustness_loss,
    cosine_similarity_matrix,
    phase_sanitize,
    triplet_loss,
)


def test_numpy_contrastive_losses_and_batcher() -> None:
    anchor = np.zeros(4)
    positive = np.full(4, 0.1)
    negative = np.ones(4)

    assert triplet_loss(anchor, positive, negative, margin=0.2) == 0.0
    assert triplet_loss(anchor, negative, positive, margin=0.2) > 0.0
    assert calibration_robustness_loss(anchor, anchor) == 0.0
    assert calibration_robustness_loss(anchor, negative) == 1.0

    logits = cosine_similarity_matrix(np.eye(3), temperature=0.5)
    assert logits.shape == (3, 3)
    np.testing.assert_allclose(np.diag(logits), np.full(3, 2.0))

    batcher = ContrastiveBatcher(state_labels=["still", "still", "walk"], environment_labels=["a", "b", "a"])
    triplets = batcher.triplets()
    assert triplets[0].anchor == 0
    assert triplets[0].positive == 1
    assert triplets[0].negative == 2
    assert all(triplet.anchor != 2 for triplet in triplets)


def test_phase_sanitize_numpy_captures_ramp() -> None:
    phase = np.broadcast_to(np.arange(5, dtype=float), (2, 3, 5))
    clean = phase_sanitize(phase)

    np.testing.assert_allclose(clean[..., 0], 0.0)
    np.testing.assert_allclose(clean[..., 1:], 1.0)


def test_rf_encoder_forward_shape_and_normalization() -> None:
    torch = pytest.importorskip("torch")
    from ruview.nn.encoders import RFEncoder, RFEncoderConfig

    torch.manual_seed(0)
    config = RFEncoderConfig(
        embedding_dim=32,
        conv_channels=(8, 16),
        hidden_dim=32,
        dropout=0.0,
        use_transformer=True,
        transformer_heads=4,
    )
    model = RFEncoder(config)
    amplitude = torch.randn(2, 4, 8)
    phase = torch.randn(2, 4, 8)

    embedding = model(amplitude, phase)

    assert tuple(embedding.shape) == (2, 32)
    torch.testing.assert_close(embedding.norm(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="same shape"):
        model(amplitude, phase[:, :3, :])


def test_densepose_head_outputs_expected_maps() -> None:
    torch = pytest.importorskip("torch")
    from ruview.nn.heads import DensePoseHead

    torch.manual_seed(0)
    head = DensePoseHead(input_channels=16, hidden_channels=(16,), output_size=(10, 12), dropout=0.0)
    features = torch.randn(2, 16, 5, 6)

    out = head(features)

    assert tuple(out.keypoint_heatmaps.shape) == (2, 17, 10, 12)
    assert tuple(out.part_logits.shape) == (2, 25, 10, 12)
    assert tuple(out.uv_coordinates.shape) == (2, 48, 10, 12)
    assert tuple(out.confidence.shape) == (2, 1, 10, 12)
    assert float(out.uv_coordinates.min()) >= 0.0
    assert float(out.uv_coordinates.max()) <= 1.0


def test_projection_and_multitask_heads() -> None:
    torch = pytest.importorskip("torch")
    from ruview.nn.heads import MultiTaskHeads, ProjectionHead, TaskKind

    torch.manual_seed(0)
    embedding = torch.randn(3, 32)
    projection = ProjectionHead(input_dim=32, projection_dim=12, hidden_dim=24, dropout=0.0)(embedding)

    assert tuple(projection.shape) == (3, 12)
    torch.testing.assert_close(projection.norm(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5)

    heads = MultiTaskHeads(
        input_dim=32,
        task_dims={TaskKind.PRESENCE: 1, TaskKind.VITALS: 2, TaskKind.POSE: 51},
    )
    outputs = heads(embedding, enabled_tasks=(TaskKind.PRESENCE, TaskKind.VITALS))

    assert set(outputs) == {TaskKind.PRESENCE, TaskKind.VITALS}
    assert tuple(outputs[TaskKind.PRESENCE].values.shape) == (3, 1)
    assert tuple(outputs[TaskKind.VITALS].values.shape) == (3, 2)
    assert torch.all(outputs[TaskKind.PRESENCE].uncertainty > 0.0)


def test_csi_to_pose_transformer_shapes_and_validation() -> None:
    torch = pytest.importorskip("torch")
    from ruview.nn.transformer import CsiToPoseConfig, CsiToPoseTransformer

    torch.manual_seed(0)
    config = CsiToPoseConfig(
        num_paths=4,
        num_subcarriers=8,
        d_model=32,
        num_heads=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        embedding_dim=24,
        feature_channels=16,
        feature_map_size=4,
        heatmap_size=10,
    )
    model = CsiToPoseTransformer(config)
    amplitude = torch.randn(2, 4, 8)
    phase = torch.randn(2, 4, 8)

    out = model(amplitude, phase)

    assert tuple(out.embedding.shape) == (2, 24)
    assert tuple(out.features.shape) == (2, 16, 4, 4)
    assert tuple(out.keypoint_heatmaps.shape) == (2, 17, 10, 10)
    assert tuple(out.part_logits.shape) == (2, 25, 10, 10)
    assert tuple(out.uv_coordinates.shape) == (2, 48, 10, 10)
    assert tuple(out.confidence.shape) == (2, 1, 10, 10)
    torch.testing.assert_close(out.embedding.norm(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)

    with pytest.raises(ValueError, match="antenna paths"):
        model(amplitude[:, :3, :], phase[:, :3, :])
