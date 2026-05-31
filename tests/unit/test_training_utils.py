from __future__ import annotations

import json

import numpy as np

from ruview.training import checkpoints, export, losses, metrics, trainer


def _pose_frame() -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.zeros((17, 2), dtype=np.float64)
    keypoints[:, 0] = np.linspace(0.1, 0.9, 17)
    keypoints[:, 1] = np.linspace(0.2, 0.8, 17)
    visibility = np.ones(17, dtype=np.float64)
    return keypoints, visibility


def test_heatmaps_and_visibility_masked_mse() -> None:
    keypoints = np.array([[[0.5, 0.5], [0.1, 0.1]]], dtype=np.float64)
    visibility = np.array([[1.0, 0.0]], dtype=np.float64)
    target = losses.generate_target_heatmaps(keypoints, visibility, heatmap_size=5, sigma=1.0)

    assert target.shape == (1, 2, 5, 5)
    assert target[0, 0, 2, 2] == 1.0
    assert np.count_nonzero(target[0, 1]) == 0

    pred = target.copy()
    pred[0, 1] = 10.0
    assert losses.keypoint_heatmap_loss(pred, target, visibility) == 0.0
    assert losses.keypoint_heatmap_loss(pred, target) > 0.0

    pred[0, 0, 2, 2] = 0.0
    assert losses.keypoint_heatmap_loss(pred, target, visibility) > 0.0


def test_pck_and_oks_order_predictions() -> None:
    gt, visibility = _pose_frame()
    perfect = gt.copy()
    medium = gt.copy()
    far = gt.copy()
    medium[12:] += 0.3
    far += 0.5

    _, _, perfect_pck = metrics.compute_pck(perfect, gt, visibility, threshold=0.2)
    _, _, medium_pck = metrics.compute_pck(medium, gt, visibility, threshold=0.2)
    _, _, far_pck = metrics.compute_pck(far, gt, visibility, threshold=0.2)

    assert perfect_pck > medium_pck > far_pck
    assert metrics.compute_oks(perfect, gt, visibility) > metrics.compute_oks(medium, gt, visibility)
    assert metrics.compute_oks(medium, gt, visibility) > metrics.compute_oks(far, gt, visibility)

    aggregate = metrics.aggregate_metrics(np.stack([perfect, medium]), np.stack([gt, gt]), np.stack([visibility, visibility]))
    assert aggregate.frames_evaluated == 2
    assert aggregate.keypoints_evaluated == 34
    assert len(aggregate.per_joint_pck) == 17


def test_checkpoint_manifest_roundtrip_and_top_k(tmp_path) -> None:
    paths = []
    for epoch, pck in [(1, 0.41), (2, 0.52), (3, 0.49)]:
        path = tmp_path / f"epoch-{epoch}.json"
        paths.append(path)
        checkpoints.save_checkpoint_manifest(
            path,
            epoch=epoch,
            metrics={"val_pck": pck, "val_oks": pck - 0.1},
            config={"batch_size": 2},
            weights_path=f"weights-{epoch}.pt",
        )

    loaded = checkpoints.load_checkpoint_manifest(paths[1])
    assert loaded.epoch == 2
    assert loaded.metric("val_pck") == 0.52
    assert loaded.resolved_weights_path() == tmp_path / "weights-2.pt"

    top = checkpoints.select_best_checkpoints(paths, metric="val_pck", top_k=2)
    assert [item.epoch for item in top] == [2, 3]
    assert checkpoints.best_checkpoint(paths, metric="val_pck").epoch == 2


def test_model_export_manifest_roundtrip(tmp_path) -> None:
    weights = tmp_path / "model.weights"
    weights.write_bytes(b"tiny weights")
    manifest_path = tmp_path / "model.json"

    saved = export.save_model_export_manifest(
        manifest_path,
        model_name="tiny-pose",
        backend="numpy",
        inputs=[export.TensorSpec("csi", (1, 2, 8), "float32", "BLC")],
        outputs=[{"name": "keypoints", "shape": [1, 17, 2], "dtype": "float32", "layout": "BJC"}],
        config={"heatmap_size": 8},
        weights_path=weights,
        provenance={"commit": "abc123"},
    )

    assert len(saved.provenance_hash) == 64
    assert saved.weights_hash == export.hash_file(weights)
    assert export.shape_metadata(saved)["outputs"][0]["name"] == "keypoints"

    loaded = export.load_model_export_manifest(manifest_path)
    assert loaded.model_name == "tiny-pose"
    assert loaded.inputs[0].shape == (1, 2, 8)
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["rvf_compatible"] is True


def test_tiny_trainer_history_is_deterministic() -> None:
    gt, visibility = _pose_frame()
    examples = [
        trainer.TrainingExample(keypoints=gt, visibility=visibility, pred_keypoints=gt.copy()),
        trainer.TrainingExample(keypoints=gt, visibility=visibility, pred_keypoints=gt + 0.5),
    ]
    config = trainer.TrainingConfig(epochs=2, batch_size=1, shuffle=False)
    first = trainer.Trainer(config).fit(examples)
    second = trainer.Trainer(config).fit(examples)

    assert first.to_dict() == second.to_dict()
    assert len(first.history) == 2
    assert first.history[0].train_samples == 2
    assert first.final_metrics.num_samples == 2
    assert first.best_epoch == 1
