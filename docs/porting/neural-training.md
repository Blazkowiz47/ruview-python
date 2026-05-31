# Neural And Training Research Notes

Milestone 10 ports the neural and training surface as research scaffolding, not
as production training parity with the Rust crates. The Python work should make
tensor shapes, dataset windows, embeddings, and expected plots easy to inspect.
Full training remains optional behind the `nn` extra because PyTorch is not part
of the default dependency set.

## Source References

- `ruview-python/plan.md`
- `RuView/v2/crates/wifi-densepose-nn/README.md`
- `RuView/v2/crates/wifi-densepose-nn/src/tensor.rs`
- `RuView/v2/crates/wifi-densepose-nn/src/densepose.rs`
- `RuView/v2/crates/wifi-densepose-nn/src/translator.rs`
- `RuView/v2/crates/wifi-densepose-nn/src/inference.rs`
- `RuView/v2/crates/wifi-densepose-nn/src/rf_encoder.rs`
- `RuView/v2/crates/wifi-densepose-train/README.md`
- `RuView/v2/crates/wifi-densepose-train/src/config.rs`
- `RuView/v2/crates/wifi-densepose-train/src/dataset.rs`
- `RuView/v2/crates/wifi-densepose-train/src/subcarrier.rs`
- `RuView/v2/crates/wifi-densepose-train/src/model.rs`
- `RuView/v2/crates/wifi-densepose-train/src/trainer.rs`
- `RuView/v2/crates/wifi-densepose-train/src/losses.rs`
- `RuView/v2/crates/wifi-densepose-train/src/metrics.rs`
- `RuView/v2/crates/wifi-densepose-train/src/signal_features.rs`
- `RuView/v2/crates/wifi-densepose-sensing-server/src/dataset.rs`
- `RuView/v2/crates/wifi-densepose-sensing-server/src/adaptive_classifier.rs`
- `RuView/v2/crates/wifi-densepose-sensing-server/src/model_manager.rs`

## Deliverable Map

| Rust reference | Python deliverable | Behavior to preserve |
|---|---|---|
| `nn/src/tensor.rs`, `train/src/config.rs` | Tensor shape notes and future `ruview.nn` tensor helpers | Keep explicit batch, channel, frame, antenna, subcarrier, heatmap, and DensePose channel conventions visible. Defaults are 100 frames, 3x3 antennas, 56 model subcarriers, 17 keypoints, 24 body parts, and 256 backbone channels. |
| `nn/src/translator.rs`, `train/src/model.rs` | CSI-to-pose transformer research API and `notebooks/08_csi_to_pose_experiment.ipynb` | Demonstrate CSI feature maps becoming visual/keypoint heatmaps. The notebook uses deterministic NumPy fallbacks until an optional PyTorch implementation exists. |
| `nn/src/densepose.rs`, `train/src/model.rs` | DensePose-style head notes for segmentation, UV, and keypoint confidence plots | Preserve the output contract: body-part logits include background, UV uses 24x2 channels in the training model, and notebook confidence plots are illustrative rather than trained predictions. |
| `nn/src/rf_encoder.rs` | RF encoder, contrastive embedding helpers, and projection-head notes | Preserve 256-d RF embedding intent, task-head uncertainty, calibration-robustness loss, triplet loss, and deterministic cross-room positive sampling as research targets. |
| `train/src/dataset.rs`, `train/src/subcarrier.rs`, `train/src/signal_features.rs` | Dataset loader notes and `notebooks/09_dataset_replay_lab.ipynb` | Preserve MM-Fi layout expectations, deterministic synthetic samples, deterministic shuffling, batched replay, and 114-to-56 or native-to-model subcarrier interpolation semantics. |
| `sensing-server/src/dataset.rs` | Future replay/import bridge notes | Preserve normalized, windowed data-pipeline vocabulary and train/validation split semantics while avoiding hardware or file-dataset requirements in notebooks. |
| `train/src/losses.rs`, `train/src/metrics.rs`, `train/src/trainer.rs` | Training-loop research notes and synthetic loss-curve notebooks | Preserve the vocabulary of keypoint, DensePose, transfer, PCK, OKS, validation, and checkpoint metrics without claiming the Python notebooks train a real model. |
| `train/src/model.rs`, `train/src/trainer.rs`, `sensing-server/src/adaptive_classifier.rs`, `sensing-server/src/model_manager.rs` | Optional PyTorch-backed training, checkpoints, JSON metadata, and RVF export experiments | Keep PyTorch optional through `uv sync --extra nn`. Checkpoints can be `.pt` or JSON/RVF-compatible summaries only after a real Python model surface lands. |
| Milestone 10 notebook list | `notebooks/08_csi_to_pose_experiment.ipynb`, `09_dataset_replay_lab.ipynb`, `10_model_embedding_visualization.ipynb` | Keep notebooks valid, output-free, deterministic, and runnable without torch by using NumPy and Matplotlib fallback helpers. |

## Intentional Deviations

- This milestone documents a research workflow. It does not claim production
  parity with ONNX Runtime, `tch-rs`, Candle, TensorRT, CUDA, or the Rust
  trainer.
- PyTorch is optional in Python via the `nn` extra. The notebooks must run on
  the default plus research stack and should not import torch unless guarded.
- The notebooks use deterministic synthetic data instead of MM-Fi files,
  pretrained weights, captured CSI, or hardware recordings.
- DensePose masks, keypoint confidence, PCA clusters, and loss curves in the
  notebooks are diagnostics and placeholders. They are not accuracy evidence.
- Future export should prefer a small JSON/RVF-compatible manifest for research
  metadata before adding heavyweight model export paths.
- Rust source files under `src/ruview/nn` and `src/ruview/training` are not
  part of this docs/notebook pass.

## Notebook Intent

- `08_csi_to_pose_experiment.ipynb` visualizes a synthetic CSI feature map, a
  fallback CSI-to-keypoint heatmap, and per-keypoint confidence.
- `09_dataset_replay_lab.ipynb` visualizes deterministic replay windows,
  deterministic batch assignment, and synthetic train/validation loss curves.
- `10_model_embedding_visualization.ipynb` visualizes empty, present, motion,
  and room embedding clusters with PCA plus synthetic contrastive/loss curves.

Each notebook includes guarded `ruview.nn` or `ruview.training` imports. If those
packages are absent or still empty, the notebook uses local NumPy helpers.

## Run Path

```bash
uv sync --extra research
uv run jupyter lab notebooks
```

For optional PyTorch experiments after the Python neural modules exist:

```bash
uv sync --extra research --extra nn
```

## Verification

Initial verification for this milestone:

- `uv run python -m json.tool notebooks/08_csi_to_pose_experiment.ipynb`
- `uv run python -m json.tool notebooks/09_dataset_replay_lab.ipynb`
- `uv run python -m json.tool notebooks/10_model_embedding_visualization.ipynb`
- `uv run pytest -q tests/unit/test_notebook_json.py`

Future verification should add smoke execution of the notebook code cells with
`MPLBACKEND=Agg`, plus focused tests for any real `ruview.nn` or
`ruview.training` APIs once those modules move beyond research placeholders.
