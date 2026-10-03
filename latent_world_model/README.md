# Latent World Model

This folder contains an isolated first implementation of the subtask-terminal
latent predictor discussed for Latent VLA JEV. It predicts the visual features
of the image at the end of the current VLM-generated subtask.

## Contract

Given current-image tokens `Z_t` and instruction tokens `C_j`, the model returns
`Z_hat_end` with exactly the same shape and feature space as the frozen vision
encoder output for the successful terminal image:

```text
Z_t       = E_frozen(I_t)             [B, N, Dv]
Z_end     = stopgrad(E_frozen(I_end)) [B, N, Dv]
Z_hat_end = F_theta(Z_t, C_j)         [B, N, Dv]
```

The model does not predict pixels, intermediate dynamics, actions, PT, or FT.
The same frozen vision encoder and image preprocessing must be used for current
and terminal images and at the JEV interface. `Z_hat_end` is computed at the
subtask boundary and can be cached for the duration of that subtask.

## Architecture

The default predictor is a Perceiver-style conditional Transformer:

1. Project visual patch tokens and instruction tokens into a shared hidden size.
2. A fixed-size latent array cross-attends to the full visual and text context.
3. Latent self-attention layers form a task-conditioned memory.
4. One output query per visual patch reads that memory and predicts the matching
   terminal-image feature. The output is residualized around current features.

This is a predictor architecture, not a separate goal-token target: the output
tokens are the predicted features of the terminal image and retain the encoder's
spatial token layout. AdaLN/DiT is intentionally left as a later multimodal
extension; it is not required for the first deterministic predictor.

## Feature-cache format

Training consumes `.pt` feature-cache files so the frozen foundation encoders
can be selected and run by the project without hard-coding one vendor's model
loader. Create `train.pt` and `val.pt` with these tensors:

| Key | Shape | Meaning |
|---|---|---|
| `current_features` | `[B, N, Dv]` | Frozen encoder features for subtask-start images |
| `terminal_features` | `[B, N, Dv]` | Same encoder features for successful terminal images |
| `task_features` | `[B, L, Dt]` | Frozen text encoder token features for the subtask instruction |
| `task_mask` | `[B, L]` bool | `True` for valid instruction tokens, `False` for padding |
| `success` | `[B]` bool | Whether this sample is a verified successful subtask completion |
| `episode_ids` | list of strings | Source episode/trajectory IDs, used to detect split leakage |

Only verified successful examples contribute to terminal-feature supervision.
Do not label every recording's last frame as success if it may be a timeout,
failure, or truncated episode. Split by episode or scene before creating the
cache; the trainer rejects overlapping episode IDs across train and validation.

Both cache files should use the same encoder checkpoint, preprocessing,
feature layers, and text-token convention. For multi-camera input, cache the
tokens in a consistent order and keep the camera/layout metadata in your
upstream dataset. The built-in synthetic-cache tool is only a software smoke
test and is not a research dataset.

## Install and run

Python 3.10+ and PyTorch 2.2+ are required. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python scripts/make_toy_cache.py --output runs/toy_cache
python -m lwm.cli train \
  --train runs/toy_cache/train.pt \
  --val runs/toy_cache/val.pt \
  --output runs/toy_model \
  --epochs 3 --batch-size 8 --hidden-dim 64 --heads 4 --latent-count 8
python -m pytest -q
```

The toy cache only checks tensor contracts and training execution. For a real
experiment, replace it with cached tokens from a frozen V-JEPA 2.1, DINO, or
other selected visual encoder and a frozen text encoder. The `FrozenFeatureEncoder`
adapter in `src/lwm/backbone.py` wraps an already-loaded PyTorch image encoder;
loading a particular V-JEPA checkpoint is deliberately kept in the upstream
project-specific integration.

## Use from JEV

```python
pred_terminal = lwm(current_visual_tokens, task_text_tokens, task_mask)
# pred_terminal: [batch, visual_tokens, encoder_dim]
```

JEV should cross-attend to these predicted terminal-image features alongside
its current visual features, robot state, and PT. LWM is frozen during JEV
training in this first version. Compare oracle terminal features and predicted
terminal features to quantify the prediction-to-control gap.
