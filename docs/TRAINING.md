# Training details

## Pretrained initialization

Download official `MiniMaxAI/MiniMax-H3` first. The helper pins revision `0919fb684102ef115126612f3fbb84d1027e7317`; the transformer, VAE, and text encoder configuration/index fingerprints match the checkpoint.

Then download `model_ref2va.safetensors` from **DiffSynth-Studio/MiniMax-H3-TrainingAdapter on ModelScope**, revision `bdddb151d0dfb62b1dfa9e2f8ae6a6e1b179e098`. Its SHA256 must be `9361683349c1bf78f206c382d91b1e01b91fcddc2408e85ab39c9faa51f0d3d1`.

Fuse the pretrained DeCFG LoRA into the frozen Ref2VA transformer at scale 1, then train a new rank-64 task LoRA. The loader converts the official QKV head/role order and FFN gate/value layout. The saved task adapter excludes the preset. Inference uses the original base plus task EMA, without the preset. This follows the [DiffSynth Ref2VA training example](https://github.com/modelscope/DiffSynth-Studio/blob/cf6d843e27eba3f761ab0df84e7883c0413f062f/examples/minimax_h3/model_training/lora/MiniMax-H3-Ref2VA.sh).

## Input data

`scripts/prepare_training_data.py` reads the public `manifests/train.jsonl`, extracts target/proxy/caption files from the TAR shards, verifies their checksums, and creates the native paired manifest. Use `--limit 2` for a small preprocessing check.

For custom data, use that script's JSONL output as the schema reference. Supply unique sample IDs, target/proxy paths and SHA256 hashes, captions and caption-file hashes, 124 frames at 24 FPS, 2560 × 720 geometry, and horizontal split-screen metadata (`agent_000` left, `agent_001` right). Set `reference_image_source` to `target_split_screen_frame_000000`. Preparation supplies the unchanged task/system prompts. Regenerate local manifests after moving data to new paths. Preserve synchronized camera timing and keep evaluation source sequences outside the training split.

`--compact-proxy` in the cache preparation stage is required: source indices are `0,4,...,120,123`, retaining the source timeline. Proxy pixels are resized to 1280 × 360 and padded to 1280 × 384. Target/image reference use edge-padded 2560 × 736 canvases. Qwen hidden layer 50 supplies text features.

The `text` and `vae` stages can each use independent workers via `--rank R --world-size N`, shared output paths, and per-worker `CUDA_VISIBLE_DEVICES`. Finish all text workers before starting VAE workers, then run `merge` once. Keep all paths, manifests, and `--context-parallel-degree 8` identical across workers. Merge verifies the cache and writes `manifest.validation.json`. The launcher requires this validated cache.

## Recorded settings

| Setting | Value |
|---|---|
| Trainable LoRA | 345,899,008 parameters, 624 tensors, rank/alpha 64/64 |
| Targets | Attention Q/K/V/O and feed-forward linears |
| Optimizer | AdamW, LR 1e-5, betas (0.9, 0.99), weight decay 0 |
| LR schedule | 200-step linear warmup, then constant |
| Global batch | 4; CP8 × DP4, accumulation 1 |
| Objective | Target-row flow-velocity MSE |
| Timestep | Uniform noise ratio, shift 12 |
| Reference noise augmentation | 0.999 |
| EMA / clipping | 0.999 / max gradient norm 1 |
| Sampling | Seed 42, epoch shuffle buffer 32768 |
| Checkpoint interval | 500 steps |

This release is **step 4000**, with 16,000 global samples seen. The original three-epoch job ran to 6414 steps on 8552 clips; later checkpoints are not included. The public 5000-clip subset produces a different sample stream.

Use `STEPS`, `MODEL_ROOT`, and `DECFG_ADAPTER` environment variables to change launcher duration/paths. Additional trainer flags follow its two positional arguments. Resume a checkpoint from your own run with matching data and configuration:

```bash
STEPS=5000 bash scripts/train.sh cache/train/manifest.jsonl outputs/cocode-world \
  --resume outputs/cocode-world/step-004000
```

Resume validates the manifest, model components, preset hash/scale, optimizer settings, topology, and sample order. The released original optimizer/RNG files support archival and matching-run use; they are not a compatible optimizer continuation on the public subset. Start a new training run when data or configuration changes.

The launcher has no SSH orchestration or private host configuration. Run it on each allocated node with your scheduler. Evaluation is a separate explicit command.
