# CoCode-World

Training code, the **step-4000 LoRA checkpoint**, and eight evaluation examples for synchronized two-player video generation with **MiniMax-H3 Ref2VA** and **Proxy555** control.

[Model weights](https://huggingface.co/iapple1/CoCode-World-Model) · [Training data](https://huggingface.co/datasets/iapple1/CoCode-World-Data) · [Evaluation examples](examples/evaluation) · [Project page](https://iapple233.github.io/CCW/)

## Start from MiniMax-H3 and the released DeCFG adapter

**First download the official [MiniMax-H3 weights](https://huggingface.co/MiniMaxAI/MiniMax-H3). Then download DiffSynth-Studio's open-source [Ref2VA DeCFG LoRA](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter). Continue task training from this pretrained setup.**

This implementation fuses `model_ref2va.safetensors` into the frozen MiniMax-H3 `transformer_ref` at scale 1, then trains a fresh **rank-64 CoCode-World task LoRA**. The DeCFG preset stays fixed and is excluded from the saved task LoRA. **Inference uses the original MiniMax-H3 base and the CoCode-World EMA LoRA, without the DeCFG training preset.**

训练前先下载 **MiniMax-H3 官方权重**，再下载 **DiffSynth-Studio 开源的 Ref2VA DeCFG LoRA**，在这一预训练基础上继续进行任务训练。具体实现是先将 DeCFG 预设固定合入基座，再训练新的 rank-64 CoCode-World LoRA。推理时使用原始 MiniMax-H3 与我们的 EMA LoRA，不加载训练用 DeCFG 预设。

The DeCFG adapter is hosted on **ModelScope**. Its Ref2VA file is required; the FL2VA and Comfy variants have different formats. The download helper pins the releases and verifies the preset's SHA256.

## Install

Use Linux, **Python 3.11**, FFmpeg/FFprobe, and CUDA-compatible PyTorch. Recorded training used **32 GPUs with approximately 80 GB memory each**, arranged as CP8 × DP4 across four nodes. The inference runner uses one group of eight GPUs. Full-model training and inference require GPUs.

```bash
git clone https://github.com/IApple233/CoCode-World.git
cd CoCode-World
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e ./diffusers
```

Install `ffmpeg` using your system package manager if needed. The bundled Diffusers runtime includes the MiniMax-H3 and context-parallel code used by these entrypoints; use this local package.

## Download models

```bash
# Official MiniMax-H3, followed by DiffSynth's frozen Ref2VA DeCFG preset.
python scripts/download_models.py training
# Our step-4000 checkpoint for evaluation.
python scripts/download_models.py cocode
```

[iapple1/CoCode-World-Model](https://huggingface.co/iapple1/CoCode-World-Model) contains `ema.safetensors`, `lora.safetensors`, optimizer/RNG state, and checkpoint metadata. **EMA** is the variant used by the released examples. These are native checkpoint files for the included loader; download the base model separately.

## Prepare training data

```bash
hf download iapple1/CoCode-World-Data --repo-type dataset \
  --include 'train/*' 'manifests/train.jsonl' --local-dir datasets/CoCode-World-Data
python scripts/prepare_training_data.py \
  --dataset-dir datasets/CoCode-World-Data --output-dir datasets/prepared
python utility/cache_h3_processed720.py prepare \
  --manifest datasets/prepared/manifest.jsonl --output-dir cache/prepared \
  --model-root models/MiniMax-H3 --compact-proxy

# Run sequentially; each encoder stage uses the first visible GPU.
python utility/cache_h3_processed720.py text \
  --manifest cache/prepared/raw_manifest.jsonl --output-dir cache/train --model-root models/MiniMax-H3
python utility/cache_h3_processed720.py vae \
  --manifest cache/prepared/raw_manifest.jsonl --output-dir cache/train --model-root models/MiniMax-H3
python utility/cache_h3_processed720.py merge \
  --manifest cache/prepared/raw_manifest.jsonl --output-dir cache/train --model-root models/MiniMax-H3
```

Inputs are 2560 × 720 split-screen clips, 124 frames at 24 FPS, with `agent_000` on the left and `agent_001` on the right. The target's first frame supplies the appearance reference. Caching edge-pads targets to 2560 × 736 and uses the recorded half-resolution, temporally sampled Proxy profile.

The original checkpoint used **8,552 clips**. The public dataset contains **5,000 selected clips** from that collection; it supports this training recipe but does not reproduce the exact original sample stream or checkpoint. See [TRAINING.md](docs/TRAINING.md) for custom data, cache workers, and resume behavior.

## Train

For the recorded CP8 × DP4 topology, run this on **each of four nodes**, with shared paths and `NODE_RANK` set to `0`, `1`, `2`, or `3`:

```bash
export NNODES=4
export NODE_RANK=0                  # Change on each node.
export MASTER_ADDR=NODE_0_ADDRESS   # Replace with node 0's reachable address.
export MASTER_PORT=29500
bash scripts/train.sh cache/train/manifest.jsonl outputs/cocode-world
```

For one eight-GPU node, omit those distributed environment variables and run the same launcher. This uses CP8 × DP1, with a different effective batch. The launcher stops at step 4000 by default and uses rank/alpha `64/64`, learning rate `1e-5`, 200 warmup steps, EMA decay `0.999`, gradient checkpointing, and activation CPU offload. The full recorded configuration is in [configs/step-004000.json](configs/step-004000.json).

## Evaluate

Each [evaluation case](examples/evaluation) includes the **actual first-frame input, prompt, system prompt, all 17 window prompts, original Proxy555 video, seed, and recorded result preview**. All eight use the original step-4000 EMA LoRA and 30 denoising forwards per window.

```bash
python scripts/verify_release.py --decode-videos
# Omit --case-id to prepare all eight.
python scripts/prepare_evaluation.py --case-id case_01 --output-dir cache/eval_inputs
```

Follow [EVALUATION.md](docs/EVALUATION.md) to cache references and generate the full 60-second videos. Previews preserve all 1,440 frames but are compressed to 1600 × 450; input PNGs and Proxy videos retain their original bytes.

## Verification and provenance

Core training files were recovered from historical snapshots and checked against the original run's source hashes. See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and [source_files.json](provenance/source_files.json) for source identity and portable release changes.

```bash
python -m pip install pytest
python -m pytest -q
python scripts/verify_release.py --decode-videos
```

Release checks cover CPU training/checkpoint behavior, public-data extraction, prompt/input identity, and complete decoding of the examples. Full-model GPU training and inference were not rerun during packaging.

## License and acknowledgments

Original CoCode-World code uses the existing [MIT license](LICENSE). Bundled Diffusers and code adapted from Code World Model retain their [upstream notices](THIRD_PARTY_NOTICES.md). Model weights, datasets, and game-derived or edited example media retain their respective upstream terms; the code license does not relicense them. MiniMax-H3 uses its [Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE); the task LoRA retains its Hugging Face repository's Apache-2.0 license.

We thank [MiniMax](https://github.com/MiniMax-AI), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [Hugging Face Diffusers](https://github.com/huggingface/diffusers), and [Code World Model](https://github.com/buaacyw/code-world-model).
