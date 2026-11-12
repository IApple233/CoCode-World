# Environment and models

Use Linux, Python 3.11, FFmpeg/FFprobe, and CUDA-compatible PyTorch. Install the dependencies and the bundled MiniMax-H3 Diffusers runtime using the commands in the [README](../README.md#getting-started).

The training launchers use eight GPUs per node. The released checkpoint was trained on four nodes, with 32 GPUs of approximately 80 GB memory each, using CP8 × DP4. One node uses CP8 × DP1 with a different effective batch size. The inference runner uses one group of eight GPUs. GPU encoding is also required when caching text and video inputs.

Download the official `MiniMaxAI/MiniMax-H3` weights first. For training, also download the Ref2VA DeCFG adapter from `DiffSynth-Studio/MiniMax-H3-TrainingAdapter` on ModelScope. The download script pins model revisions and checks the adapter's SHA256. Use `model_ref2va.safetensors`; the FL2VA and Comfy variants have different formats.

```bash
# Official base, then the Ref2VA training preset.
python scripts/download_models.py training

# CoCode-World step-4000 LoRA and EMA checkpoint.
python scripts/download_models.py cocode
```

The training loader fuses the DeCFG preset into the frozen base and trains a separate rank-64 task LoRA. Inference loads the original base and our EMA LoRA, without the DeCFG preset. The model repository also contains the non-EMA LoRA, optimizer/RNG state, and metadata.
