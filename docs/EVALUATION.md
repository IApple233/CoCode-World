# Evaluation

The [eight cases](../examples/evaluation) are selected original step-4000 **EMA** examples covering riding, interaction, and exploration. They are qualitative examples rather than a random quantitative benchmark. The first-frame PNG is the actual edited appearance reference, not a hard pixel clamp on generated frame 0.

`prompt.txt` and `system_prompt.txt` describe window 0. **Use `prompts.json` for all 17 windows**, including continuation prompts, scene descriptions, intervals, and seed.

## Prepare and cache

```bash
python scripts/download_models.py base
python scripts/download_models.py cocode
python scripts/verify_release.py --decode-videos
python scripts/prepare_evaluation.py --case-id case_01 --output-dir cache/eval_inputs

python utility/cache_h3_processed720.py prepare \
  --manifest cache/eval_inputs/manifest.jsonl --output-dir cache/eval_prepared \
  --model-root models/MiniMax-H3 --compact-proxy --preserve-system-prompt
python utility/cache_h3_processed720.py text \
  --manifest cache/eval_prepared/raw_manifest.jsonl --output-dir cache/eval \
  --model-root models/MiniMax-H3 --preserve-system-prompt
python utility/cache_h3_processed720.py vae \
  --manifest cache/eval_prepared/raw_manifest.jsonl --output-dir cache/eval \
  --model-root models/MiniMax-H3 --preserve-system-prompt
python utility/cache_h3_processed720.py merge \
  --manifest cache/eval_prepared/raw_manifest.jsonl --output-dir cache/eval \
  --model-root models/MiniMax-H3 --preserve-system-prompt
```

Omit `--case-id` to prepare all eight; repeat it to choose several. Preparation losslessly encodes each Proxy window. The final window has 80 source frames and repeats its final frame 44 times. A hash-verified completed preparation can be reused.

The cache schema includes a clean target slot. Inference preparation fills this unused slot with a Proxy encoding; generation initializes target latents from seeded noise and never reads those clean target values. No ground-truth RGB video is required. These external-reference records must not be used for training.

## Generate

Use one node with eight suitable GPUs:

```bash
bash scripts/evaluate.sh cache/eval/manifest.jsonl cache/eval_inputs/plan.json outputs/evaluation
```

An optional fourth argument restricts generation to one prepared case, such as `case_01`. `MODEL_ROOT` and `CHECKPOINT` override default locations. No DeCFG preset is loaded. The runner validates the released checkpoint fingerprints, base components, Proxy profile, and complete case windows.

Each 124-frame window uses 30 denoising forwards, shift 12, and reference noise augmentation 0.999. Continuations overlap by 39 frames and advance by 85 frames. They retain 12 temporal latent frames, project that prefix at each denoising step, use case-level noise, and encode one boundary RGB frame as the next image reference. Seventeen windows are assembled in latent space, decoded, and cropped to 1440 frames at 24 FPS.

Results appear in `outputs/evaluation/case_XX/generated_long.mp4`, alongside reports and latent artifacts. Completed stages can be resumed; changes to inputs or settings require a fresh output directory. Previews in this repository are compressed historical outputs, not fresh GPU inference or lossless metric references. Input PNG/Proxy hashes and prompt text are fixed; hardware, kernels, software, and output encoding can affect exact output equality.
