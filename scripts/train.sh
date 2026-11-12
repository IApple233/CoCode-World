#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 2 ]]; then
  echo "Usage: scripts/train.sh CACHE_MANIFEST OUTPUT_DIR [extra trainer flags]" >&2
  exit 2
fi
manifest=$1
output=$2
shift 2
model_root=${MODEL_ROOT:-models/MiniMax-H3}
decfg=${DECFG_ADAPTER:-models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter/model_ref2va.safetensors}
nnodes=${NNODES:-1}
node_rank=${NODE_RANK:-0}
if (( nnodes > 1 )); then
  : "${MASTER_ADDR:?Set MASTER_ADDR to the address of node 0 on every node}"
fi
exec torchrun --nnodes="$nnodes" --nproc_per_node=8 --node_rank="$node_rank" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-29500}" \
  scripts/train.py \
  --manifest "$manifest" --output-dir "$output" --model-root "$model_root" \
  --decfg-adapter "$decfg" --decfg-adapter-scale 1 \
  --context-parallel-degree 8 --steps "${STEPS:-4000}" \
  --lora-rank 64 --lora-alpha 64 --lora-targets qkvo-ffn \
  --learning-rate 1e-5 --warmup-steps 200 --shuffle-buffer-size 32768 \
  --adam-beta1 0.9 --adam-beta2 0.99 --weight-decay 0 \
  --ema-decay 0.999 --max-gradient-norm 1 --gradient-accumulation 1 \
  --reference-noise-aug 0.999 --video-shift 12 --seed 42 \
  --save-every 500 --activation-cpu-offload --trust-validated-cache "$@"
