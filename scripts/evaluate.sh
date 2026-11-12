#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 3 ]]; then
  echo "Usage: scripts/evaluate.sh CACHE_MANIFEST PLAN_JSON OUTPUT_DIR [case_XX]" >&2
  exit 2
fi
manifest=$1
plan=$2
output=$3
extra=()
if [[ $# -ge 4 ]]; then extra=(--case-id "$4"); fi
common=(--manifest "$manifest" --plan "$plan" --output-dir "$output"
  --model-root "${MODEL_ROOT:-models/MiniMax-H3}"
  --checkpoint "${CHECKPOINT:-checkpoints/step-004000}" --num-forwards 30 --nodes 1)
torchrun --standalone --nproc_per_node=8 scripts/evaluate.py \
  --stage w0 "${common[@]}" "${extra[@]}"
for index in $(seq 1 16); do
  python scripts/evaluate.py --stage prefix --window-index "$index" "${common[@]}" "${extra[@]}"
  torchrun --standalone --nproc_per_node=8 scripts/evaluate.py \
    --stage continuation --window-index "$index" "${common[@]}" "${extra[@]}"
done
python scripts/evaluate.py --stage render "${common[@]}" "${extra[@]}"
