#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <device_id> <alg> <output_prefix> [extra args...]" >&2
  exit 1
fi

DEVICE_ID="$1"
ALG="$2"
OUTPUT_PREFIX="$3"
shift 3

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
PYTHON_BIN="/mnt/liuyx_data/miniconda/envs/a3/bin/python"
EXP_DIR="$ROOT/experiments"

COMMON_ARGS=(
  --alg "$ALG"
  --device npu
  --partition iid
  --num-workers 10
  --delay-gap 0.5
  --delay-jitter 0.1
  --train-part-size 12000
  --batch-size 600
  --test-part-size 1000
  --test-batch-size 1000
  --epochs 60
  --lr 0.01
  --log-interval 240
  --precision 0
)

cd "$ROOT"

for SEED in 4 5 6 7 8 9 10; do
  OUT_FILE="$EXP_DIR/${OUTPUT_PREFIX}${SEED}.pkl"
  if [ -f "$OUT_FILE" ]; then
    echo "[skip] $(date -Iseconds) ${OUT_FILE}"
    continue
  fi

  echo "[start] $(date -Iseconds) alg=${ALG} seed=${SEED} device=${DEVICE_ID}"
  ASCEND_RT_VISIBLE_DEVICES="$DEVICE_ID" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  "$PYTHON_BIN" async_distributed_main.py \
    "${COMMON_ARGS[@]}" \
    --seed "$SEED" \
    --dump-data "$OUT_FILE" \
    "$@"
  echo "[done] $(date -Iseconds) alg=${ALG} seed=${SEED} device=${DEVICE_ID}"
done
