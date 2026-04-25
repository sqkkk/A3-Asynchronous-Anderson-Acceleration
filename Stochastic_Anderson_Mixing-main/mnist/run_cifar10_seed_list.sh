#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <device_id> <alg> <output_prefix> <seed_csv> [extra args...]" >&2
  exit 1
fi

DEVICE_ID="$1"
ALG="$2"
OUTPUT_PREFIX="$3"
SEED_CSV="$4"
shift 4

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
PYTHON_BIN="/mnt/liuyx_data/miniconda/envs/a3/bin/python"
EXP_DIR="$ROOT/experiments"

COMMON_ARGS=(
  --alg "$ALG"
  --dataset cifar10
  --model resnet18
  --device npu
  --partition iid
  --num-workers 2
  --delay-gap 0.1
  --delay-jitter 0.02
  --train-part-size 50000
  --batch-size 128
  --test-part-size 10000
  --test-batch-size 1000
  --epochs 100
  --lr 0.05
  --lr-schedule cosine
  --lr-min-ratio 0.01
  --weight-decay 5e-4
  --log-interval 3920
  --precision 0
)

cd "$ROOT"

IFS=',' read -r -a SEEDS <<< "$SEED_CSV"

for SEED in "${SEEDS[@]}"; do
  OUT_FILE="$EXP_DIR/${OUTPUT_PREFIX}${SEED}.pkl"
  mkdir -p "$(dirname "$OUT_FILE")"
  if [ -f "$OUT_FILE" ] && [ -s "$OUT_FILE" ]; then
    echo "[skip] $(date -Iseconds) ${OUT_FILE}"
    continue
  fi
  if [ -f "$OUT_FILE" ] && [ ! -s "$OUT_FILE" ]; then
    echo "[resume] $(date -Iseconds) ${OUT_FILE} (zero-byte placeholder)"
  fi

  echo "[start] $(date -Iseconds) alg=${ALG} seed=${SEED} device=${DEVICE_ID}"
  RUN_ARGS=(
    async_distributed_main.py
    "${COMMON_ARGS[@]}"
    --seed "$SEED"
    --dump-data "$OUT_FILE"
    "$@"
  )
  CMD_FILE="${OUT_FILE}.cmd.txt"
  {
    echo "timestamp=$(date -Iseconds)"
    echo "workdir=$ROOT"
    echo "output=$OUT_FILE"
    printf 'command='
    printf '%q ' \
      "ASCEND_RT_VISIBLE_DEVICES=$DEVICE_ID" \
      "OMP_NUM_THREADS=1" \
      "MKL_NUM_THREADS=1" \
      "$PYTHON_BIN" \
      "-u" \
      "${RUN_ARGS[@]}"
    echo
  } > "$CMD_FILE"

  ASCEND_RT_VISIBLE_DEVICES="$DEVICE_ID" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  "$PYTHON_BIN" -u "${RUN_ARGS[@]}"
  echo "[done] $(date -Iseconds) alg=${ALG} seed=${SEED} device=${DEVICE_ID}"
done
