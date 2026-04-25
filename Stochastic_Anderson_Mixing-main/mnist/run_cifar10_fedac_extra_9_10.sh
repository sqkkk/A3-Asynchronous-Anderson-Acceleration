#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
PYTHON_BIN="/mnt/liuyx_data/miniconda/envs/a3/bin/python"
EXP_DIR="$ROOT/experiments"
LOG_DIR="$EXP_DIR/cifar10_fedac_10seed_logs"
mkdir -p "$LOG_DIR"

cd "$ROOT"

COMMON_ARGS=(
  --alg fedac
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
  --fedac-buffer-size 5
  --fedac-eta-g 0.0003
  --fedac-beta1 0.6
  --fedac-beta2 0.9
  --cv-server-lr 0.5
  --cv-momentum 1.0
  --grad-clip-norm 0.0
)

run_seed() {
  local device_id="$1"
  local seed="$2"
  local out_file="$EXP_DIR/cifar10_multiseed_fedac_tuned_seed${seed}.pkl"
  local log_file="$LOG_DIR/extra_device${device_id}_seed${seed}.log"

  if [ -s "$out_file" ]; then
    echo "[skip] $(date -Iseconds) existing non-empty result: $out_file"
    return 0
  fi

  # Reserve the canonical output path so the existing device7 queue skips
  # seed9/seed10 after seed8 finishes. The direct run overwrites this file.
  : > "$out_file"

  RUN_ARGS=(
    async_distributed_main.py
    "${COMMON_ARGS[@]}"
    --seed "$seed"
    --dump-data "$out_file"
  )

  {
    echo "timestamp=$(date -Iseconds)"
    echo "workdir=$ROOT"
    echo "output=$out_file"
    printf 'command='
    printf '%q ' \
      "ASCEND_RT_VISIBLE_DEVICES=$device_id" \
      "OMP_NUM_THREADS=1" \
      "MKL_NUM_THREADS=1" \
      "$PYTHON_BIN" \
      "${RUN_ARGS[@]}"
    echo
  } > "${out_file}.cmd.txt"

  echo "[start] $(date -Iseconds) alg=fedac seed=${seed} device=${device_id}" > "$log_file"
  ASCEND_RT_VISIBLE_DEVICES="$device_id" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  "$PYTHON_BIN" "${RUN_ARGS[@]}" >> "$log_file" 2>&1
  echo "[done] $(date -Iseconds) alg=fedac seed=${seed} device=${device_id}" >> "$log_file"
}

run_seed 0 9 &
run_seed 1 10 &

wait
echo "done"
