#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
EXP_DIR="$ROOT/experiments"
LOG_DIR="$EXP_DIR/validation_settings_10seed_logs"
PYTHON_BIN="/mnt/liuyx_data/miniconda/envs/a3/bin/python"

mkdir -p "$LOG_DIR"
cd "$ROOT"

COMMON_ARGS=(
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

touch_placeholders() {
  local prefix="$1"
  for seed in {1..10}; do
    local out_file="$EXP_DIR/${prefix}${seed}.pkl"
    if [ ! -f "$out_file" ]; then
      : > "$out_file"
    fi
  done
}

run_group() {
  local device_id="$1"
  local alg="$2"
  local out_prefix="$3"
  local tmp_prefix="$4"
  local seed_csv="$5"
  shift 5

  IFS=',' read -r -a seeds <<< "$seed_csv"
  for seed in "${seeds[@]}"; do
    local out_file="$EXP_DIR/${out_prefix}${seed}.pkl"
    local tmp_file="$EXP_DIR/${tmp_prefix}${seed}.pkl"
    if [ -s "$out_file" ]; then
      echo "[skip] $(date -Iseconds) ${out_file}"
      continue
    fi

    echo "[start] $(date -Iseconds) alg=${alg} seed=${seed} device=${device_id}"
    ASCEND_RT_VISIBLE_DEVICES="$device_id" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    "$PYTHON_BIN" async_distributed_main.py \
      "${COMMON_ARGS[@]}" \
      --alg "$alg" \
      --seed "$seed" \
      --dump-data "$tmp_file" \
      "$@"
    mv -f "$tmp_file" "$out_file"
    echo "[done] $(date -Iseconds) alg=${alg} seed=${seed} device=${device_id}"
  done
}

touch_placeholders "validate10_workers20_fedasync_seed"
touch_placeholders "validate10_workers20_fedac_tuned_seed"

(
  run_group 0 fedasync validate10_workers20_fedasync_seed validate10_workers20_fedasync_parallel_tmp_seed 1,3,5,7,9 \
    --num-workers 20 \
    --fedasync-decay 1.0
) > "$LOG_DIR/workers20_fedasync_odd.log" 2>&1 &

(
  run_group 1 fedasync validate10_workers20_fedasync_seed validate10_workers20_fedasync_parallel_tmp_seed 2,4,6,8,10 \
    --num-workers 20 \
    --fedasync-decay 1.0
) > "$LOG_DIR/workers20_fedasync_even.log" 2>&1 &

(
  run_group 6 fedac validate10_workers20_fedac_tuned_seed validate10_workers20_fedac_parallel_tmp_seed 1,3,5,7,9 \
    --num-workers 20 \
    --fedac-buffer-size 5 \
    --fedac-eta-g 0.0003 \
    --fedac-beta1 0.6 \
    --fedac-beta2 0.9 \
    --cv-server-lr 0.5 \
    --cv-momentum 1.0
) > "$LOG_DIR/workers20_fedac_odd.log" 2>&1 &

(
  run_group 7 fedac validate10_workers20_fedac_tuned_seed validate10_workers20_fedac_parallel_tmp_seed 2,4,6,8,10 \
    --num-workers 20 \
    --fedac-buffer-size 5 \
    --fedac-eta-g 0.0003 \
    --fedac-beta1 0.6 \
    --fedac-beta2 0.9 \
    --cv-server-lr 0.5 \
    --cv-momentum 1.0
) > "$LOG_DIR/workers20_fedac_even.log" 2>&1 &

wait
