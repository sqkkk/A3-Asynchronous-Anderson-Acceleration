#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/mnist_fedac_10seed_logs"
EXP_DIR="$ROOT/experiments"

mkdir -p "$LOG_DIR"

# Reuse the tuned seed-1 result under the 10-seed naming scheme.
if [ -f "$EXP_DIR/fedac_mnist_tune_eta3e4_buf5_cv05.pkl" ] && [ ! -f "$EXP_DIR/afl_same_setting_fedac_tuned_seed1.pkl" ]; then
  cp "$EXP_DIR/fedac_mnist_tune_eta3e4_buf5_cv05.pkl" "$EXP_DIR/afl_same_setting_fedac_tuned_seed1.pkl"
fi

launch() {
  local device_id="$1"
  local job_name="$2"
  local seed_csv="$3"
  shift 3
  nohup "$ROOT/run_mnist_seed_list.sh" "$device_id" fedac afl_same_setting_fedac_tuned_seed "$seed_csv" "$@" \
    >"$LOG_DIR/${job_name}.log" 2>&1 &
  echo "${job_name} pid=$! seeds=${seed_csv}"
}

launch 3 fedac_a "2,7" \
  --fedac-buffer-size 5 \
  --fedac-eta-g 0.0003 \
  --fedac-beta1 0.6 \
  --fedac-beta2 0.9 \
  --cv-server-lr 0.5 \
  --cv-momentum 1.0

launch 4 fedac_b "3,8" \
  --fedac-buffer-size 5 \
  --fedac-eta-g 0.0003 \
  --fedac-beta1 0.6 \
  --fedac-beta2 0.9 \
  --cv-server-lr 0.5 \
  --cv-momentum 1.0

launch 5 fedac_c "4,9" \
  --fedac-buffer-size 5 \
  --fedac-eta-g 0.0003 \
  --fedac-beta1 0.6 \
  --fedac-beta2 0.9 \
  --cv-server-lr 0.5 \
  --cv-momentum 1.0

launch 6 fedac_d "5,10" \
  --fedac-buffer-size 5 \
  --fedac-eta-g 0.0003 \
  --fedac-beta1 0.6 \
  --fedac-beta2 0.9 \
  --cv-server-lr 0.5 \
  --cv-momentum 1.0

launch 7 fedac_e "6" \
  --fedac-buffer-size 5 \
  --fedac-eta-g 0.0003 \
  --fedac-beta1 0.6 \
  --fedac-beta2 0.9 \
  --cv-server-lr 0.5 \
  --cv-momentum 1.0
