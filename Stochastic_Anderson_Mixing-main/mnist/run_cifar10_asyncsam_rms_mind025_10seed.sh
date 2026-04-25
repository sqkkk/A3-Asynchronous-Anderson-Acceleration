#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/cifar10_asyncsam_rms_mind025_10seed_logs"
mkdir -p "$LOG_DIR"

cd "$ROOT"

COMMON_SAM_ARGS=(
  --sam-precond rms
  --sam-precond-beta 0.90
  --sam-precond-init 1.0
  --sam-precond-min-denom 0.25
  --sam-period 2
  --sam-min-cosine 0.10
  --sam-base-mix 0.50
)

run_seed_group() {
  local device_id="$1"
  local seed_csv="$2"
  bash ./run_cifar10_seed_list.sh "$device_id" asyncsam cifar10_multiseed_asyncsam_rms_mind025_seed "$seed_csv" \
    "${COMMON_SAM_ARGS[@]}" \
    > "$LOG_DIR/device${device_id}_seeds_${seed_csv//,/}.log" 2>&1 &
}

run_seed_group 0 1
run_seed_group 1 2
run_seed_group 2 3
run_seed_group 3 4
run_seed_group 4 5
run_seed_group 5 6
run_seed_group 6 7
run_seed_group 7 8,9,10

wait
echo "done"
