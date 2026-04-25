#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/cifar10_fedac_10seed_logs"
mkdir -p "$LOG_DIR"

cd "$ROOT"

# Shared CIFAR-10 protocol is inherited from run_cifar10_seed_list.sh.
# These are FedAC-only knobs selected from the seed10 sweep.
COMMON_FEDAC_ARGS=(
  --fedac-buffer-size 5
  --fedac-eta-g 0.0003
  --fedac-beta1 0.6
  --fedac-beta2 0.9
  --cv-server-lr 0.5
  --cv-momentum 1.0
  --grad-clip-norm 0.0
)

run_seed_group() {
  local device_id="$1"
  local seed_csv="$2"
  bash ./run_cifar10_seed_list.sh "$device_id" fedac cifar10_multiseed_fedac_tuned_seed "$seed_csv" \
    "${COMMON_FEDAC_ARGS[@]}" \
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
