#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/cifar10_workers20_fedac_exact_bank8_logs"

mkdir -p "$LOG_DIR"
cd "$ROOT"

SHARED_ARGS=(
  --num-workers 20
  --batch-size 64
  --epochs 100
  --lr 0.01
  --delay-gap 0.0
  --delay-jitter 0.0
  --partial-dump-every-epoch 1
  --early-abort-epoch 10
  --early-abort-min-acc 15.0
  --fedac-beta1 0.6
  --fedac-beta2 0.9
  --fedac-gamma 0.99
  --fedac-buffer-size 7
  --fedac-local-epochs 3
)

launch_job() {
  local device_id="$1"
  local cfg="$2"
  shift 2
  (
    bash ./run_cifar10_seed_list.sh "$device_id" fedac "cifar10_workers20_fedac_exact_bank8_${cfg}_seed" "1" "${SHARED_ARGS[@]}" "$@"
  ) > "$LOG_DIR/${cfg}_seed1.log" 2>&1 &
}

launch_job 4 h0_eta3e4 --fedac-eta-g 0.0003
launch_job 5 h1_eta5e4 --fedac-eta-g 0.0005
launch_job 6 h2_eta7e4 --fedac-eta-g 0.0007
launch_job 7 h3_eta1e3 --fedac-eta-g 0.0010

wait
echo "done"
