#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/cifar10_workers20_fedac_exact_bank6_logs"

mkdir -p "$LOG_DIR"
cd "$ROOT"

SHARED_ARGS=(
  --num-workers 20
  --batch-size 64
  --epochs 20
  --lr 0.01
  --delay-gap 0.0
  --delay-jitter 0.0
  --partial-dump-every-epoch 1
  --early-abort-epoch 6
  --early-abort-min-acc 18.0
  --fedac-beta1 0.9
  --fedac-beta2 0.99
  --fedac-gamma 0.99
  --fedac-local-epochs 3
)

launch_job() {
  local device_id="$1"
  local cfg="$2"
  shift 2
  (
    bash ./run_cifar10_seed_list.sh "$device_id" fedac "cifar10_workers20_fedac_exact_bank6_${cfg}_seed" "1" "${SHARED_ARGS[@]}" "$@"
  ) > "$LOG_DIR/${cfg}_seed1.log" 2>&1 &
}

launch_job 0 f0_smooth_buf7_eta1e4 \
  --fedac-buffer-size 7 \
  --fedac-eta-g 0.0001

launch_job 1 f1_smooth_buf9_eta1e4 \
  --fedac-buffer-size 9 \
  --fedac-eta-g 0.0001

launch_job 2 f2_smooth_buf7_eta2e4 \
  --fedac-buffer-size 7 \
  --fedac-eta-g 0.0002

launch_job 3 f3_smooth_buf9_eta2e4 \
  --fedac-buffer-size 9 \
  --fedac-eta-g 0.0002

wait
echo "done"
