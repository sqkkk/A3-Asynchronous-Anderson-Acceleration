#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/cifar10_workers20_fedac_exact_bank7_logs"

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
  --fedac-gamma 1.0
)

launch_job() {
  local device_id="$1"
  local cfg="$2"
  shift 2
  (
    bash ./run_cifar10_seed_list.sh "$device_id" fedac "cifar10_workers20_fedac_exact_bank7_${cfg}_seed" "1" "${SHARED_ARGS[@]}" "$@"
  ) > "$LOG_DIR/${cfg}_seed1.log" 2>&1 &
}

launch_job 4 g0_smooth_buf7_e1_eta1e3 \
  --fedac-buffer-size 7 \
  --fedac-local-epochs 1 \
  --fedac-eta-g 0.001

launch_job 5 g1_smooth_buf11_e1_eta1e3 \
  --fedac-buffer-size 11 \
  --fedac-local-epochs 1 \
  --fedac-eta-g 0.001

launch_job 6 g2_smooth_buf7_e2_eta5e4 \
  --fedac-buffer-size 7 \
  --fedac-local-epochs 2 \
  --fedac-eta-g 0.0005

launch_job 7 g3_smooth_buf11_e2_eta5e4 \
  --fedac-buffer-size 11 \
  --fedac-local-epochs 2 \
  --fedac-eta-g 0.0005

wait
echo "done"
