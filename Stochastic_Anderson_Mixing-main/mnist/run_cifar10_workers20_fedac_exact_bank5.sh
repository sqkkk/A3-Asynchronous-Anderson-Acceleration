#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/cifar10_workers20_fedac_exact_bank5_logs"

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
  --early-abort-min-acc 20.0
  --fedac-beta1 0.6
  --fedac-beta2 0.9
)

launch_job() {
  local device_id="$1"
  local cfg="$2"
  shift 2
  (
    bash ./run_cifar10_seed_list.sh "$device_id" fedac "cifar10_workers20_fedac_exact_bank5_${cfg}_seed" "1" "${SHARED_ARGS[@]}" "$@"
  ) > "$LOG_DIR/${cfg}_seed1.log" 2>&1 &
}

# Baseline around the current best exact region, now with persistent local
# momentum state and AFL-style local-lr decay semantics.
launch_job 4 e0_buf7_e3_eta3e4_g099 \
  --fedac-buffer-size 7 \
  --fedac-eta-g 0.0003 \
  --fedac-gamma 0.99 \
  --fedac-local-epochs 3

launch_job 5 e1_buf7_e5_eta3e4_g099 \
  --fedac-buffer-size 7 \
  --fedac-eta-g 0.0003 \
  --fedac-gamma 0.99 \
  --fedac-local-epochs 5

launch_job 6 e2_buf9_e3_eta3e4_g099 \
  --fedac-buffer-size 9 \
  --fedac-eta-g 0.0003 \
  --fedac-gamma 0.99 \
  --fedac-local-epochs 3

launch_job 7 e3_buf9_e5_eta3e4_g099 \
  --fedac-buffer-size 9 \
  --fedac-eta-g 0.0003 \
  --fedac-gamma 0.99 \
  --fedac-local-epochs 5

launch_job 5 e4_buf7_e3_eta5e4_g099 \
  --fedac-buffer-size 7 \
  --fedac-eta-g 0.0005 \
  --fedac-gamma 0.99 \
  --fedac-local-epochs 3

launch_job 6 e5_buf7_e3_eta3e4_g100 \
  --fedac-buffer-size 7 \
  --fedac-eta-g 0.0003 \
  --fedac-gamma 1.0 \
  --fedac-local-epochs 3

wait
echo "done"
