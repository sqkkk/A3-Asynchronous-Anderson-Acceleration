#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/cifar10_workers20_fedac_exact_bank3_logs"

mkdir -p "$LOG_DIR"
cd "$ROOT"

SHARED_ARGS=(
  --num-workers 20
  --batch-size 64
  --epochs 40
  --lr 0.01
  --delay-gap 0.0
  --delay-jitter 0.0
  --partial-dump-every-epoch 1
  --early-abort-epoch 10
  --early-abort-min-acc 18.0
  --fedac-beta1 0.6
  --fedac-beta2 0.9
)

launch_job() {
  local device_id="$1"
  local cfg="$2"
  shift 2
  (
    bash ./run_cifar10_seed_list.sh "$device_id" fedac "cifar10_workers20_fedac_exact_bank3_${cfg}_seed" "1" "${SHARED_ARGS[@]}" "$@"
  ) > "$LOG_DIR/${cfg}_seed1.log" 2>&1 &
}

launch_job 0 c0_buf5_e3_eta2e4 \
  --fedac-buffer-size 5 \
  --fedac-eta-g 0.0002 \
  --fedac-local-epochs 3

launch_job 1 c1_buf5_e3_eta3e4 \
  --fedac-buffer-size 5 \
  --fedac-eta-g 0.0003 \
  --fedac-local-epochs 3

launch_job 2 c2_buf5_e3_eta4e4 \
  --fedac-buffer-size 5 \
  --fedac-eta-g 0.0004 \
  --fedac-local-epochs 3

launch_job 3 c3_buf5_e2_eta3e4 \
  --fedac-buffer-size 5 \
  --fedac-eta-g 0.0003 \
  --fedac-local-epochs 2

launch_job 4 c4_buf7_e3_eta3e4 \
  --fedac-buffer-size 7 \
  --fedac-eta-g 0.0003 \
  --fedac-local-epochs 3

launch_job 5 c5_buf5_e4_eta2e4 \
  --fedac-buffer-size 5 \
  --fedac-eta-g 0.0002 \
  --fedac-local-epochs 4

wait
echo "done"
