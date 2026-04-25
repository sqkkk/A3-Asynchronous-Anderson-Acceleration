#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/cifar10_workers20_fedac_bank10_legacy_like_logs"

mkdir -p "$LOG_DIR"
cd "$ROOT"

SHARED_ARGS=(
  --num-workers 20
  --batch-size 64
  --epochs 200
  --lr 0.01
  --delay-gap 0.0
  --delay-jitter 0.0
  --partial-dump-every-epoch 1
  --early-abort-epoch 20
  --early-abort-min-acc 25.0
  --fedac-beta1 0.6
  --fedac-beta2 0.9
  --fedac-buffer-size 7
  --fedac-local-epochs 3
  --fedac-lr-mode shared
  --fedac-persist-optimizer 0
)

launch_job() {
  local device_id="$1"
  local cfg="$2"
  shift 2
  (
    bash ./run_cifar10_seed_list.sh "$device_id" fedac "cifar10_workers20_fedac_bank10_${cfg}_seed" "1" "${SHARED_ARGS[@]}" "$@"
  ) > "$LOG_DIR/${cfg}_seed1.log" 2>&1 &
}

launch_job 4 k0_eta2p5e4 --fedac-eta-g 0.00025
launch_job 5 k1_eta3e4 --fedac-eta-g 0.00030
launch_job 6 k2_eta3p5e4 --fedac-eta-g 0.00035
launch_job 7 k3_eta4e4 --fedac-eta-g 0.00040

wait
echo "done"
