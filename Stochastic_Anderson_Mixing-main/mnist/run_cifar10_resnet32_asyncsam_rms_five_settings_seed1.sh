#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/cifar10_resnet32_asyncsam_rms_five_settings_seed1_logs"
mkdir -p "$LOG_DIR"
cd "$ROOT"

SEEDS="1"

COMMON_ARGS=(
  --model resnet32
  --batch-size 64
  --epochs 100
  --lr 0.05
  --sam-momentum 0.9
  --sam-precond rms
  --sam-precond-beta 0.90
  --sam-precond-init 1.0
  --sam-precond-min-denom 0.1
  --sam-hist-length 4
  --sam-period 2
  --sam-base-mix 0.8
  --sam-max-history-staleness 2
  --sam-max-cond 10000
  --sam-max-step-ratio 1.0
  --sam-anchor-tol 1.0
  --grad-clip-norm 0.0
  --partial-dump-every-epoch 1
)

launch_job() {
  local device_id="$1"
  local log_name="$2"
  local prefix="$3"
  shift 3
  (
    bash ./run_cifar10_seed_list.sh "$device_id" asyncsam "$prefix" "$SEEDS" "${COMMON_ARGS[@]}" "$@"
  ) > "$LOG_DIR/$log_name" 2>&1 &
}

launch_job 0 base.log cifar10_resnet32_p3check_base_asyncsam_rms_seed
launch_job 1 labelsorted.log cifar10_resnet32_p3check_labelsorted_asyncsam_rms_seed \
  --partition label_sorted
launch_job 2 dirichlet.log cifar10_resnet32_p3check_dirichlet_asyncsam_rms_seed \
  --partition dirichlet \
  --dirichlet-alpha 0.05 \
  --dirichlet-min-size 10
launch_job 3 highdelay.log cifar10_resnet32_p3check_highdelay_asyncsam_rms_seed \
  --delay-gap 0.5 \
  --delay-jitter 0.1
launch_job 4 workers20.log cifar10_resnet32_p3check_workers20_asyncsam_rms_seed \
  --num-workers 20 \
  --delay-gap 0.0 \
  --delay-jitter 0.0

wait
echo "done"
