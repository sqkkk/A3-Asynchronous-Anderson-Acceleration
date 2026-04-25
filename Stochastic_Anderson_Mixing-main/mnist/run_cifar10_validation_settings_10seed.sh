#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/cifar10_validation_settings_10seed_logs"

mkdir -p "$LOG_DIR"
cd "$ROOT"

SEEDS="1,2,3,4,5,6,7,8,9,10"

# Reuse the CIFAR-10 mainline settings so the shared protocol stays fixed.
ASYNC_SAM_RMS_ARGS=(
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
)

FEDASYNC_ARGS=(
  --fedasync-decay 1.0
)

FEDBUFF_ARGS=(
  --fedbuff-k 3
  --fedbuff-etag 5.0
)

FEDAC_ARGS=(
  --fedac-buffer-size 5
  --fedac-eta-g 0.0003
  --fedac-beta1 0.6
  --fedac-beta2 0.9
  --cv-server-lr 0.5
  --cv-momentum 1.0
  --grad-clip-norm 0.0
)

launch_job() {
  local device_id="$1"
  local log_name="$2"
  local alg="$3"
  local prefix="$4"
  shift 4
  (
    bash ./run_cifar10_seed_list.sh "$device_id" "$alg" "$prefix" "$SEEDS" "$@"
  ) > "$LOG_DIR/$log_name" 2>&1 &
}

# CIFAR-10 / ResNet18: stronger non-IID split.
launch_job 0 labelsorted_asyncsam_rms.log asyncsam cifar10_validate10_labelsorted_asyncsam_rms_seed \
  --partition label_sorted \
  "${ASYNC_SAM_RMS_ARGS[@]}"
launch_job 1 labelsorted_asyncsgd.log asyncsgd cifar10_validate10_labelsorted_asyncsgd_seed \
  --partition label_sorted
launch_job 2 labelsorted_fedasync.log fedasync cifar10_validate10_labelsorted_fedasync_seed \
  --partition label_sorted \
  "${FEDASYNC_ARGS[@]}"
launch_job 3 labelsorted_fedbuff.log fedbuff cifar10_validate10_labelsorted_fedbuff_seed \
  --partition label_sorted \
  "${FEDBUFF_ARGS[@]}"
launch_job 4 labelsorted_fedac.log fedac cifar10_validate10_labelsorted_fedac_tuned_seed \
  --partition label_sorted \
  "${FEDAC_ARGS[@]}"

# CIFAR-10 / ResNet18: stronger async delay than the base 0.1/0.02 setting.
launch_job 5 highdelay_asyncsam_rms.log asyncsam cifar10_validate10_highdelay_asyncsam_rms_seed \
  --delay-gap 0.5 \
  --delay-jitter 0.1 \
  "${ASYNC_SAM_RMS_ARGS[@]}"
launch_job 6 highdelay_asyncsgd.log asyncsgd cifar10_validate10_highdelay_asyncsgd_seed \
  --delay-gap 0.5 \
  --delay-jitter 0.1
launch_job 7 highdelay_fedasync.log fedasync cifar10_validate10_highdelay_fedasync_seed \
  --delay-gap 0.5 \
  --delay-jitter 0.1 \
  "${FEDASYNC_ARGS[@]}"
launch_job 0 highdelay_fedbuff.log fedbuff cifar10_validate10_highdelay_fedbuff_seed \
  --delay-gap 0.5 \
  --delay-jitter 0.1 \
  "${FEDBUFF_ARGS[@]}"
launch_job 1 highdelay_fedac.log fedac cifar10_validate10_highdelay_fedac_tuned_seed \
  --delay-gap 0.5 \
  --delay-jitter 0.1 \
  "${FEDAC_ARGS[@]}"

# CIFAR-10 / ResNet18: higher concurrency, with a small stale-base correction.
launch_job 2 workers20_asyncsam_rms.log asyncsam cifar10_validate10_workers20_asyncsam_rms_seed \
  --num-workers 20 \
  --sam-stale-base-mix 0.2 \
  "${ASYNC_SAM_RMS_ARGS[@]}"
launch_job 3 workers20_asyncsgd.log asyncsgd cifar10_validate10_workers20_asyncsgd_seed \
  --num-workers 20
launch_job 4 workers20_fedasync.log fedasync cifar10_validate10_workers20_fedasync_seed \
  --num-workers 20 \
  "${FEDASYNC_ARGS[@]}"
launch_job 5 workers20_fedbuff.log fedbuff cifar10_validate10_workers20_fedbuff_seed \
  --num-workers 20 \
  "${FEDBUFF_ARGS[@]}"
launch_job 6 workers20_fedac.log fedac cifar10_validate10_workers20_fedac_tuned_seed \
  --num-workers 20 \
  "${FEDAC_ARGS[@]}"

wait
echo "done"
