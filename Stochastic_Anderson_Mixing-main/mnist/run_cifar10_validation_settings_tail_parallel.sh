#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
EXP_DIR="$ROOT/experiments"
LOG_DIR="$EXP_DIR/cifar10_validation_settings_10seed_logs"

mkdir -p "$LOG_DIR"
cd "$ROOT"

SEEDS="5,6,7,8,9,10"

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

reserve_tail_outputs() {
  local prefix="$1"
  local seed
  for seed in 5 6 7 8 9 10; do
    local out_file="$EXP_DIR/${prefix}${seed}.pkl"
    if [ ! -f "$out_file" ]; then
      : > "$out_file"
    fi
  done
}

launch_tail_job() {
  local device_id="$1"
  local log_name="$2"
  local alg="$3"
  local prefix="$4"
  shift 4
  reserve_tail_outputs "$prefix"
  (
    bash ./run_cifar10_seed_list.sh "$device_id" "$alg" "$prefix" "$SEEDS" "$@"
  ) > "$LOG_DIR/$log_name" 2>&1 &
}

# Offload the tail seeds to use the spare NPU memory headroom and shorten wall-clock time.
launch_tail_job 7 labelsorted_asyncsam_rms_tail.log asyncsam cifar10_validate10_labelsorted_asyncsam_rms_seed \
  --partition label_sorted \
  "${ASYNC_SAM_RMS_ARGS[@]}"
launch_tail_job 6 labelsorted_asyncsgd_tail.log asyncsgd cifar10_validate10_labelsorted_asyncsgd_seed \
  --partition label_sorted
launch_tail_job 5 labelsorted_fedasync_tail.log fedasync cifar10_validate10_labelsorted_fedasync_seed \
  --partition label_sorted \
  "${FEDASYNC_ARGS[@]}"
launch_tail_job 4 labelsorted_fedbuff_tail.log fedbuff cifar10_validate10_labelsorted_fedbuff_seed \
  --partition label_sorted \
  "${FEDBUFF_ARGS[@]}"
launch_tail_job 3 labelsorted_fedac_tail.log fedac cifar10_validate10_labelsorted_fedac_tuned_seed \
  --partition label_sorted \
  "${FEDAC_ARGS[@]}"

launch_tail_job 2 highdelay_asyncsam_rms_tail.log asyncsam cifar10_validate10_highdelay_asyncsam_rms_seed \
  --delay-gap 0.5 \
  --delay-jitter 0.1 \
  "${ASYNC_SAM_RMS_ARGS[@]}"
launch_tail_job 1 highdelay_asyncsgd_tail.log asyncsgd cifar10_validate10_highdelay_asyncsgd_seed \
  --delay-gap 0.5 \
  --delay-jitter 0.1
launch_tail_job 0 highdelay_fedasync_tail.log fedasync cifar10_validate10_highdelay_fedasync_seed \
  --delay-gap 0.5 \
  --delay-jitter 0.1 \
  "${FEDASYNC_ARGS[@]}"
launch_tail_job 7 highdelay_fedbuff_tail.log fedbuff cifar10_validate10_highdelay_fedbuff_seed \
  --delay-gap 0.5 \
  --delay-jitter 0.1 \
  "${FEDBUFF_ARGS[@]}"
launch_tail_job 6 highdelay_fedac_tail.log fedac cifar10_validate10_highdelay_fedac_tuned_seed \
  --delay-gap 0.5 \
  --delay-jitter 0.1 \
  "${FEDAC_ARGS[@]}"

launch_tail_job 5 workers20_asyncsam_rms_tail.log asyncsam cifar10_validate10_workers20_asyncsam_rms_seed \
  --num-workers 20 \
  --sam-stale-base-mix 0.2 \
  "${ASYNC_SAM_RMS_ARGS[@]}"
launch_tail_job 4 workers20_asyncsgd_tail.log asyncsgd cifar10_validate10_workers20_asyncsgd_seed \
  --num-workers 20
launch_tail_job 3 workers20_fedasync_tail.log fedasync cifar10_validate10_workers20_fedasync_seed \
  --num-workers 20 \
  "${FEDASYNC_ARGS[@]}"
launch_tail_job 2 workers20_fedbuff_tail.log fedbuff cifar10_validate10_workers20_fedbuff_seed \
  --num-workers 20 \
  "${FEDBUFF_ARGS[@]}"
launch_tail_job 1 workers20_fedac_tail.log fedac cifar10_validate10_workers20_fedac_tuned_seed \
  --num-workers 20 \
  "${FEDAC_ARGS[@]}"

wait
echo "done"
