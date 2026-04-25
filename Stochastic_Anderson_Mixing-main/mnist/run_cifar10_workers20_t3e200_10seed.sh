#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/cifar10_workers20_t3e200_10seed_logs"
PLOT_SCRIPT="$ROOT/plot_cifar10_workers20_t3e200_10seed.py"
PYTHON_BIN="/mnt/liuyx_data/miniconda/envs/a3/bin/python"

mkdir -p "$LOG_DIR"
cd "$ROOT"

SEEDS_ODD="1,3,5,7,9"
SEEDS_EVEN="2,4,6,8,10"

# Shared protocol chosen from the AsyncSAM+RMS workers20 search:
# - smaller batch size matters
# - extra synthetic delay hurts
# We keep this shared protocol identical across methods for fairness.
SHARED_ARGS=(
  --num-workers 20
  --batch-size 64
  --epochs 200
  --lr 0.01
  --delay-gap 0.0
  --delay-jitter 0.0
)

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
  --sam-stale-base-mix 0.2
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
  local seed_csv="$2"
  local log_name="$3"
  local alg="$4"
  local prefix="$5"
  shift 5
  (
    bash ./run_cifar10_seed_list.sh "$device_id" "$alg" "$prefix" "$seed_csv" "${SHARED_ARGS[@]}" "$@"
  ) > "$LOG_DIR/$log_name" 2>&1 &
}

# Use every available card. Since each run uses only a small fraction of 64GB HBM,
# two extra jobs are colocated to avoid idle devices and keep the machine busy.
launch_job 0 "$SEEDS_ODD" asyncsam_rms_odd.log asyncsam cifar10_workers20_t3e200_asyncsam_rms_seed \
  "${ASYNC_SAM_RMS_ARGS[@]}"
launch_job 1 "$SEEDS_EVEN" asyncsam_rms_even.log asyncsam cifar10_workers20_t3e200_asyncsam_rms_seed \
  "${ASYNC_SAM_RMS_ARGS[@]}"

launch_job 2 "$SEEDS_ODD" asyncsgd_odd.log asyncsgd cifar10_workers20_t3e200_asyncsgd_seed
launch_job 3 "$SEEDS_EVEN" asyncsgd_even.log asyncsgd cifar10_workers20_t3e200_asyncsgd_seed

launch_job 4 "$SEEDS_ODD" fedasync_odd.log fedasync cifar10_workers20_t3e200_fedasync_seed \
  "${FEDASYNC_ARGS[@]}"
launch_job 5 "$SEEDS_EVEN" fedasync_even.log fedasync cifar10_workers20_t3e200_fedasync_seed \
  "${FEDASYNC_ARGS[@]}"

launch_job 6 "$SEEDS_ODD" fedbuff_odd.log fedbuff cifar10_workers20_t3e200_fedbuff_seed \
  "${FEDBUFF_ARGS[@]}"
launch_job 7 "$SEEDS_EVEN" fedbuff_even.log fedbuff cifar10_workers20_t3e200_fedbuff_seed \
  "${FEDBUFF_ARGS[@]}"

launch_job 0 "$SEEDS_ODD" fedac_odd.log fedac cifar10_workers20_t3e200_fedac_tuned_seed \
  "${FEDAC_ARGS[@]}"
launch_job 1 "$SEEDS_EVEN" fedac_even.log fedac cifar10_workers20_t3e200_fedac_tuned_seed \
  "${FEDAC_ARGS[@]}"

wait
"$PYTHON_BIN" "$PLOT_SCRIPT"
echo "done"
