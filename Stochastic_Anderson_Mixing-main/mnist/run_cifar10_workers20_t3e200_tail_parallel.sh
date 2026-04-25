#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/cifar10_workers20_t3e200_10seed_logs"

mkdir -p "$LOG_DIR"
cd "$ROOT"

SEEDS_ODD_TAIL="7,9"
SEEDS_EVEN_TAIL="8,10"

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

# Back-half seed helpers. Using later seeds reduces the chance of colliding
# with the main launcher before the helper finishes and lets us exploit free HBM.
launch_job 0 "$SEEDS_ODD_TAIL" asyncsam_rms_tail_odd.log asyncsam cifar10_workers20_t3e200_asyncsam_rms_seed \
  "${ASYNC_SAM_RMS_ARGS[@]}"
launch_job 1 "$SEEDS_EVEN_TAIL" asyncsam_rms_tail_even.log asyncsam cifar10_workers20_t3e200_asyncsam_rms_seed \
  "${ASYNC_SAM_RMS_ARGS[@]}"

launch_job 2 "$SEEDS_ODD_TAIL" asyncsgd_tail_odd.log asyncsgd cifar10_workers20_t3e200_asyncsgd_seed
launch_job 3 "$SEEDS_EVEN_TAIL" asyncsgd_tail_even.log asyncsgd cifar10_workers20_t3e200_asyncsgd_seed

launch_job 4 "$SEEDS_ODD_TAIL" fedasync_tail_odd.log fedasync cifar10_workers20_t3e200_fedasync_seed \
  "${FEDASYNC_ARGS[@]}"
launch_job 5 "$SEEDS_EVEN_TAIL" fedasync_tail_even.log fedasync cifar10_workers20_t3e200_fedasync_seed \
  "${FEDASYNC_ARGS[@]}"

launch_job 6 "$SEEDS_ODD_TAIL" fedbuff_tail_odd.log fedbuff cifar10_workers20_t3e200_fedbuff_seed \
  "${FEDBUFF_ARGS[@]}"
launch_job 7 "$SEEDS_EVEN_TAIL" fedbuff_tail_even.log fedbuff cifar10_workers20_t3e200_fedbuff_seed \
  "${FEDBUFF_ARGS[@]}"

launch_job 0 "$SEEDS_ODD_TAIL" fedac_tail_odd.log fedac cifar10_workers20_t3e200_fedac_tuned_seed \
  "${FEDAC_ARGS[@]}"
launch_job 1 "$SEEDS_EVEN_TAIL" fedac_tail_even.log fedac cifar10_workers20_t3e200_fedac_tuned_seed \
  "${FEDAC_ARGS[@]}"

wait
echo "tail_done"
