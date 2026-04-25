#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/cifar10_10seed_logs"
mkdir -p "$LOG_DIR"

cd "$ROOT"

# Reuse the existing seed=10 runs; backfill seeds 1..9.
ODD="1,3,5,7,9"
EVEN="2,4,6,8,10"

bash ./run_cifar10_seed_list.sh 0 asyncsam cifar10_multiseed_asyncsam_rms_legacy92_seed "$ODD" \
  --sam-momentum 0.9 \
  --sam-precond rms --sam-precond-beta 0.90 --sam-precond-init 1.0 --sam-precond-min-denom 0.1 \
  --sam-hist-length 4 --sam-period 2 --sam-base-mix 0.8 --sam-max-history-staleness 2 \
  --sam-max-cond 10000 --sam-max-step-ratio 1.0 --sam-anchor-tol 1.0 --grad-clip-norm 0.0 \
  > "$LOG_DIR/asyncsam_rms_odd.log" 2>&1 &
P0=$!

bash ./run_cifar10_seed_list.sh 1 asyncsam cifar10_multiseed_asyncsam_rms_legacy92_seed "$EVEN" \
  --sam-momentum 0.9 \
  --sam-precond rms --sam-precond-beta 0.90 --sam-precond-init 1.0 --sam-precond-min-denom 0.1 \
  --sam-hist-length 4 --sam-period 2 --sam-base-mix 0.8 --sam-max-history-staleness 2 \
  --sam-max-cond 10000 --sam-max-step-ratio 1.0 --sam-anchor-tol 1.0 --grad-clip-norm 0.0 \
  > "$LOG_DIR/asyncsam_rms_even.log" 2>&1 &
P1=$!

bash ./run_cifar10_seed_list.sh 2 asyncsgd cifar10_multiseed_asyncsgd_seed "$ODD" \
  > "$LOG_DIR/asyncsgd_odd.log" 2>&1 &
P2=$!

bash ./run_cifar10_seed_list.sh 3 asyncsgd cifar10_multiseed_asyncsgd_seed "$EVEN" \
  > "$LOG_DIR/asyncsgd_even.log" 2>&1 &
P3=$!

bash ./run_cifar10_seed_list.sh 4 fedasync cifar10_multiseed_fedasync_seed "$ODD" \
  > "$LOG_DIR/fedasync_odd.log" 2>&1 &
P4=$!

bash ./run_cifar10_seed_list.sh 5 fedasync cifar10_multiseed_fedasync_seed "$EVEN" \
  > "$LOG_DIR/fedasync_even.log" 2>&1 &
P5=$!

bash ./run_cifar10_seed_list.sh 6 fedbuff cifar10_multiseed_fedbuff_seed "$ODD" \
  --fedbuff-k 3 --fedbuff-etag 5.0 \
  > "$LOG_DIR/fedbuff_odd.log" 2>&1 &
P6=$!

bash ./run_cifar10_seed_list.sh 7 fedbuff cifar10_multiseed_fedbuff_seed "$EVEN" \
  --fedbuff-k 3 --fedbuff-etag 5.0 \
  > "$LOG_DIR/fedbuff_even.log" 2>&1 &
P7=$!

wait "$P0" "$P1" "$P2" "$P3" "$P4" "$P5" "$P6" "$P7"

echo "done"
