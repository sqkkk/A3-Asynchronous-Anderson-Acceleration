#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/fedbuff_rms_10seed_logs"
mkdir -p "$LOG_DIR"

cd "$ROOT"

# Reuse seed1 if it already exists; backfill 2..10 in parallel across 8 NPUs.
./run_mnist_seed_list.sh 0 fedbuff fedbuff_rms_iid_seed 2,10 \
  --fedbuff-k 3 --fedbuff-etag 5.0 \
  --sam-precond rms --sam-precond-beta 0.90 --sam-precond-init 1.0 \
  > "$LOG_DIR/dev0.log" 2>&1 &
P0=$!

./run_mnist_seed_list.sh 1 fedbuff fedbuff_rms_iid_seed 3 \
  --fedbuff-k 3 --fedbuff-etag 5.0 \
  --sam-precond rms --sam-precond-beta 0.90 --sam-precond-init 1.0 \
  > "$LOG_DIR/dev1.log" 2>&1 &
P1=$!

./run_mnist_seed_list.sh 2 fedbuff fedbuff_rms_iid_seed 4 \
  --fedbuff-k 3 --fedbuff-etag 5.0 \
  --sam-precond rms --sam-precond-beta 0.90 --sam-precond-init 1.0 \
  > "$LOG_DIR/dev2.log" 2>&1 &
P2=$!

./run_mnist_seed_list.sh 3 fedbuff fedbuff_rms_iid_seed 5 \
  --fedbuff-k 3 --fedbuff-etag 5.0 \
  --sam-precond rms --sam-precond-beta 0.90 --sam-precond-init 1.0 \
  > "$LOG_DIR/dev3.log" 2>&1 &
P3=$!

./run_mnist_seed_list.sh 4 fedbuff fedbuff_rms_iid_seed 6 \
  --fedbuff-k 3 --fedbuff-etag 5.0 \
  --sam-precond rms --sam-precond-beta 0.90 --sam-precond-init 1.0 \
  > "$LOG_DIR/dev4.log" 2>&1 &
P4=$!

./run_mnist_seed_list.sh 5 fedbuff fedbuff_rms_iid_seed 7 \
  --fedbuff-k 3 --fedbuff-etag 5.0 \
  --sam-precond rms --sam-precond-beta 0.90 --sam-precond-init 1.0 \
  > "$LOG_DIR/dev5.log" 2>&1 &
P5=$!

./run_mnist_seed_list.sh 6 fedbuff fedbuff_rms_iid_seed 8 \
  --fedbuff-k 3 --fedbuff-etag 5.0 \
  --sam-precond rms --sam-precond-beta 0.90 --sam-precond-init 1.0 \
  > "$LOG_DIR/dev6.log" 2>&1 &
P6=$!

./run_mnist_seed_list.sh 7 fedbuff fedbuff_rms_iid_seed 9 \
  --fedbuff-k 3 --fedbuff-etag 5.0 \
  --sam-precond rms --sam-precond-beta 0.90 --sam-precond-init 1.0 \
  > "$LOG_DIR/dev7.log" 2>&1 &
P7=$!

echo "launched"
wait "$P0" "$P1" "$P2" "$P3" "$P4" "$P5" "$P6" "$P7"
echo "done"
