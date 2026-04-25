#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/validation_settings_10seed_logs"

mkdir -p "$LOG_DIR"
cd "$ROOT"

(
  ./run_mnist_seed_list.sh 0 fedasync validate10_workers20_fedasync_retry_seed 1,5,9 \
    --num-workers 20 \
    --fedasync-decay 1.0
) > "$LOG_DIR/workers20_fedasync_retry_odd.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 1 fedasync validate10_workers20_fedasync_retry_seed 2,6,10 \
    --num-workers 20 \
    --fedasync-decay 1.0
) > "$LOG_DIR/workers20_fedasync_retry_even.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 2 fedasync validate10_workers20_fedasync_retry_seed 3,7 \
    --num-workers 20 \
    --fedasync-decay 1.0
) > "$LOG_DIR/workers20_fedasync_retry_extra_a.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 3 fedasync validate10_workers20_fedasync_retry_seed 4,8 \
    --num-workers 20 \
    --fedasync-decay 1.0
) > "$LOG_DIR/workers20_fedasync_retry_extra_b.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 4 fedac validate10_workers20_fedac_tuned_retry_seed 1,5,9 \
    --num-workers 20 \
    --fedac-buffer-size 5 \
    --fedac-eta-g 0.0003 \
    --fedac-beta1 0.6 \
    --fedac-beta2 0.9 \
    --cv-server-lr 0.5 \
    --cv-momentum 1.0
) > "$LOG_DIR/workers20_fedac_retry_odd.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 5 fedac validate10_workers20_fedac_tuned_retry_seed 2,6,10 \
    --num-workers 20 \
    --fedac-buffer-size 5 \
    --fedac-eta-g 0.0003 \
    --fedac-beta1 0.6 \
    --fedac-beta2 0.9 \
    --cv-server-lr 0.5 \
    --cv-momentum 1.0
) > "$LOG_DIR/workers20_fedac_retry_even.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 6 fedac validate10_workers20_fedac_tuned_retry_seed 3,7 \
    --num-workers 20 \
    --fedac-buffer-size 5 \
    --fedac-eta-g 0.0003 \
    --fedac-beta1 0.6 \
    --fedac-beta2 0.9 \
    --cv-server-lr 0.5 \
    --cv-momentum 1.0
) > "$LOG_DIR/workers20_fedac_retry_extra_a.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 7 fedac validate10_workers20_fedac_tuned_retry_seed 4,8 \
    --num-workers 20 \
    --fedac-buffer-size 5 \
    --fedac-eta-g 0.0003 \
    --fedac-beta1 0.6 \
    --fedac-beta2 0.9 \
    --cv-server-lr 0.5 \
    --cv-momentum 1.0
) > "$LOG_DIR/workers20_fedac_retry_extra_b.log" 2>&1 &

wait
