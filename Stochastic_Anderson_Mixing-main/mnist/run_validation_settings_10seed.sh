#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
EXP_DIR="$ROOT/experiments"
LOG_DIR="$EXP_DIR/validation_settings_10seed_logs"
PYTHON_BIN="/mnt/liuyx_data/miniconda/envs/a3/bin/python"

mkdir -p "$LOG_DIR"
cd "$ROOT"

# Reuse available seed-1 results where they already match the target protocol.
cp -f "$EXP_DIR/validate_labelsorted_asyncsam_seed1.pkl" "$EXP_DIR/validate10_labelsorted_asyncsam_rms_seed1.pkl"
cp -f "$EXP_DIR/validate_labelsorted_asyncsgd_rms_seed1.pkl" "$EXP_DIR/validate10_labelsorted_asyncsgd_rms_seed1.pkl"
cp -f "$EXP_DIR/validate_labelsorted_fedbuff_seed1.pkl" "$EXP_DIR/validate10_labelsorted_fedbuff_seed1.pkl"

cp -f "$EXP_DIR/validate_highdelay_asyncsam_seed1.pkl" "$EXP_DIR/validate10_highdelay_asyncsam_rms_seed1.pkl"
cp -f "$EXP_DIR/validate_highdelay_asyncsgd_rms_seed1.pkl" "$EXP_DIR/validate10_highdelay_asyncsgd_rms_seed1.pkl"
cp -f "$EXP_DIR/validate_highdelay_fedbuff_seed1.pkl" "$EXP_DIR/validate10_highdelay_fedbuff_seed1.pkl"

cp -f "$EXP_DIR/validate_workers20_asyncsam_mix02_seed1.pkl" "$EXP_DIR/validate10_workers20_asyncsam_rms_seed1.pkl"
cp -f "$EXP_DIR/validate_workers20_asyncsgd_rms_seed1.pkl" "$EXP_DIR/validate10_workers20_asyncsgd_rms_seed1.pkl"
cp -f "$EXP_DIR/validate_workers20_fedbuff_seed1.pkl" "$EXP_DIR/validate10_workers20_fedbuff_seed1.pkl"

(
  ./run_mnist_seed_list.sh 0 asyncsam validate10_labelsorted_asyncsam_rms_seed 2,3,4,5,6,7,8,9,10 \
    --partition label_sorted \
    --sam-precond rms \
    --sam-precond-beta 0.90 \
    --sam-precond-init 1.0 \
    --sam-batch-accept \
    --sam-batch-tol 0.0
) > "$LOG_DIR/labelsorted_asyncsam_rms.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 1 asyncsam validate10_labelsorted_asyncsam_seed 1,2,3,4,5,6,7,8,9,10 \
    --partition label_sorted \
    --sam-batch-accept \
    --sam-batch-tol 0.0
) > "$LOG_DIR/labelsorted_asyncsam.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 2 asyncsgd validate10_labelsorted_asyncsgd_rms_seed 2,3,4,5,6,7,8,9,10 \
    --partition label_sorted \
    --sam-precond rms \
    --sam-precond-beta 0.90 \
    --sam-precond-init 1.0
) > "$LOG_DIR/labelsorted_asyncsgd_rms.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 3 fedbuff validate10_labelsorted_fedbuff_seed 2,3,4,5,6,7,8,9,10 \
    --partition label_sorted \
    --fedbuff-k 3 \
    --fedbuff-etag 5.0
) > "$LOG_DIR/labelsorted_fedbuff.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 4 fedasync validate10_labelsorted_fedasync_seed 1,2,3,4,5,6,7,8,9,10 \
    --partition label_sorted \
    --fedasync-decay 1.0
) > "$LOG_DIR/labelsorted_fedasync.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 5 fedac validate10_labelsorted_fedac_tuned_seed 1,2,3,4,5,6,7,8,9,10 \
    --partition label_sorted \
    --fedac-buffer-size 5 \
    --fedac-eta-g 0.0003 \
    --fedac-beta1 0.6 \
    --fedac-beta2 0.9 \
    --cv-server-lr 0.5 \
    --cv-momentum 1.0
) > "$LOG_DIR/labelsorted_fedac.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 6 asyncsam validate10_highdelay_asyncsam_rms_seed 2,3,4,5,6,7,8,9,10 \
    --delay-gap 1.0 \
    --delay-jitter 0.2 \
    --sam-precond rms \
    --sam-precond-beta 0.90 \
    --sam-precond-init 1.0 \
    --sam-batch-accept \
    --sam-batch-tol 0.0
) > "$LOG_DIR/highdelay_asyncsam_rms.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 7 asyncsam validate10_highdelay_asyncsam_seed 1,2,3,4,5,6,7,8,9,10 \
    --delay-gap 1.0 \
    --delay-jitter 0.2 \
    --sam-batch-accept \
    --sam-batch-tol 0.0
) > "$LOG_DIR/highdelay_asyncsam.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 0 asyncsgd validate10_highdelay_asyncsgd_rms_seed 2,3,4,5,6,7,8,9,10 \
    --delay-gap 1.0 \
    --delay-jitter 0.2 \
    --sam-precond rms \
    --sam-precond-beta 0.90 \
    --sam-precond-init 1.0
) > "$LOG_DIR/highdelay_asyncsgd_rms.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 1 fedbuff validate10_highdelay_fedbuff_seed 2,3,4,5,6,7,8,9,10 \
    --delay-gap 1.0 \
    --delay-jitter 0.2 \
    --fedbuff-k 3 \
    --fedbuff-etag 5.0
) > "$LOG_DIR/highdelay_fedbuff.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 2 fedasync validate10_highdelay_fedasync_seed 1,2,3,4,5,6,7,8,9,10 \
    --delay-gap 1.0 \
    --delay-jitter 0.2 \
    --fedasync-decay 1.0
) > "$LOG_DIR/highdelay_fedasync.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 3 fedac validate10_highdelay_fedac_tuned_seed 1,2,3,4,5,6,7,8,9,10 \
    --delay-gap 1.0 \
    --delay-jitter 0.2 \
    --fedac-buffer-size 5 \
    --fedac-eta-g 0.0003 \
    --fedac-beta1 0.6 \
    --fedac-beta2 0.9 \
    --cv-server-lr 0.5 \
    --cv-momentum 1.0
) > "$LOG_DIR/highdelay_fedac.log" 2>&1 &

wait

(
  ./run_mnist_seed_list.sh 0 asyncsam validate10_workers20_asyncsam_rms_seed 2,3,4,5,6,7,8,9,10 \
    --num-workers 20 \
    --sam-precond rms \
    --sam-precond-beta 0.90 \
    --sam-precond-init 1.0 \
    --sam-batch-accept \
    --sam-batch-tol 0.0 \
    --sam-stale-base-mix 0.2
) > "$LOG_DIR/workers20_asyncsam_rms.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 1 asyncsam validate10_workers20_asyncsam_seed 1,2,3,4,5,6,7,8,9,10 \
    --num-workers 20 \
    --sam-batch-accept \
    --sam-batch-tol 0.0 \
    --sam-stale-base-mix 0.2
) > "$LOG_DIR/workers20_asyncsam.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 2 asyncsgd validate10_workers20_asyncsgd_rms_seed 2,3,4,5,6,7,8,9,10 \
    --num-workers 20 \
    --sam-precond rms \
    --sam-precond-beta 0.90 \
    --sam-precond-init 1.0
) > "$LOG_DIR/workers20_asyncsgd_rms.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 3 fedbuff validate10_workers20_fedbuff_seed 2,3,4,5,6,7,8,9,10 \
    --num-workers 20 \
    --fedbuff-k 3 \
    --fedbuff-etag 5.0
) > "$LOG_DIR/workers20_fedbuff.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 4 fedasync validate10_workers20_fedasync_seed 1,2,3,4,5,6,7,8,9,10 \
    --num-workers 20 \
    --fedasync-decay 1.0
) > "$LOG_DIR/workers20_fedasync.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 5 fedac validate10_workers20_fedac_tuned_seed 1,2,3,4,5,6,7,8,9,10 \
    --num-workers 20 \
    --fedac-buffer-size 5 \
    --fedac-eta-g 0.0003 \
    --fedac-beta1 0.6 \
    --fedac-beta2 0.9 \
    --cv-server-lr 0.5 \
    --cv-momentum 1.0
) > "$LOG_DIR/workers20_fedac.log" 2>&1 &

wait

"$PYTHON_BIN" "$ROOT/plot_validation_settings_10seed.py"
