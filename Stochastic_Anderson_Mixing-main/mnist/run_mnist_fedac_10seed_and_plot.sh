#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
EXP_DIR="$ROOT/experiments"
LOG_DIR="$EXP_DIR/mnist_fedac_10seed_logs"
PYTHON_BIN="/mnt/liuyx_data/miniconda/envs/a3/bin/python"

mkdir -p "$LOG_DIR"
cp -f "$EXP_DIR/fedac_mnist_tune_eta3e4_buf5_cv05.pkl" "$EXP_DIR/afl_same_setting_fedac_tuned_seed1.pkl"

cd "$ROOT"

(
  ./run_mnist_seed_list.sh 3 fedac afl_same_setting_fedac_tuned_seed 2,7 \
    --fedac-buffer-size 5 \
    --fedac-eta-g 0.0003 \
    --fedac-beta1 0.6 \
    --fedac-beta2 0.9 \
    --cv-server-lr 0.5 \
    --cv-momentum 1.0
) > "$LOG_DIR/fedac_a.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 4 fedac afl_same_setting_fedac_tuned_seed 3,8 \
    --fedac-buffer-size 5 \
    --fedac-eta-g 0.0003 \
    --fedac-beta1 0.6 \
    --fedac-beta2 0.9 \
    --cv-server-lr 0.5 \
    --cv-momentum 1.0
) > "$LOG_DIR/fedac_b.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 5 fedac afl_same_setting_fedac_tuned_seed 4,9 \
    --fedac-buffer-size 5 \
    --fedac-eta-g 0.0003 \
    --fedac-beta1 0.6 \
    --fedac-beta2 0.9 \
    --cv-server-lr 0.5 \
    --cv-momentum 1.0
) > "$LOG_DIR/fedac_c.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 6 fedac afl_same_setting_fedac_tuned_seed 5,10 \
    --fedac-buffer-size 5 \
    --fedac-eta-g 0.0003 \
    --fedac-beta1 0.6 \
    --fedac-beta2 0.9 \
    --cv-server-lr 0.5 \
    --cv-momentum 1.0
) > "$LOG_DIR/fedac_d.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 7 fedac afl_same_setting_fedac_tuned_seed 6 \
    --fedac-buffer-size 5 \
    --fedac-eta-g 0.0003 \
    --fedac-beta1 0.6 \
    --fedac-beta2 0.9 \
    --cv-server-lr 0.5 \
    --cv-momentum 1.0
) > "$LOG_DIR/fedac_e.log" 2>&1 &

wait

"$PYTHON_BIN" "$ROOT/plot_async_multiseed_compare.py" \
  --output-dir "$EXP_DIR/afl_same_setting_10seed_fig_full" \
  --title "MNIST IID Async Comparison (10 Seeds, FedAC Tuned)" \
  --group "AsyncSGD=$(printf "%s," "$EXP_DIR"/asyncsgd_iid_e60_npu_lr001_seed{1..10}.pkl | sed 's/,$//')" \
  --group "AsyncSAM RMS+AA=$(printf "%s," "$EXP_DIR"/asyncsam_precond_q3_rms90_i1_batch0_seed{1..10}.pkl | sed 's/,$//')" \
  --group "FedAsync=$(printf "%s," "$EXP_DIR"/afl_same_setting_fedasync_seed{1..10}.pkl | sed 's/,$//')" \
  --group "FedBuff=$(printf "%s," "$EXP_DIR"/afl_same_setting_fedbuff_seed{1..10}.pkl | sed 's/,$//')" \
  --group "AsyncAA=$(printf "%s," "$EXP_DIR"/afl_same_setting_asyncaa_seed{1..10}.pkl | sed 's/,$//')" \
  --group "FedAC tuned=$(printf "%s," "$EXP_DIR"/afl_same_setting_fedac_tuned_seed{1..10}.pkl | sed 's/,$//')"
