#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
EXP_DIR="$ROOT/experiments"
PYTHON_BIN="/mnt/liuyx_data/miniconda/envs/a3/bin/python"
PLOT_SCRIPT="$ROOT/plot_asyncsam_lr_sweep.py"

wait_for_result() {
  local path="$1"
  while [ ! -s "$path" ]; do
    sleep 60
  done
}

plot_mnist() {
  local fedac_path="$EXP_DIR/afl_same_setting_fedac_seed1_rerun.pkl"
  wait_for_result "$fedac_path"

  "$PYTHON_BIN" "$PLOT_SCRIPT" \
    --output-dir "$EXP_DIR/afl_same_setting_compare_fig_fedac_rerun" \
    --title "MNIST Seed1 Comparison (Corrected FedAC)" \
    --run "AsyncSGD=$EXP_DIR/asyncsgd_iid_e60_npu_lr001_seed1.pkl" \
    --run "AsyncSAM+RMS=$EXP_DIR/asyncsam_precond_q3_rms90_i1_batch0_seed1.pkl" \
    --run "FedAsync=$EXP_DIR/afl_same_setting_fedasync_seed1.pkl" \
    --run "FedBuff=$EXP_DIR/afl_same_setting_fedbuff_seed1.pkl" \
    --run "AsyncAA=$EXP_DIR/afl_same_setting_asyncaa_seed1.pkl" \
    --run "FedAC=$fedac_path"
}

plot_cifar() {
  local fedac_path="$EXP_DIR/cifar10_seed10_fedac_fixaug_e100_rerun.pkl"
  wait_for_result "$fedac_path"

  "$PYTHON_BIN" "$PLOT_SCRIPT" \
    --output-dir "$EXP_DIR/cifar10_seed10_full_compare_fig_fedac_rerun" \
    --title "CIFAR-10 Seed10 Full Comparison (Corrected FedAC)" \
    --run "AsyncSGD=$EXP_DIR/cifar10_seed10_asyncsgd_fixaug_e100trim.pkl" \
    --run "FedAsync=$EXP_DIR/cifar10_seed10_fedasync_fixaug_e100.pkl" \
    --run "FedBuff=$EXP_DIR/cifar10_seed10_fedbuff_fixaug_e100.pkl" \
    --run "AsyncAA=$EXP_DIR/cifar10_seed10_asyncaa_fixaug_e100_fast.pkl" \
    --run "AsyncSAM+RMS=$EXP_DIR/cifar10_seed10_asyncsam_rms_fixaug_e100.pkl" \
    --run "FedAC=$fedac_path"
}

plot_mnist &
plot_cifar &
wait

