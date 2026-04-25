#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
EXP_DIR="$ROOT/experiments"
PYTHON_BIN="/mnt/liuyx_data/miniconda/envs/a3/bin/python"
OUT_DIR="$EXP_DIR/afl_same_setting_10seed_fig_full"

need_file() {
  local path="$1"
  [ -f "$path" ]
}

all_ready() {
  local seed
  for seed in 1 2 3 4 5 6 7 8 9 10; do
    need_file "$EXP_DIR/asyncsgd_iid_e60_npu_lr001_seed${seed}.pkl" || return 1
    need_file "$EXP_DIR/asyncsam_precond_q3_rms90_i1_batch0_seed${seed}.pkl" || return 1
    need_file "$EXP_DIR/afl_same_setting_fedasync_seed${seed}.pkl" || return 1
    need_file "$EXP_DIR/afl_same_setting_fedbuff_seed${seed}.pkl" || return 1
    need_file "$EXP_DIR/afl_same_setting_asyncaa_seed${seed}.pkl" || return 1
    need_file "$EXP_DIR/afl_same_setting_fedac_tuned_seed${seed}.pkl" || return 1
  done
}

group_paths() {
  local prefix="$1"
  local items=()
  local seed
  for seed in 1 2 3 4 5 6 7 8 9 10; do
    items+=("$EXP_DIR/${prefix}${seed}.pkl")
  done
  local IFS=,
  echo "${items[*]}"
}

while ! all_ready; do
  sleep 30
done

"$PYTHON_BIN" "$ROOT/plot_async_multiseed_compare.py" \
  --output-dir "$OUT_DIR" \
  --title "MNIST IID Async Comparison (10 Seeds, FedAC Tuned)" \
  --group "AsyncSGD=$(group_paths asyncsgd_iid_e60_npu_lr001_seed)" \
  --group "AsyncSAM RMS+AA=$(group_paths asyncsam_precond_q3_rms90_i1_batch0_seed)" \
  --group "FedAsync=$(group_paths afl_same_setting_fedasync_seed)" \
  --group "FedBuff=$(group_paths afl_same_setting_fedbuff_seed)" \
  --group "AsyncAA=$(group_paths afl_same_setting_asyncaa_seed)" \
  --group "FedAC tuned=$(group_paths afl_same_setting_fedac_tuned_seed)"
