#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/mnist_10seed_logs"

mkdir -p "$LOG_DIR"

launch() {
  local device_id="$1"
  local job_name="$2"
  shift 2
  nohup "$ROOT/run_mnist_seed_series.sh" "$device_id" "$@" >"$LOG_DIR/${job_name}.log" 2>&1 &
  echo "${job_name} pid=$!"
}

launch 3 asyncsam_rms asyncsam asyncsam_precond_q3_rms90_i1_batch0_seed \
  --sam-precond rms \
  --sam-precond-beta 0.90 \
  --sam-precond-init 1.0 \
  --sam-batch-accept \
  --sam-batch-tol 0.0

launch 4 asyncsgd asyncsgd asyncsgd_iid_e60_npu_lr001_seed

launch 5 fedasync fedasync afl_same_setting_fedasync_seed \
  --fedasync-decay 1.0

launch 6 fedbuff fedbuff afl_same_setting_fedbuff_seed \
  --fedbuff-k 3 \
  --fedbuff-etag 5.0

launch 7 asyncaa asyncaa afl_same_setting_asyncaa_seed
