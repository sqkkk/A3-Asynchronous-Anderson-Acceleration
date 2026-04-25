#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
OUT_DIR="$ROOT/experiments/by_family/cifar100/resnext29/base_iid_e300_seed1_4alg"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

source /mnt/liuyx_data/miniconda/bin/activate a3
cd "$ROOT"

COMMON_ARGS=(
  --dataset cifar100
  --model resnext29_8x64d
  --train-part-size 50000
  --test-part-size 10000
  --batch-size 64
  --test-batch-size 1000
  --epochs 300
  --lr 0.0015
  --lr-schedule cosine
  --lr-min-ratio 0.01
  --weight-decay 5e-4
  --cifar-augment randaugment
  --random-erasing 0.25
  --label-smoothing 0.1
  --log-interval 200
  --precision 0
)

ASYNCSAM_ARGS=(
  --sam-momentum 0.9
  --sam-precond rms
  --sam-precond-beta 0.90
  --sam-precond-init 1.0
  --sam-precond-min-denom 0.30
  --sam-hist-length 4
  --sam-period 2
  --sam-base-mix 0.8
  --sam-max-history-staleness 2
  --sam-max-cond 10000
  --sam-max-step-ratio 1.0
  --sam-anchor-tol 1.0
  --grad-clip-norm 0.0
)

FEDASYNC_ARGS=(
  --fedasync-decay 1.0
)

FEDBUFF_ARGS=(
  --fedbuff-k 3
  --fedbuff-etag 5.0
)

launch_one() {
  local device="$1"
  local method="$2"
  local alg="$3"
  shift 3
  local prefix="by_family/cifar100/resnext29/base_iid_e300_seed1_4alg/${method}_seed"
  local log_file="$LOG_DIR/${method}_seed1.log"
  echo "[$(date '+%F %T')] launch method=$method alg=$alg device=$device"
  (
    bash ./run_cifar10_seed_list.sh \
      "$device" \
      "$alg" \
      "$prefix" \
      "1" \
      "${COMMON_ARGS[@]}" \
      "$@"
  ) >"$log_file" 2>&1 &
}

launch_one 0 asyncsam_rms asyncsam "${ASYNCSAM_ARGS[@]}"
launch_one 1 asyncsgd asyncsgd
launch_one 2 fedasync fedasync "${FEDASYNC_ARGS[@]}"
launch_one 3 fedbuff fedbuff "${FEDBUFF_ARGS[@]}"

wait
echo "[$(date '+%F %T')] all runs finished"
