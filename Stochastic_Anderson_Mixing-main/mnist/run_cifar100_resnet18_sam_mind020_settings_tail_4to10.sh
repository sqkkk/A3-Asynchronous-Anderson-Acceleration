#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
EXP_DIR="$ROOT/experiments"
LOG_DIR="$EXP_DIR/cifar100_resnet18_sam_mind020_settings_tail_4to10_logs"
mkdir -p "$LOG_DIR"
cd "$ROOT"

# Complete the promising AsyncSAM+RMS private-parameter variant
# sam_precond_min_denom=0.20 for the non-base CIFAR-100 settings.
# The shared protocol is unchanged from the main CIFAR-100 comparison.
DEVICES=(4 0 5 6 7 1 2 3)
MAX_PARALLEL="${MAX_PARALLEL:-21}"
SLEEP_BETWEEN_LAUNCHES="${SLEEP_BETWEEN_LAUNCHES:-2}"
LAUNCH_IDX=0

COMMON=(
  --dataset cifar100
  --model resnet18
  --train-part-size 50000
  --test-part-size 10000
  --batch-size 64
  --epochs 300
  --lr 0.05
  --cifar-augment randaugment
  --random-erasing 0.25
  --label-smoothing 0.1
)

SAM_MIND020=(
  --sam-momentum 0.9
  --sam-precond rms
  --sam-precond-beta 0.90
  --sam-precond-init 1.0
  --sam-precond-min-denom 0.20
  --sam-hist-length 4
  --sam-period 2
  --sam-base-mix 0.8
  --sam-max-history-staleness 2
  --sam-max-cond 10000
  --sam-max-step-ratio 1.0
  --sam-anchor-tol 1.0
  --grad-clip-norm 0.0
)

wait_for_slot() {
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]]; do
    sleep 30
  done
}

launch_one() {
  local setting="$1"
  local seed="$2"
  shift 2

  local device="${DEVICES[$((LAUNCH_IDX % ${#DEVICES[@]}))]}"
  LAUNCH_IDX=$((LAUNCH_IDX + 1))
  local prefix="cifar100_tune_r18_e300_${setting}_sam_mind020_asyncsam_rms_seed"
  local out_file="$EXP_DIR/${prefix}${seed}.pkl"
  local log_file="$LOG_DIR/${setting}_seed${seed}.log"

  if [[ -s "$out_file" ]]; then
    echo "[skip-final] $out_file"
    return 0
  fi
  if pgrep -f "$out_file" >/dev/null 2>&1; then
    echo "[skip-running] $out_file"
    return 0
  fi

  wait_for_slot
  echo "[launch] device=$device setting=$setting seed=$seed"
  (
    bash ./run_cifar10_seed_list.sh "$device" asyncsam "$prefix" "$seed" "$@"
  ) > "$log_file" 2>&1 &
  sleep "$SLEEP_BETWEEN_LAUNCHES"
}

for seed in 4 5 6 7 8 9 10; do
  launch_one labelsorted "$seed" "${COMMON[@]}" "${SAM_MIND020[@]}" --partition label_sorted
  launch_one dirichlet "$seed" "${COMMON[@]}" "${SAM_MIND020[@]}" --partition dirichlet --dirichlet-alpha 0.05 --dirichlet-min-size 10
  launch_one highdelay "$seed" "${COMMON[@]}" "${SAM_MIND020[@]}" --delay-gap 0.5 --delay-jitter 0.1
done

wait
echo "done"
