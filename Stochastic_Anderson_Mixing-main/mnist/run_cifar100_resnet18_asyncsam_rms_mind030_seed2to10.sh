#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
EXP_DIR="$ROOT/experiments"
LOG_DIR="$EXP_DIR/cifar100_resnet18_asyncsam_rms_mind030_seed2to10_logs"
mkdir -p "$LOG_DIR"
cd "$ROOT"

# Expand the strongest early Base-IID candidate around the RMS denominator
# floor. Only AsyncSAM+RMS-private parameters change; the shared protocol is
# identical to the main CIFAR-100 comparison.
DEVICES=(4 0)
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

SAM_MIND030=(
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

for seed in 2 3 4 5 6 7 8 9 10; do
  device="${DEVICES[$((LAUNCH_IDX % ${#DEVICES[@]}))]}"
  LAUNCH_IDX=$((LAUNCH_IDX + 1))
  prefix="cifar100_tune_r18_e300_sam_mind030_asyncsam_rms_seed"
  out_file="$EXP_DIR/${prefix}${seed}.pkl"
  log_file="$LOG_DIR/sam_mind030_seed${seed}.log"

  if [[ -s "$out_file" ]]; then
    echo "[skip-final] $out_file"
    continue
  fi
  if pgrep -f "$out_file" >/dev/null 2>&1; then
    echo "[skip-running] $out_file"
    continue
  fi

  echo "[launch] device=$device tag=sam_mind030 seed=$seed"
  (
    bash ./run_cifar10_seed_list.sh "$device" asyncsam "$prefix" "$seed" \
      "${COMMON[@]}" "${SAM_MIND030[@]}"
  ) > "$log_file" 2>&1 &
  sleep "$SLEEP_BETWEEN_LAUNCHES"
done

wait
echo "done"
