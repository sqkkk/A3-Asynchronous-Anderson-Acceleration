#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
EXP_DIR="$ROOT/experiments"
LOG_DIR="$EXP_DIR/cifar100_resnet18_asyncsam_rms_tune_pilot_logs"
mkdir -p "$LOG_DIR"
cd "$ROOT"

DEVICES=(1 2 3 5 6 7)
LAUNCH_IDX=0
SLEEP_BETWEEN_LAUNCHES="${SLEEP_BETWEEN_LAUNCHES:-2}"

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

SAM_BASE=(
  --sam-momentum 0.9
  --sam-precond rms
  --sam-precond-beta 0.90
  --sam-precond-init 1.0
  --sam-precond-min-denom 0.1
  --sam-hist-length 4
  --sam-period 2
  --sam-base-mix 0.8
  --sam-max-history-staleness 2
  --sam-max-cond 10000
  --sam-max-step-ratio 1.0
  --sam-anchor-tol 1.0
  --grad-clip-norm 0.0
)

launch_one() {
  local tag="$1"
  local seed="$2"
  shift 2

  local device="${DEVICES[$((LAUNCH_IDX % ${#DEVICES[@]}))]}"
  LAUNCH_IDX=$((LAUNCH_IDX + 1))
  local prefix="cifar100_tune_r18_e300_${tag}_asyncsam_rms_seed"
  local out_file="$EXP_DIR/${prefix}${seed}.pkl"
  local log_file="$LOG_DIR/${tag}_seed${seed}.log"

  if [[ -s "$out_file" ]]; then
    echo "[skip-final] $out_file"
    return 0
  fi
  if pgrep -f "$out_file" >/dev/null 2>&1; then
    echo "[skip-running] $out_file"
    return 0
  fi

  echo "[launch] device=$device tag=$tag seed=$seed"
  (
    bash ./run_cifar10_seed_list.sh "$device" asyncsam "$prefix" "$seed" "$@"
  ) > "$log_file" 2>&1 &
  sleep "$SLEEP_BETWEEN_LAUNCHES"
}

for seed in 2 3; do
  launch_one sam_stop085 "$seed" "${COMMON[@]}" "${SAM_BASE[@]}" --sam-stop-fraction 0.85
  launch_one sam_mom095 "$seed" "${COMMON[@]}" "${SAM_BASE[@]}" --sam-momentum 0.95
  launch_one sam_beta095 "$seed" "${COMMON[@]}" "${SAM_BASE[@]}" --sam-precond-beta 0.95
done

wait
echo "done"
