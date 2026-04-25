#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
EXP_DIR="$ROOT/experiments"
LOG_DIR="$EXP_DIR/cifar100_resnet18_e300_ra_re_ls_asyncsam_rms_missing_8to10_logs"
mkdir -p "$LOG_DIR"
cd "$ROOT"

# Retry only the AsyncSAM+RMS jobs that OOMed on crowded devices 0/4.
# Keep the same public protocol and output prefixes as the 160-run experiment.
MAX_PARALLEL="${MAX_PARALLEL:-12}"
SLEEP_BETWEEN_LAUNCHES="${SLEEP_BETWEEN_LAUNCHES:-3}"
DEVICES=(1 2 3 5 6 7)

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

SAM_ARGS=(
  "${COMMON[@]}"
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

DIR=(--partition dirichlet --dirichlet-alpha 0.05 --dirichlet-min-size 10)
HD=(--delay-gap 0.5 --delay-jitter 0.1)

wait_for_slot() {
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]]; do
    sleep 30
  done
}

launch_one() {
  local device="$1"
  local setting="$2"
  local seed="$3"
  shift 3

  local prefix="cifar100_resnet18_e300_ra_re_ls_${setting}_asyncsam_rms_seed"
  local out_file="$EXP_DIR/${prefix}${seed}.pkl"
  local log_file="$LOG_DIR/${setting}_asyncsam_rms_seed${seed}.log"

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

idx=0
for seed in 8 9 10; do
  for setting in base labelsorted dirichlet highdelay; do
    extra=()
    case "$setting" in
      labelsorted) extra=(--partition label_sorted) ;;
      dirichlet) extra=("${DIR[@]}") ;;
      highdelay) extra=("${HD[@]}") ;;
    esac
    device="${DEVICES[$((idx % ${#DEVICES[@]}))]}"
    idx=$((idx + 1))
    launch_one "$device" "$setting" "$seed" "${SAM_ARGS[@]}" "${extra[@]}"
  done
done

wait
echo "done"
