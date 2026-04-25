#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
EXP_DIR="$ROOT/experiments"
LOG_DIR="$EXP_DIR/cifar100_resnet18_asyncsam_rms_near_mind020_seed1_logs"
mkdir -p "$LOG_DIR"
cd "$ROOT"

# Local search around the currently strongest AsyncSAM+RMS candidate:
# sam_precond_min_denom=0.20. These runs only change AsyncSAM+RMS-private
# knobs, so the shared CIFAR-100 protocol remains comparable to existing
# AsyncSGD/FedAsync/FedBuff baselines.
DEVICES=(5 6 7 1 2 3)
MAX_PARALLEL="${MAX_PARALLEL:-12}"
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
  local tag="$1"
  shift

  local device="${DEVICES[$((LAUNCH_IDX % ${#DEVICES[@]}))]}"
  LAUNCH_IDX=$((LAUNCH_IDX + 1))
  local prefix="cifar100_tune_r18_e300_${tag}_asyncsam_rms_seed"
  local out_file="$EXP_DIR/${prefix}1.pkl"
  local log_file="$LOG_DIR/${tag}_seed1.log"

  if [[ -s "$out_file" ]]; then
    echo "[skip-final] $out_file"
    return 0
  fi
  if pgrep -f "$out_file" >/dev/null 2>&1; then
    echo "[skip-running] $out_file"
    return 0
  fi

  wait_for_slot
  echo "[launch] device=$device tag=$tag seed=1"
  (
    bash ./run_cifar10_seed_list.sh "$device" asyncsam "$prefix" 1 "$@"
  ) > "$log_file" 2>&1 &

  sleep "$SLEEP_BETWEEN_LAUNCHES"
}

LAUNCH_IDX=0

launch_one sam_mind025 "${COMMON[@]}" "${SAM_MIND020[@]}" --sam-precond-min-denom 0.25
launch_one sam_mind030 "${COMMON[@]}" "${SAM_MIND020[@]}" --sam-precond-min-denom 0.30
launch_one sam_mind020_beta095 "${COMMON[@]}" "${SAM_MIND020[@]}" --sam-precond-beta 0.95
launch_one sam_mind020_mix070 "${COMMON[@]}" "${SAM_MIND020[@]}" --sam-base-mix 0.70
launch_one sam_mind020_mix090 "${COMMON[@]}" "${SAM_MIND020[@]}" --sam-base-mix 0.90
launch_one sam_mind020_stale1 "${COMMON[@]}" "${SAM_MIND020[@]}" --sam-max-history-staleness 1
launch_one sam_mind020_stale0 "${COMMON[@]}" "${SAM_MIND020[@]}" --sam-max-history-staleness 0
launch_one sam_mind020_warmup2k "${COMMON[@]}" "${SAM_MIND020[@]}" --sam-aa-warmup-updates 2000
launch_one sam_mind020_ridge1e4 "${COMMON[@]}" "${SAM_MIND020[@]}" --sam-ridge 1e-4
launch_one sam_mind020_maxratio075 "${COMMON[@]}" "${SAM_MIND020[@]}" --sam-max-step-ratio 0.75

wait
echo "done"
