#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
EXP_DIR="$ROOT/experiments"
LOG_DIR="$EXP_DIR/by_family/cifar100/resnext29/resnext29_8x16_lr003_e200_four_settings_10seed_logs"
mkdir -p "$LOG_DIR"
cd "$ROOT"

# ResNeXt29-8x16d uses about 11-12GB HBM per job on this machine. Keep the
# NPUs busy while leaving room for Ascend runtime overhead.
MAX_PARALLEL="${MAX_PARALLEL:-32}"
SLEEP_BETWEEN_LAUNCHES="${SLEEP_BETWEEN_LAUNCHES:-1}"

COMMON=(
  --dataset cifar100
  --model resnext29_8x16d
  --train-part-size 50000
  --test-part-size 10000
  --batch-size 64
  --epochs 200
  --lr 0.03
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

FEDASYNC=(--fedasync-decay 1.0)
FEDBUFF=(--fedbuff-k 3 --fedbuff-etag 5.0)
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
  local method="$3"
  local alg="$4"
  local seed="$5"
  shift 5

  local prefix="by_family/cifar100/resnext29/resnext29_8x16_lr003_e200_${setting}_${method}_seed"
  local out_file="$EXP_DIR/${prefix}${seed}.pkl"
  local log_file="$LOG_DIR/${setting}_${method}_seed${seed}.log"

  if [[ -s "$out_file" ]]; then
    echo "[skip-final] $out_file"
    return 0
  fi
  if pgrep -f "$out_file" >/dev/null 2>&1; then
    echo "[skip-running] $out_file"
    return 0
  fi

  wait_for_slot
  echo "[launch] device=$device setting=$setting method=$method seed=$seed"
  (
    bash ./run_cifar10_seed_list.sh "$device" "$alg" "$prefix" "$seed" "$@" --partial-dump-every-epoch 1
  ) > "$log_file" 2>&1 &

  # Avoid initializing many Ascend contexts at the exact same instant.
  sleep "$SLEEP_BETWEEN_LAUNCHES"
}

idx=0
SEED_LIST="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
for seed in $SEED_LIST; do
  for setting in base labelsorted dirichlet highdelay; do
    extra=()
    case "$setting" in
      labelsorted) extra=(--partition label_sorted) ;;
      dirichlet) extra=("${DIR[@]}") ;;
      highdelay) extra=("${HD[@]}") ;;
    esac

    device=$((idx % 8)); idx=$((idx + 1))
    launch_one "$device" "$setting" asyncsam_rms asyncsam "$seed" "${SAM_ARGS[@]}" "${extra[@]}"

    device=$((idx % 8)); idx=$((idx + 1))
    launch_one "$device" "$setting" asyncsgd asyncsgd "$seed" "${COMMON[@]}" "${extra[@]}"

    device=$((idx % 8)); idx=$((idx + 1))
    launch_one "$device" "$setting" fedasync fedasync "$seed" "${COMMON[@]}" "${extra[@]}" "${FEDASYNC[@]}"

    device=$((idx % 8)); idx=$((idx + 1))
    launch_one "$device" "$setting" fedbuff fedbuff "$seed" "${COMMON[@]}" "${extra[@]}" "${FEDBUFF[@]}"
  done
done

wait
echo "done"
