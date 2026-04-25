#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
EXP_DIR="$ROOT/experiments"
LOG_DIR="$EXP_DIR/asyncsgd_constant_main_comparisons_10seed_logs"
mkdir -p "$LOG_DIR"
cd "$ROOT"

# This script adds a true no-downweight AsyncSGD baseline to the existing main
# comparison protocols. It keeps every public training parameter unchanged and
# only switches staleness weighting to constant, i.e. s(tau)=1.
MAX_PARALLEL="${MAX_PARALLEL:-32}"
SLEEP_BETWEEN_LAUNCHES="${SLEEP_BETWEEN_LAUNCHES:-1}"
SEED_LIST="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"

DIR=(--partition dirichlet --dirichlet-alpha 0.05 --dirichlet-min-size 10)
HD=(--delay-gap 0.5 --delay-jitter 0.1)
NO_DOWNWEIGHT=(--stale-strategy constant)

wait_for_slot() {
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]]; do
    sleep 30
  done
}

setting_args() {
  local setting="$1"
  case "$setting" in
    base) ;;
    labelsorted) printf '%s\n' --partition label_sorted ;;
    dirichlet) printf '%s\n' "${DIR[@]}" ;;
    highdelay) printf '%s\n' "${HD[@]}" ;;
    *)
      echo "unknown setting: $setting" >&2
      return 1
      ;;
  esac
}

launch_one() {
  local family="$1"
  local device="$2"
  local setting="$3"
  local seed="$4"
  local prefix="$5"
  shift 5

  local out_file="$EXP_DIR/${prefix}${seed}.pkl"
  local log_family="$LOG_DIR/$family"
  local log_file="$log_family/${setting}_asyncsgd_const_seed${seed}.log"
  mkdir -p "$(dirname "$out_file")" "$log_family"

  if [[ -s "$out_file" ]]; then
    echo "[skip-final] $out_file"
    return 0
  fi
  if pgrep -f "$out_file" >/dev/null 2>&1; then
    echo "[skip-running] $out_file"
    return 0
  fi

  wait_for_slot
  echo "[launch] family=$family device=$device setting=$setting method=asyncsgd_const seed=$seed"
  (
    bash ./run_cifar10_seed_list.sh "$device" asyncsgd "$prefix" "$seed" "$@" "${NO_DOWNWEIGHT[@]}" --partial-dump-every-epoch 1
  ) > "$log_file" 2>&1 &

  sleep "$SLEEP_BETWEEN_LAUNCHES"
}

run_family() {
  local family="$1"
  local prefix_template="$2"
  shift 2
  local common_args=("$@")

  for seed in $SEED_LIST; do
    for setting in base labelsorted dirichlet highdelay; do
      mapfile -t extra < <(setting_args "$setting")
      local prefix
      printf -v prefix "$prefix_template" "$setting"
      local device=$((JOB_INDEX % 8))
      JOB_INDEX=$((JOB_INDEX + 1))
      launch_one "$family" "$device" "$setting" "$seed" "$prefix" "${common_args[@]}" "${extra[@]}"
    done
  done
}

JOB_INDEX=0

CIFAR10_RESNET32_COMMON=(
  --model resnet32
  --batch-size 64
  --epochs 100
  --lr 0.05
)

CIFAR10_RESNET56_COMMON=(
  --model resnet56
  --batch-size 64
  --epochs 100
  --lr 0.05
)

CIFAR100_RESNET32_COMMON=(
  --dataset cifar100
  --model resnet32
  --train-part-size 50000
  --test-part-size 10000
  --batch-size 64
  --epochs 100
  --lr 0.05
)

CIFAR100_RESNET18_E300_COMMON=(
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

CIFAR100_RESNEXT29_E200_COMMON=(
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

run_family \
  cifar10_resnet32 \
  'cifar10_resnet32_p3_%s_asyncsgd_const_seed' \
  "${CIFAR10_RESNET32_COMMON[@]}"

run_family \
  cifar10_resnet56 \
  'cifar10_resnet56_p3_%s_asyncsgd_const_seed' \
  "${CIFAR10_RESNET56_COMMON[@]}"

run_family \
  cifar100_resnet32 \
  'cifar100_resnet32_p3_%s_asyncsgd_const_seed' \
  "${CIFAR100_RESNET32_COMMON[@]}"

run_family \
  cifar100_resnet18_e300_ra_re_ls \
  'cifar100_resnet18_e300_ra_re_ls_%s_asyncsgd_const_seed' \
  "${CIFAR100_RESNET18_E300_COMMON[@]}"

run_family \
  cifar100_resnext29_8x16_lr003_e200 \
  'by_family/cifar100/resnext29/resnext29_8x16_lr003_e200_%s_asyncsgd_const_seed' \
  "${CIFAR100_RESNEXT29_E200_COMMON[@]}"

wait
echo "done"
