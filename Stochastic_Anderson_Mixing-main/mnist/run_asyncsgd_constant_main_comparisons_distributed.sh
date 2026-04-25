#!/usr/bin/env bash
set -u -o pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
EXP_DIR="$ROOT/experiments"
NODE_TAG="${NODE_TAG:-$(hostname)}"
LOG_DIR="$EXP_DIR/asyncsgd_constant_main_comparisons_distributed_logs/$NODE_TAG"
LOCK_DIR="$EXP_DIR/asyncsgd_constant_main_comparisons_locks"
mkdir -p "$LOG_DIR" "$LOCK_DIR"
cd "$ROOT" || exit 1

# Multi-node work stealing launcher for the no-downweight AsyncSGD baseline.
# All nodes iterate over the same task list and atomically claim tasks by mkdir.
# This keeps shared protocol parameters unchanged and only uses s(tau)=1.
MAX_PARALLEL="${MAX_PARALLEL:-32}"
SLEEP_BETWEEN_LAUNCHES="${SLEEP_BETWEEN_LAUNCHES:-1}"
SEED_LIST="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
DEVICE_IDS_STR="${DEVICE_IDS:-0 1 2 3 4 5 6 7}"
read -r -a DEVICE_IDS_ARR <<< "$DEVICE_IDS_STR"

DIR=(--partition dirichlet --dirichlet-alpha 0.05 --dirichlet-min-size 10)
HD=(--delay-gap 0.5 --delay-jitter 0.1)
NO_DOWNWEIGHT=(--stale-strategy constant)

safe_key() {
  local path="$1"
  path="${path#"$EXP_DIR"/}"
  echo "$path" | tr '/ ' '__'
}

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

claim_task() {
  local out_file="$1"
  local key lock
  key="$(safe_key "$out_file")"
  lock="$LOCK_DIR/$key.lock"

  if [[ -s "$out_file" ]]; then
    echo "[skip-final] $out_file"
    return 1
  fi
  if mkdir "$lock" 2>/dev/null; then
    {
      echo "node=$NODE_TAG"
      echo "pid=$$"
      echo "time=$(date -Iseconds)"
      echo "output=$out_file"
    } > "$lock/owner.txt"
    return 0
  fi

  echo "[skip-locked] $out_file"
  return 1
}

launch_one() {
  local family="$1"
  local setting="$2"
  local seed="$3"
  local prefix="$4"
  shift 4

  local out_file="$EXP_DIR/${prefix}${seed}.pkl"
  local key lock log_file device_idx device
  key="$(safe_key "$out_file")"
  lock="$LOCK_DIR/$key.lock"
  log_file="$LOG_DIR/${family}_${setting}_asyncsgd_const_seed${seed}.log"
  mkdir -p "$(dirname "$out_file")"

  if ! claim_task "$out_file"; then
    return 0
  fi

  wait_for_slot
  device_idx=$((JOB_INDEX % ${#DEVICE_IDS_ARR[@]}))
  device="${DEVICE_IDS_ARR[$device_idx]}"
  JOB_INDEX=$((JOB_INDEX + 1))

  echo "[launch] node=$NODE_TAG device=$device family=$family setting=$setting seed=$seed"
  (
    if bash ./run_cifar10_seed_list.sh "$device" asyncsgd "$prefix" "$seed" "$@" "${NO_DOWNWEIGHT[@]}" --partial-dump-every-epoch 1; then
      echo "[task-done] $(date -Iseconds) $out_file"
    else
      status=$?
      echo "[task-fail] $(date -Iseconds) status=$status $out_file"
      rm -rf "$lock"
      exit "$status"
    fi
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
      launch_one "$family" "$setting" "$seed" "$prefix" "${common_args[@]}" "${extra[@]}"
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
