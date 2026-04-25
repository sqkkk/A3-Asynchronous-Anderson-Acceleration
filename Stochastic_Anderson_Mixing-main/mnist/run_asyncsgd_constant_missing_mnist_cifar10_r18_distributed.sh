#!/usr/bin/env bash
set -u -o pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
EXP_DIR="$ROOT/experiments"
NODE_TAG="${NODE_TAG:-$(hostname)}"
LOG_DIR="$EXP_DIR/asyncsgd_constant_missing_mnist_cifar10_r18_logs/$NODE_TAG"
LOCK_DIR="$EXP_DIR/asyncsgd_constant_main_comparisons_locks"
mkdir -p "$LOG_DIR" "$LOCK_DIR"
cd "$ROOT" || exit 1

# Supplemental no-downweight AsyncSGD runs omitted from the first distributed
# batch: CIFAR-10/ResNet18 legacy + validation protocols and MNIST protocols.
MAX_PARALLEL="${MAX_PARALLEL:-16}"
SLEEP_BETWEEN_LAUNCHES="${SLEEP_BETWEEN_LAUNCHES:-1}"
SEED_LIST="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
DEVICE_IDS_STR="${DEVICE_IDS:-0 1 2 3 4 5 6 7}"
read -r -a DEVICE_IDS_ARR <<< "$DEVICE_IDS_STR"
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

launch_helper() {
  local helper="$1"
  local family="$2"
  local setting="$3"
  local prefix="$4"
  local seed="$5"
  shift 5

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
    if bash "$helper" "$device" asyncsgd "$prefix" "$seed" "$@" "${NO_DOWNWEIGHT[@]}" --partial-dump-every-epoch 1; then
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

JOB_INDEX=0

for seed in $SEED_LIST; do
  # CIFAR-10 / ResNet18 legacy base IID.
  launch_helper ./run_cifar10_seed_list.sh cifar10_resnet18 base \
    cifar10_multiseed_asyncsgd_const_seed "$seed"

  # CIFAR-10 / ResNet18 validation settings.
  launch_helper ./run_cifar10_seed_list.sh cifar10_resnet18 labelsorted \
    cifar10_validate10_labelsorted_asyncsgd_const_seed "$seed" \
    --partition label_sorted

  launch_helper ./run_cifar10_seed_list.sh cifar10_resnet18 highdelay \
    cifar10_validate10_highdelay_asyncsgd_const_seed "$seed" \
    --delay-gap 0.5 \
    --delay-jitter 0.1

  launch_helper ./run_cifar10_seed_list.sh cifar10_resnet18 workers20_original \
    cifar10_validate10_workers20_asyncsgd_const_seed "$seed" \
    --num-workers 20

  launch_helper ./run_cifar10_seed_list.sh cifar10_resnet18 workers20_t3e200 \
    cifar10_workers20_t3e200_asyncsgd_const_seed "$seed" \
    --num-workers 20 \
    --batch-size 64 \
    --epochs 200 \
    --lr 0.01 \
    --delay-gap 0.0 \
    --delay-jitter 0.0

  # MNIST base and validation settings.
  launch_helper ./run_mnist_seed_list.sh mnist base \
    mnist_asyncsgd_const_seed "$seed"

  launch_helper ./run_mnist_seed_list.sh mnist labelsorted \
    validate10_labelsorted_asyncsgd_const_seed "$seed" \
    --partition label_sorted

  launch_helper ./run_mnist_seed_list.sh mnist dirichlet005 \
    validate10_dirichlet005_asyncsgd_const_seed "$seed" \
    --partition dirichlet \
    --dirichlet-alpha 0.05 \
    --dirichlet-min-size 10

  launch_helper ./run_mnist_seed_list.sh mnist highdelay \
    validate10_highdelay_asyncsgd_const_seed "$seed" \
    --delay-gap 1.0 \
    --delay-jitter 0.2

  launch_helper ./run_mnist_seed_list.sh mnist workers20 \
    validate10_workers20_asyncsgd_const_seed "$seed" \
    --num-workers 20
done

wait
echo "done"
