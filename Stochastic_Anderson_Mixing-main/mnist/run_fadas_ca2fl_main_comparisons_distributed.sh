#!/usr/bin/env bash
set -u -o pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
EXP_DIR="$ROOT/experiments"
NODE_TAG="${NODE_TAG:-$(hostname)}"
LOG_DIR="$EXP_DIR/fadas_ca2fl_main_comparisons_logs/$NODE_TAG"
LOCK_DIR="$EXP_DIR/fadas_ca2fl_main_comparisons_locks"
mkdir -p "$LOG_DIR" "$LOCK_DIR"
cd "$ROOT" || exit 1

# Add FADAS and CA2FL to the established comparison protocols. Public training
# parameters are inherited from the original scripts; only method-private knobs
# below differ by algorithm.
MAX_PARALLEL="${MAX_PARALLEL:-32}"
SLEEP_BETWEEN_LAUNCHES="${SLEEP_BETWEEN_LAUNCHES:-1}"
SEED_LIST="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
METHOD_LIST="${METHODS:-fadas ca2fl}"
DEVICE_IDS_STR="${DEVICE_IDS:-0 1 2 3 4 5 6 7}"
read -r -a DEVICE_IDS_ARR <<< "$DEVICE_IDS_STR"

DIR=(--partition dirichlet --dirichlet-alpha 0.05 --dirichlet-min-size 10)
HD=(--delay-gap 0.5 --delay-jitter 0.1)
FADAS_ARGS=(
  --fadas-m "${FADAS_M:-5}"
  --fadas-tau-c "${FADAS_TAU_C:-1}"
  --fadas-beta1 "${FADAS_BETA1:-0.9}"
  --fadas-beta2 "${FADAS_BETA2:-0.99}"
  --fadas-eps "${FADAS_EPS:-1e-8}"
  --fadas-eta "${FADAS_ETA:-0.001}"
  --fadas-use-vhat "${FADAS_USE_VHAT:-1}"
  --fadas-delay-adapt "${FADAS_DELAY_ADAPT:-1}"
)
CA2FL_ARGS=(
  --ca2fl-m "${CA2FL_M:-10}"
  --ca2fl-eta "${CA2FL_ETA:-0.01}"
)

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

method_args() {
  local method="$1"
  case "$method" in
    fadas) printf '%s\n' "${FADAS_ARGS[@]}" ;;
    ca2fl) printf '%s\n' "${CA2FL_ARGS[@]}" ;;
    *)
      echo "unknown method: $method" >&2
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

launch_helper() {
  local helper="$1"
  local family="$2"
  local setting="$3"
  local method="$4"
  local prefix="$5"
  local seed="$6"
  shift 6

  local out_file="$EXP_DIR/${prefix}${seed}.pkl"
  local key lock log_file device_idx device
  key="$(safe_key "$out_file")"
  lock="$LOCK_DIR/$key.lock"
  log_file="$LOG_DIR/${family}_${setting}_${method}_seed${seed}.log"
  mkdir -p "$(dirname "$out_file")"

  if ! claim_task "$out_file"; then
    return 0
  fi

  wait_for_slot
  device_idx=$((JOB_INDEX % ${#DEVICE_IDS_ARR[@]}))
  device="${DEVICE_IDS_ARR[$device_idx]}"
  JOB_INDEX=$((JOB_INDEX + 1))

  echo "[launch] node=$NODE_TAG device=$device family=$family setting=$setting method=$method seed=$seed"
  (
    if bash "$helper" "$device" "$method" "$prefix" "$seed" "$@" --partial-dump-every-epoch 1; then
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

launch_cifar_four_setting_family() {
  local family="$1"
  local prefix_template="$2"
  shift 2
  local common_args=("$@")

  for seed in $SEED_LIST; do
    for setting in base labelsorted dirichlet highdelay; do
      mapfile -t extra < <(setting_args "$setting")
      for method in $METHOD_LIST; do
        mapfile -t private < <(method_args "$method")
        local prefix
        printf -v prefix "$prefix_template" "$setting" "$method"
        launch_helper ./run_cifar10_seed_list.sh "$family" "$setting" "$method" "$prefix" "$seed" \
          "${common_args[@]}" "${extra[@]}" "${private[@]}"
      done
    done
  done
}

launch_cifar10_resnet18_suite() {
  local seed method prefix
  for seed in $SEED_LIST; do
    for method in $METHOD_LIST; do
      mapfile -t private < <(method_args "$method")

      prefix="cifar10_multiseed_${method}_seed"
      launch_helper ./run_cifar10_seed_list.sh cifar10_resnet18 base "$method" "$prefix" "$seed" \
        "${private[@]}"

      prefix="cifar10_validate10_labelsorted_${method}_seed"
      launch_helper ./run_cifar10_seed_list.sh cifar10_resnet18 labelsorted "$method" "$prefix" "$seed" \
        --partition label_sorted "${private[@]}"

      prefix="cifar10_validate10_highdelay_${method}_seed"
      launch_helper ./run_cifar10_seed_list.sh cifar10_resnet18 highdelay "$method" "$prefix" "$seed" \
        --delay-gap 0.5 --delay-jitter 0.1 "${private[@]}"

      prefix="cifar10_workers20_t3e200_${method}_seed"
      launch_helper ./run_cifar10_seed_list.sh cifar10_resnet18 workers20_t3e200 "$method" "$prefix" "$seed" \
        --num-workers 20 --batch-size 64 --epochs 200 --lr 0.01 --delay-gap 0.0 --delay-jitter 0.0 "${private[@]}"
    done
  done
}

launch_mnist_suite() {
  local seed method prefix
  for seed in $SEED_LIST; do
    for method in $METHOD_LIST; do
      mapfile -t private < <(method_args "$method")

      prefix="mnist_${method}_seed"
      launch_helper ./run_mnist_seed_list.sh mnist base "$method" "$prefix" "$seed" \
        "${private[@]}"

      prefix="validate10_labelsorted_${method}_seed"
      launch_helper ./run_mnist_seed_list.sh mnist labelsorted "$method" "$prefix" "$seed" \
        --partition label_sorted "${private[@]}"

      prefix="validate10_dirichlet005_${method}_seed"
      launch_helper ./run_mnist_seed_list.sh mnist dirichlet005 "$method" "$prefix" "$seed" \
        --partition dirichlet --dirichlet-alpha 0.05 --dirichlet-min-size 10 "${private[@]}"

      prefix="validate10_highdelay_${method}_seed"
      launch_helper ./run_mnist_seed_list.sh mnist highdelay "$method" "$prefix" "$seed" \
        --delay-gap 1.0 --delay-jitter 0.2 "${private[@]}"

      prefix="validate10_workers20_${method}_seed"
      launch_helper ./run_mnist_seed_list.sh mnist workers20 "$method" "$prefix" "$seed" \
        --num-workers 20 "${private[@]}"
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

launch_mnist_suite
launch_cifar10_resnet18_suite

launch_cifar_four_setting_family \
  cifar10_resnet32 \
  'cifar10_resnet32_p3_%s_%s_seed' \
  "${CIFAR10_RESNET32_COMMON[@]}"

launch_cifar_four_setting_family \
  cifar10_resnet56 \
  'cifar10_resnet56_p3_%s_%s_seed' \
  "${CIFAR10_RESNET56_COMMON[@]}"

launch_cifar_four_setting_family \
  cifar100_resnet18_e300_ra_re_ls \
  'cifar100_resnet18_e300_ra_re_ls_%s_%s_seed' \
  "${CIFAR100_RESNET18_E300_COMMON[@]}"

launch_cifar_four_setting_family \
  cifar100_resnext29_8x16_lr003_e200 \
  'by_family/cifar100/resnext29/resnext29_8x16_lr003_e200_%s_%s_seed' \
  "${CIFAR100_RESNEXT29_E200_COMMON[@]}"

wait
echo "done"
