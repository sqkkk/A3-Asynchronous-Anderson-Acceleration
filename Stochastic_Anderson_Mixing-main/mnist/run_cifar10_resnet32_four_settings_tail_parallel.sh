#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/cifar10_resnet32_four_settings_fixed_public_10seed_logs"
mkdir -p "$LOG_DIR"
cd "$ROOT"

SAM_ARGS=(
  --model resnet32
  --batch-size 64
  --epochs 100
  --lr 0.05
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
COMMON=(--model resnet32 --batch-size 64 --epochs 100 --lr 0.05)
FEDBUFF=(--fedbuff-k 3 --fedbuff-etag 5.0)
FEDASYNC=(--fedasync-decay 1.0)
FEDAC=(
  --fedac-client-mode legacy
  --fedac-buffer-size 3
  --fedac-eta-g 0.0001
  --fedac-beta1 0.9
  --fedac-beta2 0.99
  --cv-server-lr 0.1
  --cv-momentum 0.25
  --cv-global-clip-norm 10.0
)

launch_tail() {
  local device="$1"
  local log="$2"
  local alg="$3"
  local prefix="$4"
  local seeds="$5"
  shift 5
  (
    bash ./run_cifar10_seed_list.sh "$device" "$alg" "$prefix" "$seeds" "$@"
  ) > "$LOG_DIR/$log" 2>&1 &
}

# Existing main launcher is handling seed2 now. Start additional tails from seed3 onward.
SEEDS_A="3,5,7,9"
SEEDS_B="4,6,8,10"

# Base IID
launch_tail 0 base_asyncsam_rms_tail_a.log asyncsam cifar10_resnet32_p3_base_asyncsam_rms_seed "$SEEDS_A" "${SAM_ARGS[@]}"
launch_tail 1 base_asyncsam_rms_tail_b.log asyncsam cifar10_resnet32_p3_base_asyncsam_rms_seed "$SEEDS_B" "${SAM_ARGS[@]}"
launch_tail 2 base_asyncsgd_tail_a.log asyncsgd cifar10_resnet32_p3_base_asyncsgd_seed "$SEEDS_A" "${COMMON[@]}"
launch_tail 3 base_asyncsgd_tail_b.log asyncsgd cifar10_resnet32_p3_base_asyncsgd_seed "$SEEDS_B" "${COMMON[@]}"
launch_tail 4 base_fedasync_tail_a.log fedasync cifar10_resnet32_p3_base_fedasync_seed "$SEEDS_A" "${COMMON[@]}" "${FEDASYNC[@]}"
launch_tail 5 base_fedasync_tail_b.log fedasync cifar10_resnet32_p3_base_fedasync_seed "$SEEDS_B" "${COMMON[@]}" "${FEDASYNC[@]}"
launch_tail 6 base_fedbuff_tail_a.log fedbuff cifar10_resnet32_p3_base_fedbuff_seed "$SEEDS_A" "${COMMON[@]}" "${FEDBUFF[@]}"
launch_tail 7 base_fedbuff_tail_b.log fedbuff cifar10_resnet32_p3_base_fedbuff_seed "$SEEDS_B" "${COMMON[@]}" "${FEDBUFF[@]}"

# Label-sorted non-IID
launch_tail 0 labelsorted_asyncsam_rms_tail_a.log asyncsam cifar10_resnet32_p3_labelsorted_asyncsam_rms_seed "$SEEDS_A" "${SAM_ARGS[@]}" --partition label_sorted
launch_tail 1 labelsorted_asyncsam_rms_tail_b.log asyncsam cifar10_resnet32_p3_labelsorted_asyncsam_rms_seed "$SEEDS_B" "${SAM_ARGS[@]}" --partition label_sorted
launch_tail 2 labelsorted_asyncsgd_tail_a.log asyncsgd cifar10_resnet32_p3_labelsorted_asyncsgd_seed "$SEEDS_A" "${COMMON[@]}" --partition label_sorted
launch_tail 3 labelsorted_asyncsgd_tail_b.log asyncsgd cifar10_resnet32_p3_labelsorted_asyncsgd_seed "$SEEDS_B" "${COMMON[@]}" --partition label_sorted
launch_tail 4 labelsorted_fedasync_tail_a.log fedasync cifar10_resnet32_p3_labelsorted_fedasync_seed "$SEEDS_A" "${COMMON[@]}" --partition label_sorted "${FEDASYNC[@]}"
launch_tail 5 labelsorted_fedasync_tail_b.log fedasync cifar10_resnet32_p3_labelsorted_fedasync_seed "$SEEDS_B" "${COMMON[@]}" --partition label_sorted "${FEDASYNC[@]}"
launch_tail 6 labelsorted_fedbuff_tail_a.log fedbuff cifar10_resnet32_p3_labelsorted_fedbuff_seed "$SEEDS_A" "${COMMON[@]}" --partition label_sorted "${FEDBUFF[@]}"
launch_tail 7 labelsorted_fedbuff_tail_b.log fedbuff cifar10_resnet32_p3_labelsorted_fedbuff_seed "$SEEDS_B" "${COMMON[@]}" --partition label_sorted "${FEDBUFF[@]}"

# Dirichlet non-IID
DIR=(--partition dirichlet --dirichlet-alpha 0.05 --dirichlet-min-size 10)
launch_tail 0 dirichlet_asyncsam_rms_tail_a.log asyncsam cifar10_resnet32_p3_dirichlet_asyncsam_rms_seed "$SEEDS_A" "${SAM_ARGS[@]}" "${DIR[@]}"
launch_tail 1 dirichlet_asyncsam_rms_tail_b.log asyncsam cifar10_resnet32_p3_dirichlet_asyncsam_rms_seed "$SEEDS_B" "${SAM_ARGS[@]}" "${DIR[@]}"
launch_tail 2 dirichlet_asyncsgd_tail_a.log asyncsgd cifar10_resnet32_p3_dirichlet_asyncsgd_seed "$SEEDS_A" "${COMMON[@]}" "${DIR[@]}"
launch_tail 3 dirichlet_asyncsgd_tail_b.log asyncsgd cifar10_resnet32_p3_dirichlet_asyncsgd_seed "$SEEDS_B" "${COMMON[@]}" "${DIR[@]}"
launch_tail 4 dirichlet_fedasync_tail_a.log fedasync cifar10_resnet32_p3_dirichlet_fedasync_seed "$SEEDS_A" "${COMMON[@]}" "${DIR[@]}" "${FEDASYNC[@]}"
launch_tail 5 dirichlet_fedasync_tail_b.log fedasync cifar10_resnet32_p3_dirichlet_fedasync_seed "$SEEDS_B" "${COMMON[@]}" "${DIR[@]}" "${FEDASYNC[@]}"
launch_tail 6 dirichlet_fedbuff_tail_a.log fedbuff cifar10_resnet32_p3_dirichlet_fedbuff_seed "$SEEDS_A" "${COMMON[@]}" "${DIR[@]}" "${FEDBUFF[@]}"
launch_tail 7 dirichlet_fedbuff_tail_b.log fedbuff cifar10_resnet32_p3_dirichlet_fedbuff_seed "$SEEDS_B" "${COMMON[@]}" "${DIR[@]}" "${FEDBUFF[@]}"

# High delay
HD=(--delay-gap 0.5 --delay-jitter 0.1)
launch_tail 0 highdelay_asyncsam_rms_tail_a.log asyncsam cifar10_resnet32_p3_highdelay_asyncsam_rms_seed "$SEEDS_A" "${SAM_ARGS[@]}" "${HD[@]}"
launch_tail 1 highdelay_asyncsam_rms_tail_b.log asyncsam cifar10_resnet32_p3_highdelay_asyncsam_rms_seed "$SEEDS_B" "${SAM_ARGS[@]}" "${HD[@]}"
launch_tail 2 highdelay_asyncsgd_tail_a.log asyncsgd cifar10_resnet32_p3_highdelay_asyncsgd_seed "$SEEDS_A" "${COMMON[@]}" "${HD[@]}"
launch_tail 3 highdelay_asyncsgd_tail_b.log asyncsgd cifar10_resnet32_p3_highdelay_asyncsgd_seed "$SEEDS_B" "${COMMON[@]}" "${HD[@]}"
launch_tail 4 highdelay_fedasync_tail_a.log fedasync cifar10_resnet32_p3_highdelay_fedasync_seed "$SEEDS_A" "${COMMON[@]}" "${HD[@]}" "${FEDASYNC[@]}"
launch_tail 5 highdelay_fedasync_tail_b.log fedasync cifar10_resnet32_p3_highdelay_fedasync_seed "$SEEDS_B" "${COMMON[@]}" "${HD[@]}" "${FEDASYNC[@]}"
launch_tail 6 highdelay_fedbuff_tail_a.log fedbuff cifar10_resnet32_p3_highdelay_fedbuff_seed "$SEEDS_A" "${COMMON[@]}" "${HD[@]}" "${FEDBUFF[@]}"
launch_tail 7 highdelay_fedbuff_tail_b.log fedbuff cifar10_resnet32_p3_highdelay_fedbuff_seed "$SEEDS_B" "${COMMON[@]}" "${HD[@]}" "${FEDBUFF[@]}"

wait
echo "done"
