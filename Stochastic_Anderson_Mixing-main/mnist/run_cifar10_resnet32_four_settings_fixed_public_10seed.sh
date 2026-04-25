#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/cifar10_resnet32_four_settings_fixed_public_10seed_logs"

mkdir -p "$LOG_DIR"
cd "$ROOT"

SEEDS="1,2,3,4,5,6,7,8,9,10"
JOB_INDEX=0

COMMON_ARGS=(
  --model resnet32
  --batch-size 64
  --epochs 100
  --lr 0.05
)

ASYNC_SAM_RMS_ARGS=(
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

FEDASYNC_ARGS=(
  --fedasync-decay 1.0
)

FEDBUFF_ARGS=(
  --fedbuff-k 3
  --fedbuff-etag 5.0
)

FEDAC_ARGS=(
  --fedac-client-mode legacy
  --fedac-lr-mode shared
  --fedac-persist-optimizer 0
  --fedac-buffer-size 5
  --fedac-eta-g 0.0003
  --fedac-beta1 0.6
  --fedac-beta2 0.9
  --cv-server-lr 0.5
  --cv-momentum 1.0
  --grad-clip-norm 0.0
)

launch_rr() {
  local log_name="$1"
  local alg="$2"
  local prefix="$3"
  shift 3
  local device_id=$((JOB_INDEX % 8))
  JOB_INDEX=$((JOB_INDEX + 1))
  (
    bash ./run_cifar10_seed_list.sh "$device_id" "$alg" "$prefix" "$SEEDS" "${COMMON_ARGS[@]}" "$@"
  ) > "$LOG_DIR/$log_name" 2>&1 &
}

# 1. Base IID
launch_rr base_asyncsam_rms.log asyncsam cifar10_resnet32_p3_base_asyncsam_rms_seed \
  "${ASYNC_SAM_RMS_ARGS[@]}"
launch_rr base_asyncsgd.log asyncsgd cifar10_resnet32_p3_base_asyncsgd_seed
launch_rr base_fedasync.log fedasync cifar10_resnet32_p3_base_fedasync_seed \
  "${FEDASYNC_ARGS[@]}"
launch_rr base_fedbuff.log fedbuff cifar10_resnet32_p3_base_fedbuff_seed \
  "${FEDBUFF_ARGS[@]}"
launch_rr base_fedac.log fedac cifar10_resnet32_p3_base_fedac_seed \
  "${FEDAC_ARGS[@]}"

# 2. Label-sorted non-IID
launch_rr labelsorted_asyncsam_rms.log asyncsam cifar10_resnet32_p3_labelsorted_asyncsam_rms_seed \
  --partition label_sorted \
  "${ASYNC_SAM_RMS_ARGS[@]}"
launch_rr labelsorted_asyncsgd.log asyncsgd cifar10_resnet32_p3_labelsorted_asyncsgd_seed \
  --partition label_sorted
launch_rr labelsorted_fedasync.log fedasync cifar10_resnet32_p3_labelsorted_fedasync_seed \
  --partition label_sorted \
  "${FEDASYNC_ARGS[@]}"
launch_rr labelsorted_fedbuff.log fedbuff cifar10_resnet32_p3_labelsorted_fedbuff_seed \
  --partition label_sorted \
  "${FEDBUFF_ARGS[@]}"
launch_rr labelsorted_fedac.log fedac cifar10_resnet32_p3_labelsorted_fedac_seed \
  --partition label_sorted \
  "${FEDAC_ARGS[@]}"

# 3. Dirichlet non-IID
launch_rr dirichlet_asyncsam_rms.log asyncsam cifar10_resnet32_p3_dirichlet_asyncsam_rms_seed \
  --partition dirichlet \
  --dirichlet-alpha 0.05 \
  --dirichlet-min-size 10 \
  "${ASYNC_SAM_RMS_ARGS[@]}"
launch_rr dirichlet_asyncsgd.log asyncsgd cifar10_resnet32_p3_dirichlet_asyncsgd_seed \
  --partition dirichlet \
  --dirichlet-alpha 0.05 \
  --dirichlet-min-size 10
launch_rr dirichlet_fedasync.log fedasync cifar10_resnet32_p3_dirichlet_fedasync_seed \
  --partition dirichlet \
  --dirichlet-alpha 0.05 \
  --dirichlet-min-size 10 \
  "${FEDASYNC_ARGS[@]}"
launch_rr dirichlet_fedbuff.log fedbuff cifar10_resnet32_p3_dirichlet_fedbuff_seed \
  --partition dirichlet \
  --dirichlet-alpha 0.05 \
  --dirichlet-min-size 10 \
  "${FEDBUFF_ARGS[@]}"
launch_rr dirichlet_fedac.log fedac cifar10_resnet32_p3_dirichlet_fedac_seed \
  --partition dirichlet \
  --dirichlet-alpha 0.05 \
  --dirichlet-min-size 10 \
  "${FEDAC_ARGS[@]}"

# 4. High-delay IID
launch_rr highdelay_asyncsam_rms.log asyncsam cifar10_resnet32_p3_highdelay_asyncsam_rms_seed \
  --delay-gap 0.5 \
  --delay-jitter 0.1 \
  "${ASYNC_SAM_RMS_ARGS[@]}"
launch_rr highdelay_asyncsgd.log asyncsgd cifar10_resnet32_p3_highdelay_asyncsgd_seed \
  --delay-gap 0.5 \
  --delay-jitter 0.1
launch_rr highdelay_fedasync.log fedasync cifar10_resnet32_p3_highdelay_fedasync_seed \
  --delay-gap 0.5 \
  --delay-jitter 0.1 \
  "${FEDASYNC_ARGS[@]}"
launch_rr highdelay_fedbuff.log fedbuff cifar10_resnet32_p3_highdelay_fedbuff_seed \
  --delay-gap 0.5 \
  --delay-jitter 0.1 \
  "${FEDBUFF_ARGS[@]}"
launch_rr highdelay_fedac.log fedac cifar10_resnet32_p3_highdelay_fedac_seed \
  --delay-gap 0.5 \
  --delay-jitter 0.1 \
  "${FEDAC_ARGS[@]}"

wait
echo "done"
