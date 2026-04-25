#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
LOG_DIR="$ROOT/experiments/cifar10_resnet32_five_settings_10seed_logs"

mkdir -p "$LOG_DIR"
cd "$ROOT"

SEEDS="1,2,3,4,5,6,7,8,9,10"
JOB_INDEX=0

MODEL_ARGS=(
  --model resnet32
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

FEDAC_BASE_ARGS=(
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

FEDAC_WORKERS20_ARGS=(
  --fedac-client-mode legacy
  --fedac-lr-mode shared
  --fedac-persist-optimizer 0
  --fedac-buffer-size 3
  --fedac-eta-g 0.0001
  --fedac-beta1 0.9
  --fedac-beta2 0.99
  --cv-server-lr 0.1
  --cv-momentum 0.25
  --cv-global-clip-norm 10.0
  --grad-clip-norm 0.0
)

BASE_IID_ARGS=(
  "${MODEL_ARGS[@]}"
)

LABEL_SORTED_ARGS=(
  "${MODEL_ARGS[@]}"
  --partition label_sorted
)

DIRICHLET_ARGS=(
  "${MODEL_ARGS[@]}"
  --partition dirichlet
  --dirichlet-alpha 0.05
  --dirichlet-min-size 10
)

HIGH_DELAY_ARGS=(
  "${MODEL_ARGS[@]}"
  --delay-gap 0.5
  --delay-jitter 0.1
)

WORKERS20_ARGS=(
  "${MODEL_ARGS[@]}"
  --num-workers 20
  --batch-size 64
  --epochs 200
  --lr 0.01
  --delay-gap 0.0
  --delay-jitter 0.0
)

launch_rr() {
  local log_name="$1"
  local alg="$2"
  local prefix="$3"
  shift 3
  local device_id=$((JOB_INDEX % 8))
  JOB_INDEX=$((JOB_INDEX + 1))
  (
    bash ./run_cifar10_seed_list.sh "$device_id" "$alg" "$prefix" "$SEEDS" "$@"
  ) > "$LOG_DIR/$log_name" 2>&1 &
}

# 1. Base IID
launch_rr base_asyncsam_rms.log asyncsam cifar10_resnet32_base_asyncsam_rms_seed \
  "${BASE_IID_ARGS[@]}" "${ASYNC_SAM_RMS_ARGS[@]}"
launch_rr base_asyncsgd.log asyncsgd cifar10_resnet32_base_asyncsgd_seed \
  "${BASE_IID_ARGS[@]}"
launch_rr base_fedasync.log fedasync cifar10_resnet32_base_fedasync_seed \
  "${BASE_IID_ARGS[@]}" "${FEDASYNC_ARGS[@]}"
launch_rr base_fedbuff.log fedbuff cifar10_resnet32_base_fedbuff_seed \
  "${BASE_IID_ARGS[@]}" "${FEDBUFF_ARGS[@]}"
launch_rr base_fedac.log fedac cifar10_resnet32_base_fedac_seed \
  "${BASE_IID_ARGS[@]}" "${FEDAC_BASE_ARGS[@]}"

# 2. Label-sorted non-IID
launch_rr labelsorted_asyncsam_rms.log asyncsam cifar10_resnet32_labelsorted_asyncsam_rms_seed \
  "${LABEL_SORTED_ARGS[@]}" "${ASYNC_SAM_RMS_ARGS[@]}"
launch_rr labelsorted_asyncsgd.log asyncsgd cifar10_resnet32_labelsorted_asyncsgd_seed \
  "${LABEL_SORTED_ARGS[@]}"
launch_rr labelsorted_fedasync.log fedasync cifar10_resnet32_labelsorted_fedasync_seed \
  "${LABEL_SORTED_ARGS[@]}" "${FEDASYNC_ARGS[@]}"
launch_rr labelsorted_fedbuff.log fedbuff cifar10_resnet32_labelsorted_fedbuff_seed \
  "${LABEL_SORTED_ARGS[@]}" "${FEDBUFF_ARGS[@]}"
launch_rr labelsorted_fedac.log fedac cifar10_resnet32_labelsorted_fedac_seed \
  "${LABEL_SORTED_ARGS[@]}" "${FEDAC_BASE_ARGS[@]}"

# 3. Dirichlet non-IID
launch_rr dirichlet_asyncsam_rms.log asyncsam cifar10_resnet32_dirichlet_asyncsam_rms_seed \
  "${DIRICHLET_ARGS[@]}" "${ASYNC_SAM_RMS_ARGS[@]}"
launch_rr dirichlet_asyncsgd.log asyncsgd cifar10_resnet32_dirichlet_asyncsgd_seed \
  "${DIRICHLET_ARGS[@]}"
launch_rr dirichlet_fedasync.log fedasync cifar10_resnet32_dirichlet_fedasync_seed \
  "${DIRICHLET_ARGS[@]}" "${FEDASYNC_ARGS[@]}"
launch_rr dirichlet_fedbuff.log fedbuff cifar10_resnet32_dirichlet_fedbuff_seed \
  "${DIRICHLET_ARGS[@]}" "${FEDBUFF_ARGS[@]}"
launch_rr dirichlet_fedac.log fedac cifar10_resnet32_dirichlet_fedac_seed \
  "${DIRICHLET_ARGS[@]}" "${FEDAC_BASE_ARGS[@]}"

# 4. High-delay IID
launch_rr highdelay_asyncsam_rms.log asyncsam cifar10_resnet32_highdelay_asyncsam_rms_seed \
  "${HIGH_DELAY_ARGS[@]}" "${ASYNC_SAM_RMS_ARGS[@]}"
launch_rr highdelay_asyncsgd.log asyncsgd cifar10_resnet32_highdelay_asyncsgd_seed \
  "${HIGH_DELAY_ARGS[@]}"
launch_rr highdelay_fedasync.log fedasync cifar10_resnet32_highdelay_fedasync_seed \
  "${HIGH_DELAY_ARGS[@]}" "${FEDASYNC_ARGS[@]}"
launch_rr highdelay_fedbuff.log fedbuff cifar10_resnet32_highdelay_fedbuff_seed \
  "${HIGH_DELAY_ARGS[@]}" "${FEDBUFF_ARGS[@]}"
launch_rr highdelay_fedac.log fedac cifar10_resnet32_highdelay_fedac_seed \
  "${HIGH_DELAY_ARGS[@]}" "${FEDAC_BASE_ARGS[@]}"

# 5. IID 20 workers tuned protocol
launch_rr workers20_asyncsam_rms.log asyncsam cifar10_resnet32_workers20_asyncsam_rms_seed \
  "${WORKERS20_ARGS[@]}" --sam-stale-base-mix 0.2 "${ASYNC_SAM_RMS_ARGS[@]}"
launch_rr workers20_asyncsgd.log asyncsgd cifar10_resnet32_workers20_asyncsgd_seed \
  "${WORKERS20_ARGS[@]}"
launch_rr workers20_fedasync.log fedasync cifar10_resnet32_workers20_fedasync_seed \
  "${WORKERS20_ARGS[@]}" "${FEDASYNC_ARGS[@]}"
launch_rr workers20_fedbuff.log fedbuff cifar10_resnet32_workers20_fedbuff_seed \
  "${WORKERS20_ARGS[@]}" "${FEDBUFF_ARGS[@]}"
launch_rr workers20_fedac.log fedac cifar10_resnet32_workers20_fedac_seed \
  "${WORKERS20_ARGS[@]}" "${FEDAC_WORKERS20_ARGS[@]}"

wait
echo "done"
