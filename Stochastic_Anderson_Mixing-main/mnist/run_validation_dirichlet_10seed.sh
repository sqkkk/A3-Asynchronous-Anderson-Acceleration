#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
EXP_DIR="$ROOT/experiments"
LOG_DIR="$EXP_DIR/validation_dirichlet_10seed_logs"
PYTHON_BIN="/mnt/liuyx_data/miniconda/envs/a3/bin/python"

mkdir -p "$LOG_DIR"
cd "$ROOT"

SEEDS="1,2,3,4,5,6,7,8,9,10"
DIRICHLET_ARGS=(
  --partition dirichlet
  --dirichlet-alpha 0.05
  --dirichlet-min-size 10
)

(
  ./run_mnist_seed_list.sh 2 asyncsam validate10_dirichlet005_asyncsam_rms_seed "$SEEDS" \
    "${DIRICHLET_ARGS[@]}" \
    --sam-precond rms \
    --sam-precond-beta 0.90 \
    --sam-precond-init 1.0 \
    --sam-batch-accept \
    --sam-batch-tol 0.0
) > "$LOG_DIR/dirichlet_asyncsam_rms.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 3 asyncsam validate10_dirichlet005_asyncsam_seed "$SEEDS" \
    "${DIRICHLET_ARGS[@]}" \
    --sam-batch-accept \
    --sam-batch-tol 0.0
) > "$LOG_DIR/dirichlet_asyncsam.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 4 asyncsgd validate10_dirichlet005_asyncsgd_rms_seed "$SEEDS" \
    "${DIRICHLET_ARGS[@]}" \
    --sam-precond rms \
    --sam-precond-beta 0.90 \
    --sam-precond-init 1.0
) > "$LOG_DIR/dirichlet_asyncsgd_rms.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 5 fedasync validate10_dirichlet005_fedasync_seed "$SEEDS" \
    "${DIRICHLET_ARGS[@]}" \
    --fedasync-decay 1.0
) > "$LOG_DIR/dirichlet_fedasync.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 6 fedbuff validate10_dirichlet005_fedbuff_seed "$SEEDS" \
    "${DIRICHLET_ARGS[@]}" \
    --fedbuff-k 3 \
    --fedbuff-etag 5.0
) > "$LOG_DIR/dirichlet_fedbuff.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 7 fedac validate10_dirichlet005_fedac_tuned_seed "$SEEDS" \
    "${DIRICHLET_ARGS[@]}" \
    --fedac-buffer-size 5 \
    --fedac-eta-g 0.0003 \
    --fedac-beta1 0.6 \
    --fedac-beta2 0.9 \
    --cv-server-lr 0.5 \
    --cv-momentum 1.0
) > "$LOG_DIR/dirichlet_fedac.log" 2>&1 &

wait
"$PYTHON_BIN" "$ROOT/plot_validation_dirichlet_10seed.py"
