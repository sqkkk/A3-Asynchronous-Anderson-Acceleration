#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/liuyx_data/miniconda/envs/a3/bin/python}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/experiments/compare_seed1}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-100}"
TRAIN_PART_SIZE="${TRAIN_PART_SIZE:-12000}"
TEST_PART_SIZE="${TEST_PART_SIZE:-1000}"
BATCH_SIZE="${BATCH_SIZE:-6000}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1000}"
LR="${LR:-0.1}"

mkdir -p "${OUT_DIR}"

run_one() {
  local optimizer="$1"
  shift
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${PYTHON_BIN}" "${ROOT_DIR}/main.py" \
    --optimizer "${optimizer}" \
    --train-part-size "${TRAIN_PART_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --test-part-size "${TEST_PART_SIZE}" \
    --test-batch-size "${TEST_BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --log-interval 1000 \
    --no-cuda \
    --seed "${SEED}" \
    --dump-data "${OUT_DIR}/${optimizer}_seed${SEED}.pkl" \
    "$@"
}

run_one sgd
run_one adagrad
run_one rmsprop
run_one adasam
