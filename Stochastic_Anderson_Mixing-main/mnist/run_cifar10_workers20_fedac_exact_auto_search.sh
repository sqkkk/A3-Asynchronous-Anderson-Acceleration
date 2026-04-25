#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
PYTHON_BIN="/mnt/liuyx_data/miniconda/envs/a3/bin/python"

cd "$ROOT"
"$PYTHON_BIN" auto_fedac_exact_search.py
