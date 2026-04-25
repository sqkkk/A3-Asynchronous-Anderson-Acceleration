#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
BASE_DIR="$ROOT/experiments/by_family/cifar100/resnext29"
LOG_FILE="$BASE_DIR/resnext29_8x16_lr003_e200_tuned_baselines_watch.log"
PYTHON_BIN="/mnt/liuyx_data/miniconda/envs/a3/bin/python"
EXPECTED=80

cd "$ROOT"

while true; do
  read -r finals partials <<< "$(
    "$PYTHON_BIN" - <<'PY'
from pathlib import Path

base = Path("/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist/experiments/by_family/cifar100/resnext29")
paths = list(base.glob("resnext29_8x16_lr003_e200_*_fed*_tuned_seed*.pkl"))
finals = [p for p in paths if not str(p).endswith(".partial.pkl") and p.stat().st_size > 0]
partials = list(base.glob("resnext29_8x16_lr003_e200_*_fed*_tuned_seed*.pkl.partial.pkl"))
print(len(finals), len(partials))
PY
  )"

  echo "[$(date -Iseconds)] finals=${finals}/${EXPECTED} partials=${partials}" | tee -a "$LOG_FILE"

  if [[ "$finals" -ge "$EXPECTED" ]]; then
    echo "[$(date -Iseconds)] all tuned baseline runs completed; regenerating plots" | tee -a "$LOG_FILE"
    "$PYTHON_BIN" plot_cifar100_resnext29_base_iid_compare.py 2>&1 | tee -a "$LOG_FILE"
    "$PYTHON_BIN" plot_cifar100_resnext29_8x16_lr003_e200_four_settings_10seed.py 2>&1 | tee -a "$LOG_FILE"
    echo "[$(date -Iseconds)] done" | tee -a "$LOG_FILE"
    break
  fi

  sleep 300
done
