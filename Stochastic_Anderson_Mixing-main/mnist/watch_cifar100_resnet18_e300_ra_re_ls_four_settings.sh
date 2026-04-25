#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
PYTHON_BIN="/mnt/liuyx_data/miniconda/envs/a3/bin/python"
EXP_DIR="$ROOT/experiments"
WATCH_DIR="$EXP_DIR/cifar100_resnet18_e300_ra_re_ls_four_settings_10seed_watch"
STATUS_LOG="$WATCH_DIR/status.log"
LATEST_STATUS="$WATCH_DIR/latest_status.txt"
PLOT_LOG="$WATCH_DIR/plot.log"
mkdir -p "$WATCH_DIR"
cd "$ROOT"

status_once() {
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import pickle
import statistics
import subprocess
import time

base = Path("/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist/experiments")
settings = ["base", "labelsorted", "dirichlet", "highdelay"]
methods = ["asyncsam_rms", "asyncsgd", "fedasync", "fedbuff"]

running_cmd = (
    "ps -ef | grep '/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/"
    "mnist/experiments/cifar100_resnet18_e300_ra_re_ls_' | grep -v grep | wc -l"
)
try:
    running = int(subprocess.check_output(running_cmd, shell=True, text=True).strip())
except Exception:
    running = -1

final = partial = 0
lines = [f"status_time {time.strftime('%F %T')} running={running}"]
for setting in settings:
    for method in methods:
        f = p = 0
        evals, accs, losses = [], [], []
        for seed in range(1, 11):
            stem = f"cifar100_resnet18_e300_ra_re_ls_{setting}_{method}_seed{seed}.pkl"
            out = base / stem
            pp = base / (stem + ".partial.pkl")
            target = None
            if out.exists() and out.stat().st_size > 0:
                target = out
                f += 1
            elif pp.exists() and pp.stat().st_size > 0:
                target = pp
                p += 1
            if target:
                try:
                    with target.open("rb") as fh:
                        data = pickle.load(fh)
                    acc = data.get("test_prec", [])
                    loss = data.get("test_loss", [])
                    if acc:
                        evals.append(len(acc))
                        accs.append(100 * float(acc[-1]))
                        if loss:
                            losses.append(float(loss[-1]))
                except Exception:
                    pass
        final += f
        partial += p
        if evals:
            lines.append(
                f"{setting:11s} {method:12s} {f}F/{p}P "
                f"eval={min(evals)}/{statistics.mean(evals):.1f}/{max(evals)} "
                f"acc={statistics.mean(accs):.2f} "
                f"loss={statistics.mean(losses):.4f}"
            )
        else:
            lines.append(f"{setting:11s} {method:12s} {f}F/{p}P")
lines.append(f"TOTAL final={final} partial={partial} seen={final + partial}/160")
print("\n".join(lines))
PY
}

while true; do
  status_once | tee "$LATEST_STATUS" >> "$STATUS_LOG"
  if grep -q "TOTAL final=160" "$LATEST_STATUS"; then
    {
      echo "[plot] $(date -Iseconds)"
      "$PYTHON_BIN" plot_cifar100_resnet18_e300_ra_re_ls_four_settings_10seed.py
      echo "[done] $(date -Iseconds)"
    } >> "$PLOT_LOG" 2>&1
    exit 0
  fi
  sleep 300
done
