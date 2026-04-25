#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist"
EXP_DIR="$ROOT/experiments"
LOG_DIR="$EXP_DIR/mnist_sam_ablation_10seed_logs"
PYTHON_BIN="/mnt/liuyx_data/miniconda/envs/a3/bin/python"
OUT_DIR="$EXP_DIR/mnist_sam_ablation_10seed_fig"

mkdir -p "$LOG_DIR"
cd "$ROOT"

# Reuse the validated seed-1 reference run so the 10-seed suite matches the
# existing single-seed no-RMS AsyncSAM row.
cp -f "$EXP_DIR/asyncsam_tune_v2_batch0_seed1.pkl" "$EXP_DIR/ablate_asyncsam_seed1.pkl"

(
  ./run_mnist_seed_list.sh 0 asyncsgd ablate_asyncsgd_rms_seed 2,6,10 \
    --sam-precond rms \
    --sam-precond-beta 0.90 \
    --sam-precond-init 1.0
) > "$LOG_DIR/asyncsgd_rms_a.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 1 asyncsgd ablate_asyncsgd_rms_seed 3,7 \
    --sam-precond rms \
    --sam-precond-beta 0.90 \
    --sam-precond-init 1.0
) > "$LOG_DIR/asyncsgd_rms_b.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 2 asyncsgd ablate_asyncsgd_rms_seed 4,8 \
    --sam-precond rms \
    --sam-precond-beta 0.90 \
    --sam-precond-init 1.0
) > "$LOG_DIR/asyncsgd_rms_c.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 3 asyncsgd ablate_asyncsgd_rms_seed 5,9 \
    --sam-precond rms \
    --sam-precond-beta 0.90 \
    --sam-precond-init 1.0
) > "$LOG_DIR/asyncsgd_rms_d.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 4 asyncsam ablate_asyncsam_seed 2,6,10 \
    --sam-batch-accept \
    --sam-batch-tol 0.0
) > "$LOG_DIR/asyncsam_a.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 5 asyncsam ablate_asyncsam_seed 3,7 \
    --sam-batch-accept \
    --sam-batch-tol 0.0
) > "$LOG_DIR/asyncsam_b.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 6 asyncsam ablate_asyncsam_seed 4,8 \
    --sam-batch-accept \
    --sam-batch-tol 0.0
) > "$LOG_DIR/asyncsam_c.log" 2>&1 &

(
  ./run_mnist_seed_list.sh 7 asyncsam ablate_asyncsam_seed 5,9 \
    --sam-batch-accept \
    --sam-batch-tol 0.0
) > "$LOG_DIR/asyncsam_d.log" 2>&1 &

wait

"$PYTHON_BIN" "$ROOT/plot_async_multiseed_compare.py" \
  --output-dir "$OUT_DIR" \
  --title "MNIST SAM/RMS Ablation (10 Seeds)" \
  --acc-ymax 100 \
  --loss-ymax 7 \
  --group "AsyncSGD=$(printf "%s," "$EXP_DIR"/asyncsgd_iid_e60_npu_lr001_seed{1..10}.pkl | sed 's/,$//')" \
  --group "AsyncSGD+RMS=$(printf "%s," "$EXP_DIR"/ablate_asyncsgd_rms_seed{1..10}.pkl | sed 's/,$//')" \
  --group "AsyncSAM=$(printf "%s," "$EXP_DIR"/ablate_asyncsam_seed{1..10}.pkl | sed 's/,$//')" \
  --group "AsyncSAM RMS+AA=$(printf "%s," "$EXP_DIR"/asyncsam_precond_q3_rms90_i1_batch0_seed{1..10}.pkl | sed 's/,$//')"

"$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path

summary_path = Path("/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist/experiments/mnist_sam_ablation_10seed_fig/summary.csv")
rows = list(csv.DictReader(summary_path.open()))
mean_table = summary_path.with_name("mean_table.csv")
with mean_table.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Method", "Final Acc", "Best Acc", "Final Loss", "Best Loss"])
    for row in rows:
        writer.writerow([
            row["label"],
            row["final_acc_mean"],
            row["best_acc_mean"],
            row["final_loss_mean"],
            row["best_loss_mean"],
        ])
PY
