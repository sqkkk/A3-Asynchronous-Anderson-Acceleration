from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist")
EXP_DIR = ROOT / "experiments"
OUT_DIR = EXP_DIR / "cifar10_workers20_fedac_exact_auto_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_BIN = "/mnt/liuyx_data/miniconda/envs/a3/bin/python"
RUN_SEED_LIST = ROOT / "run_cifar10_seed_list.sh"
SHORTSCREEN_SCRIPT = ROOT / "run_cifar10_workers20_fedac_shortscreen.sh"

COMMON_BASE_ARGS = [
    "--num-workers", "20",
    "--batch-size", "64",
    "--lr", "0.01",
    "--delay-gap", "0.0",
    "--delay-jitter", "0.0",
    "--partial-dump-every-epoch", "1",
]

STAGE1_CONFIGS = [
    ("s0_orig_default", ["--fedac-buffer-size", "5", "--fedac-eta-g", "0.001", "--fedac-beta1", "0.6", "--fedac-beta2", "0.9", "--fedac-local-epochs", "5"]),
    ("s1_orig_eta3e4", ["--fedac-buffer-size", "5", "--fedac-eta-g", "0.0003", "--fedac-beta1", "0.6", "--fedac-beta2", "0.9", "--fedac-local-epochs", "5"]),
    ("s2_orig_eta1e4", ["--fedac-buffer-size", "5", "--fedac-eta-g", "0.0001", "--fedac-beta1", "0.6", "--fedac-beta2", "0.9", "--fedac-local-epochs", "5"]),
    ("s3_buf3_eta3e4", ["--fedac-buffer-size", "3", "--fedac-eta-g", "0.0003", "--fedac-beta1", "0.6", "--fedac-beta2", "0.9", "--fedac-local-epochs", "5"]),
    ("s4_buf3_eta1e4", ["--fedac-buffer-size", "3", "--fedac-eta-g", "0.0001", "--fedac-beta1", "0.6", "--fedac-beta2", "0.9", "--fedac-local-epochs", "5"]),
    ("s5_smooth_buf3_eta3e4", ["--fedac-buffer-size", "3", "--fedac-eta-g", "0.0003", "--fedac-beta1", "0.9", "--fedac-beta2", "0.99", "--fedac-local-epochs", "5"]),
    ("s6_smooth_buf5_eta1e4", ["--fedac-buffer-size", "5", "--fedac-eta-g", "0.0001", "--fedac-beta1", "0.9", "--fedac-beta2", "0.99", "--fedac-local-epochs", "5"]),
    ("s7_orig_e3_eta3e4", ["--fedac-buffer-size", "5", "--fedac-eta-g", "0.0003", "--fedac-beta1", "0.6", "--fedac-beta2", "0.9", "--fedac-local-epochs", "3"]),
]


def parse_kv_args(arg_list: list[str]) -> dict[str, str]:
    parsed = {}
    idx = 0
    while idx < len(arg_list):
        key = arg_list[idx]
        if not key.startswith("--"):
            idx += 1
            continue
        if idx + 1 < len(arg_list) and not arg_list[idx + 1].startswith("--"):
            parsed[key] = arg_list[idx + 1]
            idx += 2
        else:
            parsed[key] = "1"
            idx += 1
    return parsed


def config_to_args(cfg: dict[str, str]) -> list[str]:
    out = []
    for key, value in cfg.items():
        out.extend([key, str(value)])
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_summary(meta_path: Path) -> dict:
    with meta_path.open("r", encoding="utf-8") as fh:
        obj = json.load(fh)
    summary = obj.get("summary", {})
    status = obj.get("status", {})
    args = obj.get("args", {})
    return {
        "final_acc": float(summary.get("final_acc", float("nan"))) * 100.0,
        "best_acc": float(summary.get("best_acc", float("nan"))) * 100.0,
        "final_loss": float(summary.get("final_loss", float("nan"))),
        "best_loss": float(summary.get("best_loss", float("nan"))),
        "completed_epochs": int(status.get("completed_epochs", 0) or 0),
        "stopped_early": bool(status.get("stopped_early", False)),
        "early_abort_reason": status.get("early_abort_reason"),
        "args": args,
    }


def stage1_output_meta(cfg_name: str) -> Path:
    return EXP_DIR / f"cifar10_workers20_fedac_exact_shortscreen_{cfg_name}_seed1.pkl.meta.json"


def stage1_output_partial_meta(cfg_name: str) -> Path:
    return EXP_DIR / f"cifar10_workers20_fedac_exact_shortscreen_{cfg_name}_seed1.pkl.partial.meta.json"


def stage2_output_meta(cfg_name: str, seed: int) -> Path:
    return EXP_DIR / f"cifar10_workers20_fedac_exact_stage2_{cfg_name}_seed{seed}.pkl.meta.json"


def stage1_active() -> bool:
    proc = subprocess.run(
        ["bash", "-lc", "ps -ef | rg 'async_distributed_main.py --alg fedac .*cifar10_workers20_fedac_exact_shortscreen'"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    return "async_distributed_main.py" in proc.stdout


def cfg_process_active(prefix: str) -> bool:
    proc = subprocess.run(
        ["bash", "-lc", f"ps -ef | rg '{prefix}' | rg 'async_distributed_main.py --alg fedac'"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    return "async_distributed_main.py" in proc.stdout


def best_available_meta(final_meta: Path, partial_meta: Path) -> tuple[Path | None, bool]:
    if final_meta.exists():
        return final_meta, False
    if partial_meta.exists():
        return partial_meta, True
    return None, False


def stage1_row_ready(cfg_name: str, summary: dict, is_partial: bool) -> bool:
    if not is_partial:
        return True
    if summary["stopped_early"]:
        return True
    if summary["completed_epochs"] >= 25:
        return True
    if summary["final_acc"] >= 35.0 or summary["best_acc"] >= 40.0:
        return True
    prefix = f"cifar10_workers20_fedac_exact_shortscreen_{cfg_name}_seed1.pkl"
    return not cfg_process_active(prefix)


def wait_for_stage1() -> list[dict]:
    expected = [stage1_output_meta(name) for name, _ in STAGE1_CONFIGS]
    if not all(path.exists() for path in expected):
        if not stage1_active():
            subprocess.run(["bash", str(SHORTSCREEN_SCRIPT)], cwd=str(ROOT), check=True)
        while True:
            rows = []
            ready = 0
            for name, _ in STAGE1_CONFIGS:
                final_meta = stage1_output_meta(name)
                partial_meta = stage1_output_partial_meta(name)
                meta, is_partial = best_available_meta(final_meta, partial_meta)
                if meta is None:
                    continue
                summary = load_summary(meta)
                row = {
                    "config": name,
                    "seed": 1,
                    "final_acc": summary["final_acc"],
                    "best_acc": summary["best_acc"],
                    "final_loss": summary["final_loss"],
                    "best_loss": summary["best_loss"],
                    "completed_epochs": summary["completed_epochs"],
                    "stopped_early": summary["stopped_early"],
                    "early_abort_reason": summary["early_abort_reason"],
                    "path": str(meta),
                    "is_partial": is_partial,
                }
                rows.append(row)
                if stage1_row_ready(name, summary, is_partial):
                    ready += 1
            if ready == len(STAGE1_CONFIGS):
                break
            time.sleep(20)
    else:
        rows = []
        for name, _ in STAGE1_CONFIGS:
            meta = stage1_output_meta(name)
            summary = load_summary(meta)
            rows.append(
                {
                    "config": name,
                    "seed": 1,
                    "final_acc": summary["final_acc"],
                    "best_acc": summary["best_acc"],
                    "final_loss": summary["final_loss"],
                    "best_loss": summary["best_loss"],
                    "completed_epochs": summary["completed_epochs"],
                    "stopped_early": summary["stopped_early"],
                    "early_abort_reason": summary["early_abort_reason"],
                    "path": str(meta),
                    "is_partial": False,
                }
            )
    rows.sort(key=lambda r: (-r["final_acc"], r["final_loss"], r["config"]))
    write_csv(
        OUT_DIR / "stage1_summary.csv",
        rows,
        ["config", "seed", "final_acc", "best_acc", "final_loss", "best_loss", "completed_epochs", "stopped_early", "early_abort_reason", "path", "is_partial"],
    )
    return rows


def make_bank2(best_row: dict, stage1_cfg_map: dict[str, dict[str, str]]) -> list[tuple[str, list[str]]]:
    base = dict(stage1_cfg_map[best_row["config"]])
    eta = float(base["--fedac-eta-g"])
    buf = int(base["--fedac-buffer-size"])
    local_epochs = int(base["--fedac-local-epochs"])
    beta1 = float(base["--fedac-beta1"])
    beta2 = float(base["--fedac-beta2"])
    alt_buf = 3 if buf != 3 else 5
    alt_epochs = 3 if local_epochs != 3 else 5
    bank = [
        ("b0_keep_best", config_to_args(base)),
        ("b1_eta_half", config_to_args({**base, "--fedac-eta-g": f"{eta * 0.5:.8f}"})),
        ("b2_eta_quarter", config_to_args({**base, "--fedac-eta-g": f"{eta * 0.25:.8f}"})),
        ("b3_eta_double", config_to_args({**base, "--fedac-eta-g": f"{eta * 2.0:.8f}"})),
        ("b4_alt_buffer", config_to_args({**base, "--fedac-buffer-size": str(alt_buf)})),
        ("b5_alt_epochs", config_to_args({**base, "--fedac-local-epochs": str(alt_epochs)})),
        ("b6_smooth_beta", config_to_args({**base, "--fedac-beta1": "0.9", "--fedac-beta2": "0.99"})),
        ("b7_orig_beta", config_to_args({**base, "--fedac-beta1": "0.6", "--fedac-beta2": "0.9"})),
    ]
    dedup = []
    seen = set()
    for name, args in bank:
        key = tuple(args)
        if key in seen:
            continue
        seen.add(key)
        dedup.append((name, args))
    return dedup


def launch_single_seed(device_id: int, prefix: str, seed: int, extra_args: list[str], epochs: int, abort_epoch: int, abort_min_acc: float) -> subprocess.Popen:
    cmd = [
        "bash",
        str(RUN_SEED_LIST),
        str(device_id),
        "fedac",
        prefix,
        str(seed),
        "--epochs",
        str(epochs),
        "--log-interval",
        "20",
        "--early-abort-epoch",
        str(abort_epoch),
        "--early-abort-min-acc",
        str(abort_min_acc),
        *COMMON_BASE_ARGS,
        *extra_args,
    ]
    log_path = OUT_DIR / f"{prefix}{seed}.log"
    log_fh = log_path.open("w")
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_fh, stderr=subprocess.STDOUT)


def run_bank(bank: list[tuple[str, list[str]]], prefix_base: str, epochs: int, abort_epoch: int, abort_min_acc: float) -> list[dict]:
    procs = []
    meta_paths = []
    for device_id, (name, extra_args) in enumerate(bank):
        prefix = f"{prefix_base}{name}_seed"
        meta_path = EXP_DIR / f"{prefix}{1}.pkl.meta.json"
        meta_paths.append((name, extra_args, meta_path))
        if meta_path.exists():
            continue
        procs.append(launch_single_seed(device_id, prefix, 1, extra_args, epochs, abort_epoch, abort_min_acc))
    for proc in procs:
        proc.wait()
    rows = []
    for name, extra_args, meta_path in meta_paths:
        while not meta_path.exists():
            time.sleep(5)
        summary = load_summary(meta_path)
        rows.append(
            {
                "config": name,
                "seed": 1,
                "final_acc": summary["final_acc"],
                "best_acc": summary["best_acc"],
                "final_loss": summary["final_loss"],
                "best_loss": summary["best_loss"],
                "completed_epochs": summary["completed_epochs"],
                "stopped_early": summary["stopped_early"],
                "early_abort_reason": summary["early_abort_reason"],
                "path": str(meta_path),
                "args": " ".join(extra_args),
            }
        )
    rows.sort(key=lambda r: (-r["final_acc"], r["final_loss"], r["config"]))
    return rows


def run_stage2(top_rows: list[dict], cfg_lookup: dict[str, list[str]]) -> list[dict]:
    jobs = []
    devices = iter(range(8))
    for row in top_rows:
        cfg_name = row["config"]
        extra_args = cfg_lookup[cfg_name]
        for seed in (1, 2, 3):
            try:
                device = next(devices)
            except StopIteration:
                device = seed - 1
            prefix = f"cifar10_workers20_fedac_exact_stage2_{cfg_name}_seed"
            meta_path = stage2_output_meta(cfg_name, seed)
            jobs.append((cfg_name, seed, extra_args, meta_path, device, prefix))

    procs = []
    for _, seed, extra_args, meta_path, device, prefix in jobs:
        if meta_path.exists():
            continue
        procs.append(launch_single_seed(device, prefix, seed, extra_args, epochs=80, abort_epoch=20, abort_min_acc=30.0))
    for proc in procs:
        proc.wait()

    grouped = {}
    for cfg_name, seed, _, meta_path, _, _ in jobs:
        while not meta_path.exists():
            time.sleep(5)
        summary = load_summary(meta_path)
        grouped.setdefault(cfg_name, []).append(summary)

    rows = []
    for cfg_name, items in grouped.items():
        final_acc = [x["final_acc"] for x in items]
        best_acc = [x["best_acc"] for x in items]
        final_loss = [x["final_loss"] for x in items]
        rows.append(
            {
                "config": cfg_name,
                "num_seeds": len(items),
                "final_acc_mean": sum(final_acc) / len(final_acc),
                "best_acc_mean": sum(best_acc) / len(best_acc),
                "final_loss_mean": sum(final_loss) / len(final_loss),
            }
        )
    rows.sort(key=lambda r: (-r["final_acc_mean"], r["final_loss_mean"], r["config"]))
    write_csv(OUT_DIR / "stage2_summary.csv", rows, ["config", "num_seeds", "final_acc_mean", "best_acc_mean", "final_loss_mean"])
    return rows


def main() -> None:
    stage1_cfg_map = {name: parse_kv_args(args) for name, args in STAGE1_CONFIGS}
    stage1_rows = wait_for_stage1()
    best_stage1 = stage1_rows[0]

    decision = {
        "stage1_best": best_stage1,
        "action": None,
        "reason": None,
    }

    survivors = [row for row in stage1_rows if row["final_acc"] >= 35.0 or row["best_acc"] >= 40.0]
    if not survivors:
        bank2 = make_bank2(best_stage1, stage1_cfg_map)
        bank2_rows = run_bank(
            bank2,
            prefix_base="cifar10_workers20_fedac_exact_bank2_",
            epochs=40,
            abort_epoch=10,
            abort_min_acc=18.0,
        )
        write_csv(
            OUT_DIR / "bank2_summary.csv",
            bank2_rows,
            ["config", "seed", "final_acc", "best_acc", "final_loss", "best_loss", "completed_epochs", "stopped_early", "early_abort_reason", "path", "args"],
        )
        best_bank2 = bank2_rows[0]
        if best_bank2["final_acc"] < 35.0 and best_bank2["best_acc"] < 40.0:
            decision["action"] = "stop"
            decision["reason"] = "stage1_and_bank2_not_promising"
            decision["bank2_best"] = best_bank2
            with (OUT_DIR / "decision.json").open("w", encoding="utf-8") as fh:
                json.dump(decision, fh, indent=2, ensure_ascii=False)
            return
        stage2_candidates = bank2_rows[:2]
        stage2_lookup = {row["config"]: next(args for name, args in bank2 if name == row["config"]) for row in stage2_candidates}
        stage2_rows = run_stage2(stage2_candidates, stage2_lookup)
    else:
        stage2_candidates = survivors[:2]
        stage2_lookup = {row["config"]: next(args for name, args in STAGE1_CONFIGS if name == row["config"]) for row in stage2_candidates}
        stage2_rows = run_stage2(stage2_candidates, stage2_lookup)

    best_stage2 = stage2_rows[0]
    if best_stage2["final_acc_mean"] >= 50.0 and best_stage2["best_acc_mean"] >= 55.0:
        decision["action"] = "promote"
        decision["reason"] = "stage2_promising"
        decision["stage2_best"] = best_stage2
    else:
        decision["action"] = "switch_params"
        decision["reason"] = "stage2_not_good_enough"
        decision["stage2_best"] = best_stage2

    with (OUT_DIR / "decision.json").open("w", encoding="utf-8") as fh:
        json.dump(decision, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
