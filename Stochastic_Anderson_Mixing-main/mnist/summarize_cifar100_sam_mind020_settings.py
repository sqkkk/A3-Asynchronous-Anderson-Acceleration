from __future__ import annotations

import pickle
import re
import statistics
import time
from pathlib import Path


EXP_DIR = Path("/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist/experiments")

PATTERNS = {
    "base": re.compile(r"^cifar100_tune_r18_e300_sam_mind020_asyncsam_rms_seed(\d+)\.pkl(\.partial\.pkl)?$"),
    "labelsorted": re.compile(r"^cifar100_tune_r18_e300_labelsorted_sam_mind020_asyncsam_rms_seed(\d+)\.pkl(\.partial\.pkl)?$"),
    "dirichlet": re.compile(r"^cifar100_tune_r18_e300_dirichlet_sam_mind020_asyncsam_rms_seed(\d+)\.pkl(\.partial\.pkl)?$"),
    "highdelay": re.compile(r"^cifar100_tune_r18_e300_highdelay_sam_mind020_asyncsam_rms_seed(\d+)\.pkl(\.partial\.pkl)?$"),
}


def read(path: Path) -> dict:
    with path.open("rb") as f:
        data = pickle.load(f)
    test_prec = data.get("test_prec") or []
    test_loss = data.get("test_loss") or []
    status = data.get("status") or {}
    return {
        "epoch": int(status.get("completed_epochs") or len(test_prec)),
        "acc": 100.0 * float(test_prec[-1]) if test_prec else float("nan"),
        "best": 100.0 * float(max(test_prec)) if test_prec else float("nan"),
        "loss": float(test_loss[-1]) if test_loss else float("nan"),
        "final": not path.name.endswith(".partial.pkl"),
    }


def main() -> None:
    rows = []
    for setting, pattern in PATTERNS.items():
        chosen = {}
        for path in EXP_DIR.glob("cifar100_tune_r18_e300*sam_mind020_asyncsam_rms_seed*.pkl*"):
            match = pattern.match(path.name)
            if not match:
                continue
            seed = int(match.group(1))
            is_partial = bool(match.group(2))
            score = (0 if is_partial else 1, path.stat().st_mtime)
            if seed not in chosen or score > chosen[seed][0]:
                chosen[seed] = (score, path)
        for seed, (_score, path) in chosen.items():
            try:
                rec = read(path)
            except Exception as exc:
                rows.append({"setting": setting, "seed": seed, "bad": str(exc)})
                continue
            rec["setting"] = setting
            rec["seed"] = seed
            rows.append(rec)

    print(f"snapshot {time.strftime('%Y-%m-%d %H:%M:%S')} rows={len(rows)}")
    good = [r for r in rows if "bad" not in r]
    for setting in ["base", "labelsorted", "dirichlet", "highdelay"]:
        group = [r for r in good if r["setting"] == setting]
        if not group:
            print(f"{setting:<12s} no rows")
            continue
        print(
            f"{setting:<12s} n={len(group):2d} F={sum(r['final'] for r in group):2d} "
            f"ep={min(r['epoch'] for r in group):3d}/{statistics.mean(r['epoch'] for r in group):5.1f}/{max(r['epoch'] for r in group):3d} "
            f"acc={statistics.mean(r['acc'] for r in group):6.2f} "
            f"best={statistics.mean(r['best'] for r in group):6.2f} "
            f"loss={statistics.mean(r['loss'] for r in group):7.4f}"
        )
    print("setting      seed st ep   acc   best    loss")
    for r in sorted(rows, key=lambda item: (item.get("setting", ""), item.get("seed", 0))):
        if "bad" in r:
            print(f"{r['setting']:<12s} {r['seed']:>4d} BAD {r['bad']}")
            continue
        status = "F" if r["final"] else "P"
        print(
            f"{r['setting']:<12s} {r['seed']:>4d} {status} "
            f"{r['epoch']:3d} {r['acc']:6.2f} {r['best']:6.2f} {r['loss']:7.4f}"
        )


if __name__ == "__main__":
    main()
