from __future__ import annotations

import pickle
import re
import statistics
import time
from pathlib import Path


EXP_DIR = Path("/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist/experiments")
NAME_RE = re.compile(
    r"^cifar100_resnet18_e300_ra_re_ls_"
    r"(base|labelsorted|dirichlet|highdelay)_"
    r"(asyncsam_rms|asyncsgd|fedasync|fedbuff)_seed(\d+)\.pkl(\.partial\.pkl)?$"
)


def read_record(path: Path) -> dict:
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


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    chosen: dict[tuple[str, str, int], tuple[tuple[int, float], Path]] = {}
    for path in EXP_DIR.glob("cifar100_resnet18_e300_ra_re_ls_*.pkl*"):
        match = NAME_RE.match(path.name)
        if not match:
            continue
        setting, method, seed_s, partial = match.groups()
        seed = int(seed_s)
        key = (setting, method, seed)
        score = (0 if partial else 1, path.stat().st_mtime)
        if key not in chosen or score > chosen[key][0]:
            chosen[key] = (score, path)

    rows = []
    for (setting, method, seed), (_score, path) in sorted(chosen.items()):
        try:
            rec = read_record(path)
        except Exception as exc:
            rows.append({"setting": setting, "method": method, "seed": seed, "bad": str(exc)})
            continue
        rec.update({"setting": setting, "method": method, "seed": seed})
        rows.append(rec)

    print(f"snapshot {time.strftime('%Y-%m-%d %H:%M:%S')} rows={len(rows)}")
    print("setting     method       n final ep_min/mean/max final_acc        best_acc         final_loss")
    for setting in ["base", "labelsorted", "dirichlet", "highdelay"]:
        for method in ["asyncsam_rms", "asyncsgd", "fedasync", "fedbuff"]:
            group = [r for r in rows if r.get("setting") == setting and r.get("method") == method and "bad" not in r]
            if not group:
                print(f"{setting:<11s} {method:<12s} no rows")
                continue
            acc_mean, acc_std = mean_std([r["acc"] for r in group])
            best_mean, best_std = mean_std([r["best"] for r in group])
            loss_mean, loss_std = mean_std([r["loss"] for r in group])
            print(
                f"{setting:<11s} {method:<12s} {len(group):2d} {sum(r['final'] for r in group):5d} "
                f"{min(r['epoch'] for r in group):3d}/{statistics.mean(r['epoch'] for r in group):5.1f}/{max(r['epoch'] for r in group):3d} "
                f"{acc_mean:6.2f} +/- {acc_std:5.2f} "
                f"{best_mean:6.2f} +/- {best_std:5.2f} "
                f"{loss_mean:7.4f} +/- {loss_std:7.4f}"
            )

    print("\nper_seed_asyncsam_rms")
    print("setting      seed st epoch final_acc best_acc loss")
    for setting in ["base", "labelsorted", "dirichlet", "highdelay"]:
        group = [r for r in rows if r.get("setting") == setting and r.get("method") == "asyncsam_rms" and "bad" not in r]
        for r in sorted(group, key=lambda item: item["seed"]):
            st = "F" if r["final"] else "P"
            print(
                f"{setting:<12s} {r['seed']:>4d} {st} {r['epoch']:5d} "
                f"{r['acc']:9.2f} {r['best']:8.2f} {r['loss']:7.4f}"
            )


if __name__ == "__main__":
    main()
