#!/usr/bin/env python3
import csv
import math
import pickle
import statistics
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


ROOT = Path("/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist")
EXP_DIR = ROOT / "experiments"
OUT_DIR = EXP_DIR / "added_baselines_fadas_ca2fl_constant_summary"


METHOD_LABELS = {
    "fadas": "FADAS",
    "ca2fl": "CA2FL",
    "asyncsgd_const": "AsyncSGD constant",
}

SETTING_LABELS = {
    "base": "Base IID",
    "labelsorted": "Label-Sorted",
    "dirichlet": "Dirichlet",
    "dirichlet005": "Dirichlet",
    "highdelay": "High Delay",
    "workers20": "20 Workers",
    "workers20_original": "20 Workers Original",
    "workers20_t3e200": "20 Workers T3/E200",
}

FAMILY_ORDER = [
    "MNIST",
    "CIFAR-10 ResNet18",
    "CIFAR-10 ResNet32",
    "CIFAR-10 ResNet56",
    "CIFAR-100 ResNet32",
    "CIFAR-100 ResNet18 300e RA+RE+LS",
    "CIFAR-100 ResNeXt29-8x16d 200e",
]
METHOD_ORDER = ["FADAS", "CA2FL", "AsyncSGD constant"]


def classify(path: Path):
    rel = path.relative_to(EXP_DIR).as_posix()
    name = path.name
    if path.name.endswith(".partial.pkl"):
        return None

    method = None
    for token, label in METHOD_LABELS.items():
        if f"_{token}_seed" in name:
            method = label
            break
    if method is None:
        return None

    if name.startswith("mnist_"):
        return "MNIST", "Base IID", method
    if name.startswith("validate10_labelsorted_"):
        return "MNIST", "Label-Sorted", method
    if name.startswith("validate10_dirichlet005_"):
        return "MNIST", "Dirichlet", method
    if name.startswith("validate10_highdelay_"):
        return "MNIST", "High Delay", method
    if name.startswith("validate10_workers20_"):
        return "MNIST", "20 Workers", method

    if name.startswith("cifar10_multiseed_"):
        return "CIFAR-10 ResNet18", "Base IID", method
    if name.startswith("cifar10_validate10_labelsorted_"):
        return "CIFAR-10 ResNet18", "Label-Sorted", method
    if name.startswith("cifar10_validate10_highdelay_"):
        return "CIFAR-10 ResNet18", "High Delay", method
    if name.startswith("cifar10_validate10_workers20_"):
        return "CIFAR-10 ResNet18", "20 Workers Original", method
    if name.startswith("cifar10_workers20_t3e200_"):
        return "CIFAR-10 ResNet18", "20 Workers T3/E200", method

    four_setting_prefixes = [
        ("cifar10_resnet32_p3_", "CIFAR-10 ResNet32"),
        ("cifar10_resnet56_p3_", "CIFAR-10 ResNet56"),
        ("cifar100_resnet32_p3_", "CIFAR-100 ResNet32"),
        ("cifar100_resnet18_e300_ra_re_ls_", "CIFAR-100 ResNet18 300e RA+RE+LS"),
        (
            "resnext29_8x16_lr003_e200_",
            "CIFAR-100 ResNeXt29-8x16d 200e",
        ),
    ]
    for prefix, family in four_setting_prefixes:
        if name.startswith(prefix):
            rest = name[len(prefix) :]
            for setting_token, setting_label in SETTING_LABELS.items():
                if rest.startswith(f"{setting_token}_"):
                    return family, setting_label, method

    if "by_family/cifar100/resnext29/" in rel and name.startswith("resnext29_8x16_lr003_e200_"):
        rest = name[len("resnext29_8x16_lr003_e200_") :]
        for setting_token, setting_label in SETTING_LABELS.items():
            if rest.startswith(f"{setting_token}_"):
                return "CIFAR-100 ResNeXt29-8x16d 200e", setting_label, method

    return None


def mean(values):
    return statistics.mean(values) if values else math.nan


def std(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def summarize_group(items):
    finals_acc = [item["final_acc"] for item in items]
    bests_acc = [item["best_acc"] for item in items]
    finals_loss = [item["final_loss"] for item in items]
    bests_loss = [item["best_loss"] for item in items]
    seeds = sorted(item["seed"] for item in items)
    missing = [seed for seed in range(1, 11) if seed not in seeds]
    return {
        "num_seeds": len(items),
        "final_acc": mean(finals_acc),
        "final_acc_std": std(finals_acc),
        "best_acc": mean(bests_acc),
        "best_acc_std": std(bests_acc),
        "final_loss": mean(finals_loss),
        "final_loss_std": std(finals_loss),
        "best_loss": mean(bests_loss),
        "best_loss_std": std(bests_loss),
        "missing_seeds": " ".join(str(seed) for seed in missing),
    }


def load_records():
    groups = defaultdict(list)
    bad = []
    patterns = ["*fadas_seed*.pkl", "*ca2fl_seed*.pkl", "*asyncsgd_const_seed*.pkl"]
    for pattern in patterns:
        for path in EXP_DIR.rglob(pattern):
            if path.name.endswith(".partial.pkl"):
                continue
            key = classify(path)
            if key is None:
                continue
            try:
                with path.open("rb") as handle:
                    result = pickle.load(handle)
                test_prec = result.get("test_prec") or []
                test_loss = result.get("test_loss") or []
                args = result.get("args", {})
                status = result.get("status", {})
                if not test_prec or not test_loss:
                    raise ValueError("missing test metrics")
                target_epochs = int(args.get("epochs") or 0)
                completed_epochs = int(status.get("completed_epochs") or len(test_prec))
                if target_epochs > 0 and completed_epochs < target_epochs:
                    raise ValueError(f"incomplete {completed_epochs}/{target_epochs}")
                groups[key].append(
                    {
                        "seed": int(args.get("seed")),
                        "final_acc": 100.0 * float(test_prec[-1]),
                        "best_acc": 100.0 * max(float(x) for x in test_prec),
                        "final_loss": float(test_loss[-1]),
                        "best_loss": min(float(x) for x in test_loss),
                        "path": str(path),
                    }
                )
            except Exception as exc:
                bad.append((str(path), repr(exc)))
    return groups, bad


def write_summary(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary.csv"
    fieldnames = [
        "family",
        "setting",
        "method",
        "num_seeds",
        "final_acc",
        "final_acc_std",
        "best_acc",
        "best_acc_std",
        "final_loss",
        "final_loss_std",
        "best_loss",
        "best_loss_std",
        "missing_seeds",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def write_family_summary(rows):
    agg = defaultdict(list)
    for row in rows:
        agg[(row["family"], row["method"])].append(row)
    family_rows = []
    for (family, method), items in agg.items():
        family_rows.append(
            {
                "family": family,
                "method": method,
                "num_settings": len(items),
                "mean_final_acc_across_settings": mean([float(item["final_acc"]) for item in items]),
                "mean_best_acc_across_settings": mean([float(item["best_acc"]) for item in items]),
                "mean_final_loss_across_settings": mean([float(item["final_loss"]) for item in items]),
                "min_final_acc_setting": min(items, key=lambda item: float(item["final_acc"]))["setting"],
                "min_final_acc": min(float(item["final_acc"]) for item in items),
                "max_final_acc_setting": max(items, key=lambda item: float(item["final_acc"]))["setting"],
                "max_final_acc": max(float(item["final_acc"]) for item in items),
            }
        )

    order = {name: idx for idx, name in enumerate(FAMILY_ORDER)}
    method_order = {name: idx for idx, name in enumerate(METHOD_ORDER)}
    family_rows.sort(key=lambda row: (order.get(row["family"], 999), method_order.get(row["method"], 999)))

    path = OUT_DIR / "family_summary.csv"
    with path.open("w", newline="") as handle:
        fieldnames = [
            "family",
            "method",
            "num_settings",
            "mean_final_acc_across_settings",
            "mean_best_acc_across_settings",
            "mean_final_loss_across_settings",
            "min_final_acc_setting",
            "min_final_acc",
            "max_final_acc_setting",
            "max_final_acc",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(family_rows)
    return path, family_rows


def plot_family_summary(family_rows):
    if plt is None:
        return plot_family_summary_svg(family_rows), None
    families = [family for family in FAMILY_ORDER if any(row["family"] == family for row in family_rows)]
    methods = [method for method in METHOD_ORDER if any(row["method"] == method for row in family_rows)]
    lookup = {(row["family"], row["method"]): row for row in family_rows}

    fig, ax = plt.subplots(figsize=(13, 5.6))
    width = 0.24
    xs = list(range(len(families)))
    colors = {
        "FADAS": "#3366cc",
        "CA2FL": "#dc3912",
        "AsyncSGD constant": "#109618",
    }
    for offset, method in enumerate(methods):
        vals = [
            lookup.get((family, method), {}).get("mean_final_acc_across_settings", math.nan)
            for family in families
        ]
        pos = [x + (offset - (len(methods) - 1) / 2) * width for x in xs]
        ax.bar(pos, vals, width=width, label=method, color=colors.get(method))

    ax.set_ylabel("Mean Final Accuracy Across Settings (%)")
    ax.set_xticks(xs)
    ax.set_xticklabels(families, rotation=22, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=3, frameon=True)
    fig.tight_layout()
    png = OUT_DIR / "family_mean_final_acc.png"
    pdf = OUT_DIR / "family_mean_final_acc.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_family_summary_svg(family_rows):
    families = [family for family in FAMILY_ORDER if any(row["family"] == family for row in family_rows)]
    methods = [method for method in METHOD_ORDER if any(row["method"] == method for row in family_rows)]
    lookup = {(row["family"], row["method"]): row for row in family_rows}
    colors = {
        "FADAS": "#3366cc",
        "CA2FL": "#dc3912",
        "AsyncSGD constant": "#109618",
    }

    width = 1200
    height = 520
    left = 70
    right = 30
    top = 35
    bottom = 150
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_y = 100.0
    group_w = chart_w / max(1, len(families))
    bar_w = min(26, group_w / (len(methods) + 2))

    def y(val):
        return top + chart_h * (1.0 - float(val) / max_y)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:12px}.title{font-size:18px;font-weight:bold}.axis{font-size:13px}</style>',
        f'<text x="{width/2}" y="24" text-anchor="middle" class="title">Added Baselines: Mean Final Accuracy Across Settings</text>',
    ]
    for tick in range(0, 101, 20):
        yy = y(tick)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#ddd" stroke-dasharray="4 3"/>')
        lines.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end">{tick}</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="#333"/>')
    lines.append(f'<line x1="{left}" y1="{top+chart_h}" x2="{width-right}" y2="{top+chart_h}" stroke="#333"/>')
    lines.append(f'<text x="20" y="{top+chart_h/2}" transform="rotate(-90 20 {top+chart_h/2})" class="axis" text-anchor="middle">Accuracy (%)</text>')

    for i, family in enumerate(families):
        center = left + group_w * (i + 0.5)
        for j, method in enumerate(methods):
            val = lookup.get((family, method), {}).get("mean_final_acc_across_settings")
            if val is None:
                continue
            x = center + (j - (len(methods) - 1) / 2) * (bar_w + 4) - bar_w / 2
            yy = y(val)
            lines.append(
                f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{top+chart_h-yy:.1f}" fill="{colors.get(method, "#777")}"/>'
            )
        label = family.replace(" ", "&#160;")
        lines.append(
            f'<text x="{center:.1f}" y="{top+chart_h+20}" text-anchor="end" transform="rotate(-35 {center:.1f} {top+chart_h+20})">{label}</text>'
        )

    legend_x = left
    legend_y = height - 28
    for method in methods:
        lines.append(f'<rect x="{legend_x}" y="{legend_y-10}" width="14" height="14" fill="{colors.get(method, "#777")}"/>')
        lines.append(f'<text x="{legend_x+20}" y="{legend_y+2}">{method}</text>')
        legend_x += 170
    lines.append("</svg>")
    path = OUT_DIR / "family_mean_final_acc.svg"
    path.write_text("\n".join(lines))
    return path


def main():
    groups, bad = load_records()
    rows = []
    family_order = {name: idx for idx, name in enumerate(FAMILY_ORDER)}
    setting_order = {
        "Base IID": 0,
        "Label-Sorted": 1,
        "Dirichlet": 2,
        "High Delay": 3,
        "20 Workers": 4,
        "20 Workers Original": 4,
        "20 Workers T3/E200": 5,
    }
    method_order = {name: idx for idx, name in enumerate(METHOD_ORDER)}
    for (family, setting, method), items in groups.items():
        summary = summarize_group(items)
        rows.append({"family": family, "setting": setting, "method": method, **summary})
    rows.sort(
        key=lambda row: (
            family_order.get(row["family"], 999),
            setting_order.get(row["setting"], 999),
            method_order.get(row["method"], 999),
        )
    )

    summary_path = write_summary(rows)
    family_path, family_rows = write_family_summary(rows)
    png, pdf = plot_family_summary(family_rows)

    bad_path = OUT_DIR / "bad_files.txt"
    bad_path.write_text("\n".join(f"{path}\t{err}" for path, err in bad) + ("\n" if bad else ""))

    complete = sum(1 for row in rows if int(row["num_seeds"]) == 10 and not row["missing_seeds"])
    print(f"summary={summary_path}")
    print(f"family_summary={family_path}")
    print(f"figure={png}")
    print(f"rows={len(rows)} complete_rows={complete} bad_files={len(bad)}")


if __name__ == "__main__":
    main()
