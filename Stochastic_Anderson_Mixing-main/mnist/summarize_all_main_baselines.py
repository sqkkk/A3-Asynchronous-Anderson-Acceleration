#!/usr/bin/env python3
"""Merge the main 10-seed comparison tables with newly added baselines.

The project now has two classes of summaries:
- the original four main methods: AsyncSAM+RMS, AsyncSGD, FedAsync, FedBuff
- the newly added baselines: FADAS, CA2FL, AsyncSGD constant

Older summary files were produced by different plotting scripts, so this file
normalizes their column names and setting labels into one paper-facing table.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path("/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist")
EXP_DIR = ROOT / "experiments"
OUT_DIR = EXP_DIR / "all_main_baselines_summary"

ADDED_SUMMARY = EXP_DIR / "added_baselines_fadas_ca2fl_constant_summary" / "summary.csv"

CSV_SOURCES = [
    {
        "path": EXP_DIR / "afl_same_setting_10seed_fig_full" / "summary.csv",
        "family": "MNIST",
        "setting": "Base IID",
        "method_col": "label",
        "source": "mnist_base",
    },
    {
        "path": EXP_DIR / "validation_other_settings_10seed_fig" / "summary.csv",
        "family": "MNIST",
        "method_col": "method",
        "source": "mnist_validation",
    },
    {
        "path": EXP_DIR / "validation_dirichlet_10seed_fig" / "summary.csv",
        "family": "MNIST",
        "setting": "Dirichlet",
        "method_col": "method",
        "source": "mnist_dirichlet",
    },
    {
        "path": EXP_DIR / "cifar10_10seed_compare_legacy92_with_fedac_fig" / "summary.csv",
        "family": "CIFAR-10 ResNet18",
        "setting": "Base IID",
        "method_col": "label",
        "source": "cifar10_resnet18_base",
    },
    {
        "path": EXP_DIR / "cifar10_validation_settings_10seed_fig" / "summary.csv",
        "family": "CIFAR-10 ResNet18",
        "method_col": "method",
        "source": "cifar10_resnet18_validation",
    },
    {
        "path": EXP_DIR / "cifar10_workers20_t3e200_10seed_fig" / "summary.csv",
        "family": "CIFAR-10 ResNet18",
        "setting": "20 Workers T3/E200",
        "method_col": "method",
        "source": "cifar10_resnet18_workers20_t3e200",
    },
    {
        "path": EXP_DIR / "cifar10_resnet32_four_settings_fixed_public_10seed_fig" / "summary.csv",
        "family": "CIFAR-10 ResNet32",
        "method_col": "method",
        "source": "cifar10_resnet32",
    },
    {
        "path": EXP_DIR / "cifar10_resnet56_four_settings_parallel_10seed_fig" / "summary.csv",
        "family": "CIFAR-10 ResNet56",
        "method_col": "method",
        "source": "cifar10_resnet56",
    },
    {
        "path": EXP_DIR / "cifar100_resnet32_four_settings_parallel_10seed_fig" / "summary.csv",
        "family": "CIFAR-100 ResNet32",
        "method_col": "method",
        "source": "cifar100_resnet32",
    },
    {
        "path": EXP_DIR / "cifar100_resnet18_e300_ra_re_ls_four_settings_10seed_fig" / "summary.csv",
        "family": "CIFAR-100 ResNet18 300e RA+RE+LS",
        "method_col": "method",
        "source": "cifar100_resnet18_300e",
    },
    {
        "path": EXP_DIR
        / "by_family"
        / "cifar100"
        / "resnext29"
        / "resnext29_8x16_lr003_e200_four_settings_10seed_fig"
        / "summary.csv",
        "family": "CIFAR-100 ResNeXt29-8x16d 200e",
        "method_col": "method",
        "source": "cifar100_resnext29_200e",
    },
]

FAMILY_ORDER = [
    "MNIST",
    "CIFAR-10 ResNet18",
    "CIFAR-10 ResNet32",
    "CIFAR-10 ResNet56",
    "CIFAR-100 ResNet32",
    "CIFAR-100 ResNet18 300e RA+RE+LS",
    "CIFAR-100 ResNeXt29-8x16d 200e",
]

SETTING_ORDER = [
    "Base IID",
    "Label-Sorted",
    "Dirichlet",
    "High Delay",
    "20 Workers",
    "20 Workers Original",
    "20 Workers T3/E200",
]

METHOD_ORDER = [
    "AsyncSAM+RMS",
    "AsyncSGD",
    "AsyncSGD+RMS",
    "FedAsync",
    "FedBuff",
    "FADAS",
    "CA2FL",
    "AsyncSGD constant",
]

MAIN7_METHOD_ORDER = [
    "AsyncSAM+RMS",
    "AsyncSGD",
    "FedAsync",
    "FedBuff",
    "FADAS",
    "CA2FL",
    "AsyncSGD constant",
]

KEEP_METHODS = set(METHOD_ORDER)

EXCLUDED_SETTINGS = {
    ("CIFAR-10 ResNet18", "20 Workers Original"),
}

EXCLUDED_FAMILIES = {
    "CIFAR-100 ResNet32",
}

SETTING_ALIASES = {
    "Label-Sorted Non-IID": "Label-Sorted",
    "Dirichlet Non-IID": "Dirichlet",
    "High Delay IID": "High Delay",
    "IID 20 Workers": "20 Workers Original",
}

METHOD_ALIASES = {
    "AsyncSAM RMS+AA": "AsyncSAM+RMS",
    "AsyncSAM RMS": "AsyncSAM+RMS",
    "AsyncSAM RMS+AA tuned": "AsyncSAM+RMS",
    "FedAC tuned": "FedAC",
}

NUMERIC_COLUMNS = [
    "final_acc",
    "final_acc_std",
    "best_acc",
    "best_acc_std",
    "final_loss",
    "final_loss_std",
    "best_loss",
    "best_loss_std",
]


def float_or_nan(value):
    if value is None or value == "":
        return math.nan
    return float(value)


def metric(row, base):
    """Read metrics from either modern or legacy summary column names."""
    candidates = [base, f"{base}_mean"]
    for key in candidates:
        if key in row and row[key] != "":
            return row[key]
    return ""


def norm_method(value):
    value = METHOD_ALIASES.get(value, value)
    return value.strip()


def norm_setting(value):
    value = SETTING_ALIASES.get(value, value)
    return value.strip()


def source_priority(source):
    # Prefer the dedicated base summary for raw MNIST AsyncSGD; otherwise the
    # validation table has the same AsyncSAM/FedAsync/FedBuff values plus more.
    priorities = {
        "mnist_base": 10,
        "mnist_validation": 20,
        "mnist_dirichlet": 30,
    }
    return priorities.get(source, 50)


def should_include(row):
    if row["family"] in EXCLUDED_FAMILIES:
        return False
    if (row["family"], row["setting"]) in EXCLUDED_SETTINGS:
        return False
    return True


def row_sort_key(row):
    family_idx = FAMILY_ORDER.index(row["family"]) if row["family"] in FAMILY_ORDER else 999
    setting = row.get("setting", "")
    setting_idx = SETTING_ORDER.index(setting) if setting in SETTING_ORDER else 999
    method_idx = METHOD_ORDER.index(row["method"]) if row["method"] in METHOD_ORDER else 999
    return family_idx, setting_idx, method_idx, row["method"]


def load_previous_rows():
    rows_by_key = {}
    priorities = {}
    for spec in CSV_SOURCES:
        path = spec["path"]
        if not path.exists():
            print(f"[warn] missing source: {path}")
            continue
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                method = norm_method(raw.get(spec["method_col"], ""))
                if method not in KEEP_METHODS:
                    continue
                setting = norm_setting(spec.get("setting") or raw.get("setting", ""))
                if spec["family"] == "MNIST" and setting == "20 Workers Original":
                    setting = "20 Workers"
                if not setting:
                    raise ValueError(f"missing setting in {path}: {raw}")
                row = {
                    "family": spec["family"],
                    "setting": setting,
                    "method": method,
                    "num_seeds": raw.get("num_seeds") or "10",
                    "source": spec["source"],
                }
                for col in NUMERIC_COLUMNS:
                    row[col] = metric(raw, col)
                if not should_include(row):
                    continue
                key = (row["family"], row["setting"], row["method"])
                priority = source_priority(spec["source"])
                if key not in rows_by_key or priority >= priorities[key]:
                    rows_by_key[key] = row
                    priorities[key] = priority
    return rows_by_key


def load_added_rows(rows_by_key):
    if not ADDED_SUMMARY.exists():
        print(f"[warn] missing added summary: {ADDED_SUMMARY}")
        return
    with ADDED_SUMMARY.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            method = norm_method(raw["method"])
            if method not in KEEP_METHODS:
                continue
            row = {
                "family": raw["family"],
                "setting": norm_setting(raw["setting"]),
                "method": method,
                "num_seeds": raw.get("num_seeds") or "10",
                "source": "added_baselines",
            }
            for col in NUMERIC_COLUMNS:
                row[col] = raw.get(col, "")
            if not should_include(row):
                continue
            rows_by_key[(row["family"], row["setting"], row["method"])] = row


def write_summary(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "summary.csv"
    fieldnames = [
        "family",
        "setting",
        "method",
        "num_seeds",
        *NUMERIC_COLUMNS,
        "source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_wide(rows, metric_name, filename, methods=METHOD_ORDER):
    path = OUT_DIR / filename
    groups = defaultdict(dict)
    for row in rows:
        groups[(row["family"], row["setting"])][row["method"]] = row.get(metric_name, "")

    fieldnames = ["family", "setting", *methods]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for family, setting in sorted(groups, key=lambda key: row_sort_key({"family": key[0], "setting": key[1], "method": ""})):
            values = groups[(family, setting)]
            writer.writerow(
                {
                    "family": family,
                    "setting": setting,
                    **{method: values.get(method, "") for method in methods},
                }
            )
    return path


def write_family_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["method"])].append(row)

    out_rows = []
    for (family, method), items in grouped.items():
        final_acc = [float_or_nan(item["final_acc"]) for item in items]
        best_acc = [float_or_nan(item["best_acc"]) for item in items]
        final_loss = [float_or_nan(item["final_loss"]) for item in items]
        final_acc = [x for x in final_acc if not math.isnan(x)]
        best_acc = [x for x in best_acc if not math.isnan(x)]
        final_loss = [x for x in final_loss if not math.isnan(x)]
        out_rows.append(
            {
                "family": family,
                "method": method,
                "num_settings": len(items),
                "mean_final_acc_across_settings": sum(final_acc) / len(final_acc),
                "mean_best_acc_across_settings": sum(best_acc) / len(best_acc),
                "mean_final_loss_across_settings": sum(final_loss) / len(final_loss),
            }
        )
    out_rows.sort(key=row_sort_key)

    path = OUT_DIR / "family_summary.csv"
    with path.open("w", newline="") as handle:
        fieldnames = [
            "family",
            "method",
            "num_settings",
            "mean_final_acc_across_settings",
            "mean_best_acc_across_settings",
            "mean_final_loss_across_settings",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    return path, out_rows


def write_markdown_wide(rows, filename="wide_final_acc.md", methods=METHOD_ORDER):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["family"], row["setting"])][row["method"]] = row

    path = OUT_DIR / filename
    with path.open("w") as handle:
        handle.write("# All Main Baselines: Final Accuracy (%)\n\n")
        handle.write("| Family | Setting | " + " | ".join(methods) + " |\n")
        handle.write("|---|---|" + "|".join(["---:"] * len(methods)) + "|\n")
        for key in sorted(grouped, key=lambda key: row_sort_key({"family": key[0], "setting": key[1], "method": ""})):
            family, setting = key
            cells = []
            for method in methods:
                item = grouped[key].get(method)
                if not item:
                    cells.append("")
                    continue
                acc = float_or_nan(item["final_acc"])
                std = float_or_nan(item["final_acc_std"])
                if math.isnan(acc):
                    cells.append("")
                elif math.isnan(std):
                    cells.append(f"{acc:.2f}")
                else:
                    cells.append(f"{acc:.2f} +/- {std:.2f}")
            handle.write(f"| {family} | {setting} | " + " | ".join(cells) + " |\n")
    return path


def write_family_markdown(family_rows):
    path = OUT_DIR / "family_summary.md"
    with path.open("w") as handle:
        handle.write("# Mean Across Settings\n\n")
        handle.write("| Family | Method | Settings | Mean Final Acc | Mean Best Acc | Mean Final Loss |\n")
        handle.write("|---|---|---:|---:|---:|---:|\n")
        for row in family_rows:
            handle.write(
                f"| {row['family']} | {row['method']} | {row['num_settings']} | "
                f"{row['mean_final_acc_across_settings']:.2f} | "
                f"{row['mean_best_acc_across_settings']:.2f} | "
                f"{row['mean_final_loss_across_settings']:.4f} |\n"
            )
    return path


def main():
    rows_by_key = load_previous_rows()
    load_added_rows(rows_by_key)
    rows = sorted(rows_by_key.values(), key=row_sort_key)

    summary_path = write_summary(rows)
    wide_acc_path = write_wide(rows, "final_acc", "wide_final_acc.csv")
    wide_loss_path = write_wide(rows, "final_loss", "wide_final_loss.csv")
    wide_acc_main7_path = write_wide(
        rows, "final_acc", "wide_final_acc_main7.csv", MAIN7_METHOD_ORDER
    )
    wide_loss_main7_path = write_wide(
        rows, "final_loss", "wide_final_loss_main7.csv", MAIN7_METHOD_ORDER
    )
    family_path, family_rows = write_family_summary(rows)
    wide_md_path = write_markdown_wide(rows)
    wide_main7_md_path = write_markdown_wide(
        rows, "wide_final_acc_main7.md", MAIN7_METHOD_ORDER
    )
    family_md_path = write_family_markdown(family_rows)

    print(f"rows={len(rows)}")
    print(f"summary={summary_path}")
    print(f"wide_final_acc={wide_acc_path}")
    print(f"wide_final_loss={wide_loss_path}")
    print(f"wide_final_acc_main7={wide_acc_main7_path}")
    print(f"wide_final_loss_main7={wide_loss_main7_path}")
    print(f"family_summary={family_path}")
    print(f"wide_markdown={wide_md_path}")
    print(f"wide_main7_markdown={wide_main7_md_path}")
    print(f"family_markdown={family_md_path}")


if __name__ == "__main__":
    main()
