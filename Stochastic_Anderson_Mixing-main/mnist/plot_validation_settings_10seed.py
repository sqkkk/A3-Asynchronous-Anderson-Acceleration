import csv
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


BASE_DIR = Path("/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist/experiments")
OUT_DIR = BASE_DIR / "validation_other_settings_10seed_fig"


def paths(prefix: str):
    return [str(BASE_DIR / f"{prefix}{seed}.pkl") for seed in range(1, 11)]


def preferred_paths(prefix: str, retry_prefix: str | None = None):
    resolved = []
    for seed in range(1, 11):
        primary = BASE_DIR / f"{prefix}{seed}.pkl"
        retry = BASE_DIR / f"{retry_prefix}{seed}.pkl" if retry_prefix is not None else None
        candidates = [path for path in (retry, primary) if path is not None]
        chosen = None
        for path in candidates:
            if path.exists() and path.stat().st_size > 0:
                chosen = path
                break
        if chosen is None:
            chosen = retry if retry is not None and retry.exists() else primary
        resolved.append(str(chosen))
    return resolved


SETTING_FILES = {
    "Base IID": {
        "AsyncSAM+RMS": paths("asyncsam_precond_q3_rms90_i1_batch0_seed"),
        "AsyncSAM": paths("ablate_asyncsam_seed"),
        "AsyncSGD+RMS": paths("ablate_asyncsgd_rms_seed"),
        "FedAsync": paths("afl_same_setting_fedasync_seed"),
        "FedBuff": paths("afl_same_setting_fedbuff_seed"),
        "FedAC": paths("afl_same_setting_fedac_tuned_seed"),
    },
    "Label-Sorted Non-IID": {
        "AsyncSAM+RMS": paths("validate10_labelsorted_asyncsam_rms_seed"),
        "AsyncSAM": paths("validate10_labelsorted_asyncsam_seed"),
        "AsyncSGD+RMS": paths("validate10_labelsorted_asyncsgd_rms_seed"),
        "FedAsync": paths("validate10_labelsorted_fedasync_seed"),
        "FedBuff": paths("validate10_labelsorted_fedbuff_seed"),
        "FedAC": paths("validate10_labelsorted_fedac_tuned_seed"),
    },
    "High Delay IID": {
        "AsyncSAM+RMS": paths("validate10_highdelay_asyncsam_rms_seed"),
        "AsyncSAM": paths("validate10_highdelay_asyncsam_seed"),
        "AsyncSGD+RMS": paths("validate10_highdelay_asyncsgd_rms_seed"),
        "FedAsync": paths("validate10_highdelay_fedasync_seed"),
        "FedBuff": paths("validate10_highdelay_fedbuff_seed"),
        "FedAC": paths("validate10_highdelay_fedac_tuned_seed"),
    },
    "IID 20 Workers": {
        "AsyncSAM+RMS": paths("validate10_workers20_asyncsam_rms_seed"),
        "AsyncSAM": paths("validate10_workers20_asyncsam_seed"),
        "AsyncSGD+RMS": paths("validate10_workers20_asyncsgd_rms_seed"),
        "FedAsync": preferred_paths(
            "validate10_workers20_fedasync_seed",
            "validate10_workers20_fedasync_retry_seed",
        ),
        "FedBuff": paths("validate10_workers20_fedbuff_seed"),
        "FedAC": preferred_paths(
            "validate10_workers20_fedac_tuned_seed",
            "validate10_workers20_fedac_tuned_retry_seed",
        ),
    },
}


METHOD_ORDER = [
    "AsyncSAM+RMS",
    "AsyncSAM",
    "AsyncSGD+RMS",
    "FedAsync",
    "FedBuff",
    "FedAC",
]


METHOD_COLORS = {
    "AsyncSAM+RMS": "#1f77b4",
    "AsyncSAM": "#ff7f0e",
    "AsyncSGD+RMS": "#2ca02c",
    "FedAsync": "#7f7f7f",
    "FedBuff": "#9467bd",
    "FedAC": "#8c564b",
}


def load_metrics(path):
    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return {
        "final_acc": 100.0 * data["test_prec"][-1],
        "best_acc": 100.0 * max(data["test_prec"]),
        "final_loss": data["test_loss"][-1],
        "best_loss": min(data["test_loss"]),
    }


def aggregate_metrics(path_list):
    rows = [row for row in (load_metrics(path) for path in path_list) if row is not None]
    if not rows:
        out = {}
        for key in ("final_acc", "best_acc", "final_loss", "best_loss"):
            out[key] = float("nan")
            out[f"{key}_std"] = float("nan")
        return out
    out = {}
    for key in ("final_acc", "best_acc", "final_loss", "best_loss"):
        vals = np.asarray([row[key] for row in rows], dtype=float)
        out[key] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std())
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    setting_metrics = {}
    for setting, method_map in SETTING_FILES.items():
        setting_metrics[setting] = {}
        for method, file_list in method_map.items():
            metrics = aggregate_metrics(file_list)
            setting_metrics[setting][method] = metrics
            row = {"setting": setting, "method": method}
            row.update(metrics)
            row["paths"] = ";".join(file_list)
            rows.append(row)

    csv_path = OUT_DIR / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "setting",
                "method",
                "final_acc",
                "final_acc_std",
                "best_acc",
                "best_acc_std",
                "final_loss",
                "final_loss_std",
                "best_loss",
                "best_loss_std",
                "paths",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    settings = list(SETTING_FILES.keys())
    x = np.arange(len(settings))
    width = min(0.14, 0.78 / len(METHOD_ORDER))
    offsets = np.linspace(-(len(METHOD_ORDER) - 1) / 2, (len(METHOD_ORDER) - 1) / 2, len(METHOD_ORDER)) * width

    png_path = OUT_DIR / "validation_settings_compare.png"
    pdf_path = OUT_DIR / "validation_settings_compare.pdf"
    if plt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
        metric_specs = [
            ("final_acc", "final_acc_std", "Final Accuracy (%)"),
            ("best_acc", "best_acc_std", "Best Accuracy (%)"),
            ("final_loss", "final_loss_std", "Final Test Loss"),
        ]

        for ax, (metric_key, std_key, metric_label) in zip(axes, metric_specs):
            for offset, method in zip(offsets, METHOD_ORDER):
                vals = []
                errs = []
                for setting in settings:
                    metrics = setting_metrics[setting].get(method)
                    vals.append(np.nan if metrics is None else metrics[metric_key])
                    errs.append(np.nan if metrics is None else metrics[std_key])
                ax.bar(
                    x + offset,
                    vals,
                    width=width,
                    yerr=errs,
                    capsize=3,
                    label=method,
                    color=METHOD_COLORS[method],
                    alpha=0.9,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(settings, rotation=15, ha="right")
            ax.set_ylabel(metric_label)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            if metric_key != "final_loss":
                ax.set_ylim(top=100)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.08))
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
    else:
        render_with_pil(setting_metrics, settings, png_path, pdf_path)

    print(csv_path)
    print(png_path)
    if plt is not None:
        print(pdf_path)
    else:
        print(pdf_path)


def render_with_pil(setting_metrics, settings, png_path, pdf_path):
    width, height = 2100, 680
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    margin_left = 90
    margin_right = 40
    margin_top = 90
    margin_bottom = 130
    panel_gap = 30
    panel_width = (width - margin_left - margin_right - 2 * panel_gap) // 3
    panel_height = height - margin_top - margin_bottom

    metric_specs = [
        ("final_acc", "Final Accuracy (%)"),
        ("best_acc", "Best Accuracy (%)"),
        ("final_loss", "Final Test Loss"),
    ]
    max_values = {
        "final_acc": 100.0,
        "best_acc": 100.0,
        "final_loss": max(
            float(setting_metrics[s][m]["final_loss"])
            for s in settings
            for m in setting_metrics[s]
            if not np.isnan(setting_metrics[s][m]["final_loss"])
        )
        * 1.1,
    }

    draw.text((width // 2 - 170, 20), "Validation Across Other Settings (10 Seeds)", fill="black", font=title_font)

    legend_x, legend_y = margin_left, 50
    for method in METHOD_ORDER:
        color = METHOD_COLORS[method]
        draw.rectangle([legend_x, legend_y, legend_x + 14, legend_y + 14], fill=color, outline="black")
        draw.text((legend_x + 20, legend_y), method, fill="black", font=font)
        legend_x += 180
        if legend_x > width - 220:
            legend_x = margin_left
            legend_y += 22

    for panel_idx, (metric_key, metric_label) in enumerate(metric_specs):
        x0 = margin_left + panel_idx * (panel_width + panel_gap)
        y0 = margin_top
        x1 = x0 + panel_width
        y1 = y0 + panel_height
        draw.rectangle([x0, y0, x1, y1], outline="black", width=1)
        draw.text((x0 + 8, y0 - 22), metric_label, fill="black", font=font)

        for tick in range(5):
            ratio = tick / 4.0
            ty = y1 - int(ratio * (panel_height - 20))
            draw.line([x0, ty, x1, ty], fill="#dddddd", width=1)
            tick_value = max_values[metric_key] * ratio
            label = f"{tick_value:.2f}" if metric_key == "final_loss" else f"{tick_value:.0f}"
            draw.text((x0 - 40, ty - 6), label, fill="black", font=font)

        group_width = panel_width / max(len(settings), 1)
        bar_width = max(8, int(group_width / (len(METHOD_ORDER) + 2)))
        for idx, setting in enumerate(settings):
            gx = x0 + int((idx + 0.5) * group_width)
            draw.text((gx - 48, y1 + 12), setting, fill="black", font=font)
            for m_idx, method in enumerate(METHOD_ORDER):
                metrics = setting_metrics[setting].get(method)
                if metrics is None:
                    continue
                val = float(metrics[metric_key])
                if np.isnan(val):
                    continue
                scale = val / max_values[metric_key] if max_values[metric_key] > 0 else 0.0
                bar_h = int(scale * (panel_height - 20))
                bx0 = int(gx - (len(METHOD_ORDER) * bar_width) / 2 + m_idx * bar_width)
                bx1 = bx0 + bar_width - 1
                by0 = y1 - bar_h
                draw.rectangle([bx0, by0, bx1, y1], fill=METHOD_COLORS[method], outline="black")

    image.save(png_path)
    image.save(pdf_path, "PDF")


if __name__ == "__main__":
    main()
