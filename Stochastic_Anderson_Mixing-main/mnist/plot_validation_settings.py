import csv
import os
import pickle

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


BASE_DIR = "/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist/experiments"
OUT_DIR = os.path.join(BASE_DIR, "validation_other_settings_fig")


SETTING_FILES = {
    "Base IID": {
        "AsyncSAM RMS+AA": "asyncsam_precond_q3_rms90_i1_batch0_seed1.pkl",
        "AsyncSGD": "asyncsgd_iid_e60_npu_lr001_seed1.pkl",
        "AsyncSGD+RMS": "ablate_asyncsgd_rms_seed1.pkl",
        "FedBuff": "afl_same_setting_fedbuff_seed1.pkl",
        "AsyncAA+RMS": "ablate_asyncaa_rms_seed1.pkl",
    },
    "Label-Sorted Non-IID": {
        "AsyncSAM RMS+AA": "validate_labelsorted_asyncsam_seed1.pkl",
        "AsyncSGD": "validate_labelsorted_asyncsgd_seed1.pkl",
        "AsyncSGD+RMS": "validate_labelsorted_asyncsgd_rms_seed1.pkl",
        "FedBuff": "validate_labelsorted_fedbuff_seed1.pkl",
        "AsyncAA+RMS": "validate_labelsorted_asyncaa_rms_seed1.pkl",
    },
    "High Delay IID": {
        "AsyncSAM RMS+AA": "validate_highdelay_asyncsam_seed1.pkl",
        "AsyncSGD+RMS": "validate_highdelay_asyncsgd_rms_seed1.pkl",
        "FedBuff": "validate_highdelay_fedbuff_seed1.pkl",
    },
    "IID 20 Workers": {
        "AsyncSAM RMS+AA": "validate_workers20_asyncsam_mix02_seed1.pkl",
        "AsyncSGD": "validate_workers20_asyncsgd_seed1.pkl",
        "AsyncSGD+RMS": "validate_workers20_asyncsgd_rms_seed1.pkl",
        "FedBuff": "validate_workers20_fedbuff_seed1.pkl",
        "AsyncAA+RMS": "validate_workers20_asyncaa_rms_seed1.pkl",
    },
}


METHOD_ORDER = [
    "AsyncSAM RMS+AA",
    "AsyncSGD+RMS",
    "AsyncAA+RMS",
    "FedBuff",
    "AsyncSGD",
]

METHOD_COLORS = {
    "AsyncSAM RMS+AA": "#1f77b4",
    "AsyncSGD+RMS": "#2ca02c",
    "AsyncAA+RMS": "#ff7f0e",
    "FedBuff": "#9467bd",
    "AsyncSGD": "#7f7f7f",
}


def load_metrics(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return {
        "final_acc": 100.0 * data["test_prec"][-1],
        "best_acc": 100.0 * max(data["test_prec"]),
        "final_loss": data["test_loss"][-1],
        "best_loss": min(data["test_loss"]),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    setting_metrics = {}
    for setting, method_map in SETTING_FILES.items():
        setting_metrics[setting] = {}
        for method, filename in method_map.items():
            metrics = load_metrics(os.path.join(BASE_DIR, filename))
            setting_metrics[setting][method] = metrics
            row = {"setting": setting, "method": method}
            row.update(metrics)
            rows.append(row)

    csv_path = os.path.join(OUT_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["setting", "method", "final_acc", "best_acc", "final_loss", "best_loss"],
        )
        writer.writeheader()
        writer.writerows(rows)

    settings = list(SETTING_FILES.keys())
    x = np.arange(len(settings))
    width = 0.16
    offsets = np.linspace(-2, 2, len(METHOD_ORDER)) * width

    png_path = os.path.join(OUT_DIR, "validation_settings_compare.png")
    pdf_path = os.path.join(OUT_DIR, "validation_settings_compare.pdf")
    if plt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        metric_specs = [
            ("final_acc", "Final Accuracy (%)"),
            ("best_acc", "Best Accuracy (%)"),
            ("final_loss", "Final Test Loss"),
        ]

        for ax, (metric_key, metric_label) in zip(axes, metric_specs):
            for offset, method in zip(offsets, METHOD_ORDER):
                vals = []
                for setting in settings:
                    metrics = setting_metrics[setting].get(method)
                    vals.append(np.nan if metrics is None else metrics[metric_key])
                ax.bar(
                    x + offset,
                    vals,
                    width=width,
                    label=method,
                    color=METHOD_COLORS[method],
                    alpha=0.9,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(settings, rotation=15, ha="right")
            ax.set_ylabel(metric_label)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        axes[0].set_title("Validation Across Other Settings")
        axes[1].set_title("Validation Across Other Settings")
        axes[2].set_title("Validation Across Other Settings")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.06))
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
    else:
        render_with_pil(setting_metrics, settings, png_path)
    print(csv_path)
    print(png_path)
    if plt is not None:
        print(pdf_path)


def render_with_pil(setting_metrics, settings, png_path):
    width, height = 1800, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    margin_left = 70
    margin_right = 30
    margin_top = 80
    margin_bottom = 100
    panel_gap = 25
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
            setting_metrics[s][m]["final_loss"]
            for s in settings
            for m in setting_metrics[s]
        )
        * 1.1,
    }

    draw.text((width // 2 - 150, 20), "Validation Across Other Settings", fill="black", font=title_font)

    legend_x, legend_y = margin_left, 45
    for method in METHOD_ORDER:
        color = METHOD_COLORS[method]
        draw.rectangle([legend_x, legend_y, legend_x + 12, legend_y + 12], fill=color)
        draw.text((legend_x + 18, legend_y), method, fill="black", font=font)
        legend_x += 210

    for panel_idx, (metric_key, metric_label) in enumerate(metric_specs):
        x0 = margin_left + panel_idx * (panel_width + panel_gap)
        y0 = margin_top
        x1 = x0 + panel_width
        y1 = y0 + panel_height
        draw.rectangle([x0, y0, x1, y1], outline="black", width=1)
        draw.text((x0 + 8, y0 - 18), metric_label, fill="black", font=font)

        group_width = panel_width / max(len(settings), 1)
        bar_width = max(6, int(group_width / (len(METHOD_ORDER) + 2)))
        for idx, setting in enumerate(settings):
            gx = x0 + int((idx + 0.5) * group_width)
            draw.text((gx - 35, y1 + 10), setting, fill="black", font=font)
            for m_idx, method in enumerate(METHOD_ORDER):
                metrics = setting_metrics[setting].get(method)
                if metrics is None:
                    continue
                val = metrics[metric_key]
                scale = val / max_values[metric_key] if max_values[metric_key] > 0 else 0.0
                bar_h = int(scale * (panel_height - 20))
                bx0 = int(gx - (len(METHOD_ORDER) * bar_width) / 2 + m_idx * bar_width)
                bx1 = bx0 + bar_width - 1
                by0 = y1 - bar_h
                draw.rectangle([bx0, by0, bx1, y1], fill=METHOD_COLORS[method])
        for tick in range(5):
            ratio = tick / 4.0
            ty = y1 - int(ratio * (panel_height - 20))
            draw.line([x0, ty, x1, ty], fill="#dddddd", width=1)
            tick_value = max_values[metric_key] * ratio
            label = f"{tick_value:.2f}" if metric_key == "final_loss" else f"{tick_value:.0f}"
            draw.text((x0 - 36, ty - 6), label, fill="black", font=font)

    image.save(png_path)


if __name__ == "__main__":
    main()
