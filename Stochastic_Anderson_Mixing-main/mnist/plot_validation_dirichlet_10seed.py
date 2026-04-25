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
OUT_DIR = BASE_DIR / "validation_dirichlet_10seed_fig"


def paths(prefix: str):
    return [str(BASE_DIR / f"{prefix}{seed}.pkl") for seed in range(1, 11)]


METHOD_FILES = {
    "AsyncSAM+RMS": paths("validate10_dirichlet005_asyncsam_rms_seed"),
    "AsyncSAM": paths("validate10_dirichlet005_asyncsam_seed"),
    "AsyncSGD+RMS": paths("validate10_dirichlet005_asyncsgd_rms_seed"),
    "FedAsync": paths("validate10_dirichlet005_fedasync_seed"),
    "FedBuff": paths("validate10_dirichlet005_fedbuff_seed"),
    "FedAC": paths("validate10_dirichlet005_fedac_tuned_seed"),
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
    out = {"num_seeds": len(rows)}
    if not rows:
        for key in ("final_acc", "best_acc", "final_loss", "best_loss"):
            out[key] = float("nan")
            out[f"{key}_std"] = float("nan")
        return out
    for key in ("final_acc", "best_acc", "final_loss", "best_loss"):
        vals = np.asarray([row[key] for row in rows], dtype=float)
        out[key] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std())
    return out


def render_with_pil(metrics_by_method, png_path, pdf_path):
    width, height = 1700, 680
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
    finite_losses = [
        float(metrics_by_method[m]["final_loss"])
        for m in metrics_by_method
        if not np.isnan(metrics_by_method[m]["final_loss"])
    ]
    max_values = {
        "final_acc": 100.0,
        "best_acc": 100.0,
        "final_loss": (max(finite_losses) * 1.1) if finite_losses else 1.0,
    }

    draw.text((width // 2 - 245, 20), "MNIST Dirichlet Non-IID (alpha=0.05, min_size=10) - 10 Seeds", fill="black", font=title_font)

    legend_x, legend_y = margin_left, 50
    for method in METHOD_ORDER:
        color = METHOD_COLORS[method]
        draw.rectangle([legend_x, legend_y, legend_x + 14, legend_y + 14], fill=color, outline="black")
        draw.text((legend_x + 20, legend_y), method, fill="black", font=font)
        legend_x += 180

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
            draw.text((x0 - 45, ty - 6), label, fill="black", font=font)

        cluster_width = panel_width
        bar_width = cluster_width / (len(METHOD_ORDER) + 1)
        for method_idx, method in enumerate(METHOD_ORDER):
            value = metrics_by_method[method][metric_key]
            if np.isnan(value):
                continue
            normalized = value / max_values[metric_key] if max_values[metric_key] > 0 else 0.0
            normalized = max(0.0, min(1.0, normalized))
            bar_h = int(normalized * (panel_height - 20))
            bx0 = int(x0 + (method_idx + 0.5) * bar_width)
            bx1 = int(bx0 + bar_width - 10)
            by0 = y1 - bar_h
            draw.rectangle([bx0, by0, bx1, y1], fill=METHOD_COLORS[method], outline="black")
            draw.text((bx0 - 4, y1 + 10), method[:11], fill="black", font=font)

    image.save(png_path)
    image.save(pdf_path, "PDF", resolution=200.0)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    metrics_by_method = {}
    for method, file_list in METHOD_FILES.items():
        metrics = aggregate_metrics(file_list)
        metrics_by_method[method] = metrics
        row = {"method": method}
        row.update(metrics)
        row["paths"] = ";".join(file_list)
        rows.append(row)

    csv_path = OUT_DIR / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
                "paths",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    png_path = OUT_DIR / "multiseed_compare.png"
    pdf_path = OUT_DIR / "multiseed_compare.pdf"
    if plt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        metric_specs = [
            ("final_acc", "final_acc_std", "Final Accuracy (%)"),
            ("best_acc", "best_acc_std", "Best Accuracy (%)"),
            ("final_loss", "final_loss_std", "Final Test Loss"),
        ]
        x = np.arange(len(METHOD_ORDER))
        for ax, (metric_key, std_key, title) in zip(axes, metric_specs):
            vals = [metrics_by_method[m][metric_key] for m in METHOD_ORDER]
            errs = [metrics_by_method[m][std_key] for m in METHOD_ORDER]
            ax.bar(x, vals, yerr=errs, capsize=3, color=[METHOD_COLORS[m] for m in METHOD_ORDER], alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(METHOD_ORDER, rotation=20, ha="right")
            ax.set_title(title)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            if metric_key != "final_loss":
                ax.set_ylim(top=100)
        fig.suptitle("MNIST Dirichlet Non-IID (alpha=0.05, min_size=10) - 10 Seeds")
        fig.tight_layout()
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
    else:
        render_with_pil(metrics_by_method, png_path, pdf_path)

    print(csv_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
