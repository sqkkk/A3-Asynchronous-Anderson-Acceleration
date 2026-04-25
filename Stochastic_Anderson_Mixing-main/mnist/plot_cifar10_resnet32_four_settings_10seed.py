import csv
import pickle
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path("/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist/experiments")
OUT_DIR = BASE_DIR / "cifar10_resnet32_four_settings_fixed_public_10seed_fig"

SETTINGS = {
    "Base IID": "base",
    "Label-Sorted Non-IID": "labelsorted",
    "Dirichlet Non-IID": "dirichlet",
    "High Delay IID": "highdelay",
}

METHODS = {
    "AsyncSAM+RMS": "asyncsam_rms",
    "AsyncSGD": "asyncsgd",
    "FedAsync": "fedasync",
    "FedBuff": "fedbuff",
    "FedAC": "fedac",
}

METHOD_COLORS = {
    "AsyncSAM+RMS": "#0b5d7a",
    "AsyncSGD": "#2b8a3e",
    "FedAsync": "#6c757d",
    "FedBuff": "#9467bd",
    "FedAC": "#8c564b",
}


def hex_to_rgb(color: str):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def result_path(setting_key: str, method_key: str, seed: int) -> Path:
    return BASE_DIR / f"cifar10_resnet32_p3_{setting_key}_{method_key}_seed{seed}.pkl"


def load_result(path: Path):
    if (not path.exists()) or path.stat().st_size == 0:
        return None
    with path.open("rb") as f:
        data = pickle.load(f)
    acc = np.asarray(data["test_prec"], dtype=float) * 100.0
    loss = np.asarray(data["test_loss"], dtype=float)
    return {"acc": acc, "loss": loss}


def load_group(setting_key: str, method_key: str):
    results = []
    missing = []
    for seed in range(1, 11):
        item = load_result(result_path(setting_key, method_key, seed))
        if item is None:
            missing.append(seed)
        else:
            results.append(item)
    return results, missing


def summarize_group(results):
    if not results:
        return {
            "num_seeds": 0,
            "final_acc": np.nan,
            "final_acc_std": np.nan,
            "best_acc": np.nan,
            "best_acc_std": np.nan,
            "final_loss": np.nan,
            "final_loss_std": np.nan,
            "best_loss": np.nan,
            "best_loss_std": np.nan,
        }
    final_acc = np.asarray([r["acc"][-1] for r in results], dtype=float)
    best_acc = np.asarray([np.max(r["acc"]) for r in results], dtype=float)
    final_loss = np.asarray([r["loss"][-1] for r in results], dtype=float)
    best_loss = np.asarray([np.min(r["loss"]) for r in results], dtype=float)
    return {
        "num_seeds": len(results),
        "final_acc": float(final_acc.mean()),
        "final_acc_std": float(final_acc.std()),
        "best_acc": float(best_acc.mean()),
        "best_acc_std": float(best_acc.std()),
        "final_loss": float(final_loss.mean()),
        "final_loss_std": float(final_loss.std()),
        "best_loss": float(best_loss.mean()),
        "best_loss_std": float(best_loss.std()),
    }


def curve_stats(results, metric):
    if not results:
        return None, None, None
    min_len = min(len(r[metric]) for r in results)
    stacked = np.stack([r[metric][:min_len] for r in results], axis=0)
    epochs = np.arange(1, min_len + 1)
    return epochs, stacked.mean(axis=0), stacked.std(axis=0)


def build_summary():
    all_results = {}
    rows = []
    for setting_name, setting_key in SETTINGS.items():
        all_results[setting_name] = {}
        for method_name, method_key in METHODS.items():
            results, missing = load_group(setting_key, method_key)
            all_results[setting_name][method_name] = results
            row = {"setting": setting_name, "method": method_name}
            row.update(summarize_group(results))
            row["missing_seeds"] = " ".join(map(str, missing))
            rows.append(row)
    return rows, all_results


def write_summary(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "summary.csv"
    fieldnames = [
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
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def plot_summary(rows):
    if plt is None:
        return plot_summary_pil(rows)

    metrics = [
        ("final_acc", "final_acc_std", "Final Accuracy (%)", (0, 100)),
        ("best_acc", "best_acc_std", "Best Accuracy (%)", (0, 100)),
        ("final_loss", "final_loss_std", "Final Test Loss", None),
    ]
    setting_names = list(SETTINGS.keys())
    method_names = list(METHODS.keys())
    by_key = {(row["setting"], row["method"]): row for row in rows}
    x = np.arange(len(setting_names))
    bar_width = 0.82 / len(method_names)
    offsets = (np.arange(len(method_names)) - (len(method_names) - 1) / 2) * bar_width

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, (metric, std_metric, title, ylim) in zip(axes, metrics):
        for offset, method_name in zip(offsets, method_names):
            values = [float(by_key[(setting, method_name)][metric]) for setting in setting_names]
            errors = [float(by_key[(setting, method_name)][std_metric]) for setting in setting_names]
            ax.bar(
                x + offset,
                values,
                width=bar_width,
                yerr=errors,
                capsize=3,
                color=METHOD_COLORS[method_name],
                label=method_name,
                alpha=0.92,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(setting_names, rotation=18, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(method_names), loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("CIFAR-10 / ResNet32 Four Settings, 10 Seeds", y=1.12, fontsize=14)
    fig.tight_layout()
    png_path = OUT_DIR / "resnet32_four_settings_summary.png"
    pdf_path = OUT_DIR / "resnet32_four_settings_summary.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_curves(all_results):
    if plt is None:
        return plot_curves_pil(all_results)

    setting_names = list(SETTINGS.keys())
    method_names = list(METHODS.keys())
    fig, axes = plt.subplots(len(setting_names), 2, figsize=(13.5, 13.5), sharex=False)
    for row_idx, setting_name in enumerate(setting_names):
        acc_ax = axes[row_idx, 0]
        loss_ax = axes[row_idx, 1]
        for method_name in method_names:
            results = all_results[setting_name][method_name]
            epochs, acc_mean, acc_std = curve_stats(results, "acc")
            _, loss_mean, loss_std = curve_stats(results, "loss")
            if epochs is None:
                continue
            color = METHOD_COLORS[method_name]
            acc_ax.plot(epochs, acc_mean, color=color, label=method_name, linewidth=1.8)
            acc_ax.fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std, color=color, alpha=0.10, linewidth=0)
            loss_ax.plot(epochs, loss_mean, color=color, label=method_name, linewidth=1.8)
            loss_ax.fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std, color=color, alpha=0.10, linewidth=0)
        acc_ax.set_title(f"{setting_name} Accuracy")
        loss_ax.set_title(f"{setting_name} Loss")
        acc_ax.set_ylabel("Accuracy (%)")
        loss_ax.set_ylabel("Test Loss")
        acc_ax.set_ylim(0, 100)
        acc_ax.grid(True, linestyle="--", alpha=0.3)
        loss_ax.grid(True, linestyle="--", alpha=0.3)
        if row_idx == len(setting_names) - 1:
            acc_ax.set_xlabel("Epoch")
            loss_ax.set_xlabel("Epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(method_names), loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    png_path = OUT_DIR / "resnet32_four_settings_curves.png"
    pdf_path = OUT_DIR / "resnet32_four_settings_curves.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def draw_text_centered(draw, xy, text, font, fill="black"):
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y), text, font=font, fill=fill)


def save_pil(image, png_path, pdf_path):
    image.save(png_path)
    image.save(pdf_path, "PDF", resolution=200.0)


def plot_summary_pil(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 2200, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    setting_names = list(SETTINGS.keys())
    method_names = list(METHODS.keys())
    by_key = {(row["setting"], row["method"]): row for row in rows}
    panels = [
        ("final_acc", "Final Accuracy (%)", 100.0),
        ("best_acc", "Best Accuracy (%)", 100.0),
        (
            "final_loss",
            "Final Test Loss",
            1.15
            * max(
                float(row["final_loss"])
                for row in rows
                if row["final_loss"] == row["final_loss"]
            ),
        ),
    ]

    draw_text_centered(draw, (width / 2, 20), "CIFAR-10 / ResNet32 Four Settings, 10 Seeds", title_font)
    legend_x, legend_y = 95, 54
    for method_name in method_names:
        color = hex_to_rgb(METHOD_COLORS[method_name])
        draw.rectangle([legend_x, legend_y, legend_x + 18, legend_y + 18], fill=color, outline="black")
        draw.text((legend_x + 25, legend_y + 2), method_name, font=font, fill="black")
        legend_x += 245

    margin_left, margin_top, margin_bottom, panel_gap = 88, 105, 105, 42
    panel_width = (width - 2 * margin_left - 2 * panel_gap) // 3
    panel_height = height - margin_top - margin_bottom

    for panel_idx, (metric, title, ymax) in enumerate(panels):
        x0 = margin_left + panel_idx * (panel_width + panel_gap)
        y0 = margin_top
        x1 = x0 + panel_width
        y1 = y0 + panel_height
        draw.rectangle([x0, y0, x1, y1], outline="black")
        draw.text((x0 + 8, y0 - 24), title, font=font, fill="black")
        for tick in range(6):
            ratio = tick / 5.0
            ty = y1 - int(ratio * panel_height)
            draw.line([x0, ty, x1, ty], fill="#dddddd")
            value = ymax * ratio
            label = f"{value:.1f}" if metric == "final_loss" else f"{value:.0f}"
            draw.text((x0 - 50, ty - 6), label, font=font, fill="black")

        group_width = panel_width / len(setting_names)
        bar_width = group_width / (len(method_names) + 1)
        for setting_idx, setting_name in enumerate(setting_names):
            center = x0 + (setting_idx + 0.5) * group_width
            draw_text_centered(draw, (center, y1 + 14), setting_name.replace(" Non-IID", ""), font)
            for method_idx, method_name in enumerate(method_names):
                row = by_key[(setting_name, method_name)]
                value = float(row[metric])
                if not np.isfinite(value):
                    continue
                bx0 = center - group_width / 2 + (method_idx + 0.55) * bar_width
                bx1 = bx0 + bar_width * 0.78
                bar_height = max(0, min(1, value / ymax)) * panel_height
                by0 = y1 - bar_height
                draw.rectangle(
                    [int(bx0), int(by0), int(bx1), y1],
                    fill=hex_to_rgb(METHOD_COLORS[method_name]),
                    outline="black",
                )

    png_path = OUT_DIR / "resnet32_four_settings_summary.png"
    pdf_path = OUT_DIR / "resnet32_four_settings_summary.pdf"
    save_pil(image, png_path, pdf_path)
    return png_path, pdf_path


def panel_line(draw, points, color, width=3):
    if len(points) >= 2:
        draw.line([(int(x), int(y)) for x, y in points], fill=color, width=width, joint="curve")


def plot_curves_pil(all_results):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setting_names = list(SETTINGS.keys())
    method_names = list(METHODS.keys())
    width, height = 1900, 2050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    draw_text_centered(draw, (width / 2, 20), "CIFAR-10 / ResNet32 Mean Curves, 10 Seeds", title_font)
    legend_x, legend_y = 80, 54
    for method_name in method_names:
        color = hex_to_rgb(METHOD_COLORS[method_name])
        draw.line([legend_x, legend_y + 9, legend_x + 35, legend_y + 9], fill=color, width=5)
        draw.text((legend_x + 43, legend_y + 1), method_name, font=font, fill="black")
        legend_x += 250

    margin_left, margin_top, panel_gap_x, panel_gap_y = 80, 105, 80, 58
    panel_width = (width - 2 * margin_left - panel_gap_x) // 2
    panel_height = (height - margin_top - 60 - 3 * panel_gap_y) // 4

    for row_idx, setting_name in enumerate(setting_names):
        for col_idx, metric in enumerate(["acc", "loss"]):
            x0 = margin_left + col_idx * (panel_width + panel_gap_x)
            y0 = margin_top + row_idx * (panel_height + panel_gap_y)
            x1 = x0 + panel_width
            y1 = y0 + panel_height
            title = f"{setting_name} {'Accuracy' if metric == 'acc' else 'Loss'}"
            draw.rectangle([x0, y0, x1, y1], outline="black")
            draw.text((x0 + 8, y0 - 22), title, font=font, fill="black")

            method_curves = {}
            max_epoch = 1
            ymax = 100.0 if metric == "acc" else 0.0
            for method_name in method_names:
                epochs, mean, std = curve_stats(all_results[setting_name][method_name], metric)
                if epochs is None:
                    continue
                method_curves[method_name] = (epochs, mean)
                max_epoch = max(max_epoch, int(epochs[-1]))
                if metric == "loss":
                    ymax = max(ymax, float(np.nanmax(mean)))
            if metric == "loss":
                ymax = max(0.1, ymax * 1.12)

            for tick in range(6):
                ratio = tick / 5.0
                ty = y1 - int(ratio * panel_height)
                draw.line([x0, ty, x1, ty], fill="#e0e0e0")
                value = ymax * ratio
                label = f"{value:.1f}" if metric == "loss" else f"{value:.0f}"
                draw.text((x0 - 48, ty - 6), label, font=font, fill="black")
            for tick in range(5):
                ratio = tick / 4.0
                tx = x0 + int(ratio * panel_width)
                draw.line([tx, y0, tx, y1], fill="#f0f0f0")
                epoch_label = f"{int(max_epoch * ratio):d}"
                draw_text_centered(draw, (tx, y1 + 8), epoch_label, font)

            for method_name, (epochs, mean) in method_curves.items():
                color = hex_to_rgb(METHOD_COLORS[method_name])
                points = []
                for epoch, value in zip(epochs, mean):
                    px = x0 + (float(epoch) / max_epoch) * panel_width
                    py = y1 - max(0.0, min(1.0, float(value) / ymax)) * panel_height
                    points.append((px, py))
                panel_line(draw, points, color)

    png_path = OUT_DIR / "resnet32_four_settings_curves.png"
    pdf_path = OUT_DIR / "resnet32_four_settings_curves.pdf"
    save_pil(image, png_path, pdf_path)
    return png_path, pdf_path


def main():
    rows, all_results = build_summary()
    csv_path = write_summary(rows)
    summary_png, summary_pdf = plot_summary(rows)
    curves_png, curves_pdf = plot_curves(all_results)
    print(csv_path)
    print(summary_png)
    print(summary_pdf)
    print(curves_png)
    print(curves_pdf)


if __name__ == "__main__":
    main()
