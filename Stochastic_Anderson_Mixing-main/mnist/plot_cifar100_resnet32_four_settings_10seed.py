import csv
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path("/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist/experiments")
OUT_DIR = BASE_DIR / "cifar100_resnet32_four_settings_parallel_10seed_fig"

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
}

METHOD_COLORS = {
    "AsyncSAM+RMS": "#0b5d7a",
    "AsyncSGD": "#2b8a3e",
    "FedAsync": "#6c757d",
    "FedBuff": "#9467bd",
}


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def result_path(setting_key, method_key, seed):
    return BASE_DIR / f"cifar100_resnet32_p3_{setting_key}_{method_key}_seed{seed}.pkl"


def load_result(path):
    if (not path.exists()) or path.stat().st_size == 0:
        return None
    with path.open("rb") as f:
        data = pickle.load(f)
    return {
        "acc": np.asarray(data["test_prec"], dtype=float) * 100.0,
        "loss": np.asarray(data["test_loss"], dtype=float),
    }


def load_group(setting_key, method_key):
    results, missing = [], []
    for seed in range(1, 11):
        item = load_result(result_path(setting_key, method_key, seed))
        if item is None:
            missing.append(seed)
        else:
            results.append(item)
    return results, missing


def summarize(results):
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
    final_acc = np.asarray([r["acc"][-1] for r in results])
    best_acc = np.asarray([np.max(r["acc"]) for r in results])
    final_loss = np.asarray([r["loss"][-1] for r in results])
    best_loss = np.asarray([np.min(r["loss"]) for r in results])
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
    return np.arange(1, min_len + 1), stacked.mean(axis=0), stacked.std(axis=0)


def build_summary():
    rows = []
    all_results = {}
    for setting_name, setting_key in SETTINGS.items():
        all_results[setting_name] = {}
        for method_name, method_key in METHODS.items():
            results, missing = load_group(setting_key, method_key)
            all_results[setting_name][method_name] = results
            row = {"setting": setting_name, "method": method_name}
            row.update(summarize(results))
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


def draw_center(draw, x, y, text, font, fill="black"):
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y), text, font=font, fill=fill)


def save_image(image, png_path, pdf_path):
    image.save(png_path)
    image.save(pdf_path, "PDF", resolution=200.0)


def plot_summary(rows):
    width, height = 2200, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    setting_names = list(SETTINGS.keys())
    method_names = list(METHODS.keys())
    by_key = {(row["setting"], row["method"]): row for row in rows}
    finite_losses = [
        float(row["final_loss"])
        for row in rows
        if np.isfinite(float(row["final_loss"]))
    ]
    loss_ymax = 1.15 * max(finite_losses) if finite_losses else 1.0
    panels = [
        ("final_acc", "Final Accuracy (%)", 100.0),
        ("best_acc", "Best Accuracy (%)", 100.0),
        ("final_loss", "Final Test Loss", loss_ymax),
    ]

    draw_center(draw, width / 2, 20, "CIFAR-100 / ResNet32 Four Settings, 10 Seeds", font)
    legend_x, legend_y = 95, 54
    for method_name in method_names:
        color = hex_to_rgb(METHOD_COLORS[method_name])
        draw.rectangle([legend_x, legend_y, legend_x + 18, legend_y + 18], fill=color, outline="black")
        draw.text((legend_x + 25, legend_y + 2), method_name, font=font, fill="black")
        legend_x += 280

    margin_left, margin_top, margin_bottom, gap = 88, 105, 105, 42
    panel_width = (width - 2 * margin_left - 2 * gap) // 3
    panel_height = height - margin_top - margin_bottom
    for panel_idx, (metric, title, ymax) in enumerate(panels):
        x0 = margin_left + panel_idx * (panel_width + gap)
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
            draw_center(draw, center, y1 + 14, setting_name.replace(" Non-IID", ""), font)
            for method_idx, method_name in enumerate(method_names):
                value = float(by_key[(setting_name, method_name)][metric])
                if not np.isfinite(value):
                    continue
                bx0 = center - group_width / 2 + (method_idx + 0.6) * bar_width
                bx1 = bx0 + bar_width * 0.82
                by0 = y1 - max(0.0, min(1.0, value / ymax)) * panel_height
                draw.rectangle(
                    [int(bx0), int(by0), int(bx1), y1],
                    fill=hex_to_rgb(METHOD_COLORS[method_name]),
                    outline="black",
                )
    png_path = OUT_DIR / "cifar100_resnet32_four_settings_summary.png"
    pdf_path = OUT_DIR / "cifar100_resnet32_four_settings_summary.pdf"
    save_image(image, png_path, pdf_path)
    return png_path, pdf_path


def plot_curves(all_results):
    setting_names = list(SETTINGS.keys())
    method_names = list(METHODS.keys())
    width, height = 1900, 2050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw_center(draw, width / 2, 20, "CIFAR-100 / ResNet32 Mean Curves, 10 Seeds", font)
    legend_x, legend_y = 80, 54
    for method_name in method_names:
        color = hex_to_rgb(METHOD_COLORS[method_name])
        draw.line([legend_x, legend_y + 9, legend_x + 35, legend_y + 9], fill=color, width=5)
        draw.text((legend_x + 43, legend_y + 1), method_name, font=font, fill="black")
        legend_x += 280

    margin_left, margin_top, gap_x, gap_y = 80, 105, 80, 58
    panel_width = (width - 2 * margin_left - gap_x) // 2
    panel_height = (height - margin_top - 60 - 3 * gap_y) // 4
    for row_idx, setting_name in enumerate(setting_names):
        for col_idx, metric in enumerate(["acc", "loss"]):
            x0 = margin_left + col_idx * (panel_width + gap_x)
            y0 = margin_top + row_idx * (panel_height + gap_y)
            x1 = x0 + panel_width
            y1 = y0 + panel_height
            title = f"{setting_name} {'Accuracy' if metric == 'acc' else 'Loss'}"
            draw.rectangle([x0, y0, x1, y1], outline="black")
            draw.text((x0 + 8, y0 - 22), title, font=font, fill="black")

            curves = {}
            max_epoch = 1
            ymax = 100.0 if metric == "acc" else 0.0
            for method_name in method_names:
                epochs, mean, _ = curve_stats(all_results[setting_name][method_name], metric)
                if epochs is None:
                    continue
                curves[method_name] = (epochs, mean)
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
                draw_center(draw, tx, y1 + 8, f"{int(max_epoch * ratio):d}", font)

            for method_name, (epochs, mean) in curves.items():
                points = []
                for epoch, value in zip(epochs, mean):
                    px = x0 + (float(epoch) / max_epoch) * panel_width
                    py = y1 - max(0.0, min(1.0, float(value) / ymax)) * panel_height
                    points.append((int(px), int(py)))
                if len(points) >= 2:
                    draw.line(points, fill=hex_to_rgb(METHOD_COLORS[method_name]), width=3)
    png_path = OUT_DIR / "cifar100_resnet32_four_settings_curves.png"
    pdf_path = OUT_DIR / "cifar100_resnet32_four_settings_curves.pdf"
    save_image(image, png_path, pdf_path)
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
