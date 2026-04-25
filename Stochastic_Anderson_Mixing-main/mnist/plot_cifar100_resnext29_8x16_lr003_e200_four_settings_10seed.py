import csv
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(
    "/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist/experiments/by_family/cifar100/resnext29"
)
OUT_DIR = BASE_DIR / "resnext29_8x16_lr003_e200_four_settings_10seed_fig"

SETTINGS = {
    "Base IID": "base",
    "Label-Sorted Non-IID": "labelsorted",
    "Dirichlet Non-IID": "dirichlet",
    "High Delay IID": "highdelay",
}

METHODS = {
    "AsyncSAM+RMS": "asyncsam_rms",
    "AsyncSGD": "asyncsgd",
    # Use the tuned algorithm-specific server parameters while keeping the
    # public training protocol fixed across all methods.
    "FedAsync": "fedasync_tuned",
    "FedBuff": "fedbuff_tuned",
}

COLORS = {
    "AsyncSAM+RMS": "#0b5d7a",
    "AsyncSGD": "#2b8a3e",
    "FedAsync": "#6c757d",
    "FedBuff": "#9467bd",
}


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def result_path(setting_key, method_key, seed):
    return BASE_DIR / f"resnext29_8x16_lr003_e200_{setting_key}_{method_key}_seed{seed}.pkl"


def load_result(path):
    if not path.exists() or path.stat().st_size == 0:
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
    steps = np.arange(1, min_len + 1)
    return steps, stacked.mean(axis=0), stacked.std(axis=0)


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_key = {(row["setting"], row["method"]): row for row in rows}
    settings = list(SETTINGS.keys())
    methods = list(METHODS.keys())
    width, height = 2200, 780
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    panels = [
        ("final_acc", "final_acc_std", "Final Accuracy (%)", (62, 79)),
        ("best_acc", "best_acc_std", "Best Accuracy (%)", (62, 79)),
        ("final_loss", "final_loss_std", "Final Test Loss", (0.9, 1.35)),
    ]

    draw_center(draw, width / 2, 22, "CIFAR-100 / ResNeXt29-8x16d / 200 Epochs / 10 Seeds", font)
    legend_x, legend_y = 95, 58
    for method in methods:
        draw.rectangle(
            [legend_x, legend_y, legend_x + 18, legend_y + 18],
            fill=hex_to_rgb(COLORS[method]),
            outline="black",
        )
        draw.text((legend_x + 25, legend_y + 2), method, font=font, fill="black")
        legend_x += 280

    margin_left, margin_top, margin_bottom, gap = 88, 112, 105, 42
    panel_width = (width - 2 * margin_left - 2 * gap) // 3
    panel_height = height - margin_top - margin_bottom
    for panel_idx, (metric, std_metric, title, ylim) in enumerate(panels):
        ymin, ymax = ylim
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
            value = ymin + (ymax - ymin) * ratio
            label = f"{value:.2f}" if metric == "final_loss" else f"{value:.0f}"
            draw.text((x0 - 58, ty - 6), label, font=font, fill="black")
        group_width = panel_width / len(settings)
        bar_width = group_width / (len(methods) + 1)
        for setting_idx, setting in enumerate(settings):
            center = x0 + (setting_idx + 0.5) * group_width
            draw_center(
                draw,
                center,
                y1 + 14,
                setting.replace(" Non-IID", "").replace("Base IID", "Base").replace("High Delay IID", "High Delay"),
                font,
            )
            for method_idx, method in enumerate(methods):
                value = float(by_key[(setting, method)][metric])
                error = float(by_key[(setting, method)][std_metric])
                if not np.isfinite(value):
                    continue
                clipped = max(ymin, min(ymax, value))
                by0 = y1 - (clipped - ymin) / (ymax - ymin) * panel_height
                bx0 = center - group_width / 2 + (method_idx + 0.6) * bar_width
                bx1 = bx0 + bar_width * 0.82
                draw.rectangle(
                    [int(bx0), int(by0), int(bx1), y1],
                    fill=hex_to_rgb(COLORS[method]),
                    outline="black",
                )
                err_low = max(ymin, value - error)
                err_high = min(ymax, value + error)
                ey_low = y1 - (err_low - ymin) / (ymax - ymin) * panel_height
                ey_high = y1 - (err_high - ymin) / (ymax - ymin) * panel_height
                ex = int((bx0 + bx1) / 2)
                draw.line([ex, int(ey_low), ex, int(ey_high)], fill="black", width=1)
                draw.line([ex - 4, int(ey_low), ex + 4, int(ey_low)], fill="black", width=1)
                draw.line([ex - 4, int(ey_high), ex + 4, int(ey_high)], fill="black", width=1)

    png_path = OUT_DIR / "resnext29_8x16_lr003_e200_four_settings_summary.png"
    pdf_path = OUT_DIR / "resnext29_8x16_lr003_e200_four_settings_summary.pdf"
    save_image(image, png_path, pdf_path)
    return png_path, pdf_path


def plot_curves(all_results):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    settings = list(SETTINGS.keys())
    methods = list(METHODS.keys())
    width, height = 1900, 2050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw_center(draw, width / 2, 22, "CIFAR-100 / ResNeXt29-8x16d Mean Curves, 10 Seeds", font)
    legend_x, legend_y = 80, 58
    for method in methods:
        color = hex_to_rgb(COLORS[method])
        draw.line([legend_x, legend_y + 9, legend_x + 35, legend_y + 9], fill=color, width=5)
        draw.text((legend_x + 43, legend_y + 1), method, font=font, fill="black")
        legend_x += 280

    margin_left, margin_top, gap_x, gap_y = 82, 112, 80, 58
    panel_width = (width - 2 * margin_left - gap_x) // 2
    panel_height = (height - margin_top - 60 - 3 * gap_y) // 4
    for row_idx, setting in enumerate(settings):
        for col_idx, (metric, ylabel, ylim) in enumerate(
            [("acc", "Accuracy (%)", (45.0, 80.0)), ("loss", "Test Loss", (0.9, 2.4))]
        ):
            ymin, ymax = ylim
            x0 = margin_left + col_idx * (panel_width + gap_x)
            y0 = margin_top + row_idx * (panel_height + gap_y)
            x1 = x0 + panel_width
            y1 = y0 + panel_height
            draw.rectangle([x0, y0, x1, y1], outline="black")
            draw.text((x0 + 8, y0 - 24), f"{setting} - {ylabel}", font=font, fill="black")
            for tick in range(6):
                ratio = tick / 5.0
                ty = y1 - int(ratio * panel_height)
                draw.line([x0, ty, x1, ty], fill="#dddddd")
                value = ymin + (ymax - ymin) * ratio
                label = f"{value:.1f}" if metric == "loss" else f"{value:.0f}"
                draw.text((x0 - 52, ty - 6), label, font=font, fill="black")
            for tick in range(5):
                ratio = tick / 4.0
                tx = x0 + int(ratio * panel_width)
                epoch = int(1 + ratio * 199)
                draw.line([tx, y1, tx, y1 + 4], fill="black")
                draw_center(draw, tx, y1 + 8, str(epoch), font)
            for method in methods:
                steps, mean, std = curve_stats(all_results[setting][method], metric)
                if steps is None:
                    continue
                points = []
                for step, value in zip(steps, mean):
                    px = x0 + (step - 1) / max(1, steps[-1] - 1) * panel_width
                    clipped = max(ymin, min(ymax, float(value)))
                    py = y1 - (clipped - ymin) / (ymax - ymin) * panel_height
                    points.append((int(px), int(py)))
                if len(points) >= 2:
                    draw.line(points, fill=hex_to_rgb(COLORS[method]), width=3)

    png_path = OUT_DIR / "resnext29_8x16_lr003_e200_four_settings_curves.png"
    pdf_path = OUT_DIR / "resnext29_8x16_lr003_e200_four_settings_curves.pdf"
    save_image(image, png_path, pdf_path)
    return png_path, pdf_path


def main():
    rows, all_results = build_summary()
    csv_path = write_summary(rows)
    summary_png, summary_pdf = plot_summary(rows)
    curves_png, curves_pdf = plot_curves(all_results)
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_png}")
    print(f"Wrote {summary_pdf}")
    print(f"Wrote {curves_png}")
    print(f"Wrote {curves_pdf}")


if __name__ == "__main__":
    main()
