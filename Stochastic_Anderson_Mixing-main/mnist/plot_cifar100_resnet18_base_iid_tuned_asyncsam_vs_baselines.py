import csv
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path("/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist/experiments")
OUT_DIR = BASE_DIR / "cifar100_resnet18_base_iid_tuned_asyncsam_vs_baselines_fig"

METHODS = {
    "AsyncSAM+RMS (Tuned)": {
        "kind": "tuned",
        "color": "#0b5d7a",
    },
    "AsyncSGD": {
        "kind": "baseline",
        "key": "asyncsgd",
        "color": "#2b8a3e",
    },
    "FedAsync": {
        "kind": "baseline",
        "key": "fedasync",
        "color": "#6c757d",
    },
    "FedBuff": {
        "kind": "baseline",
        "key": "fedbuff",
        "color": "#9467bd",
    },
}


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def load_result(path):
    if (not path.exists()) or path.stat().st_size == 0:
        return None
    with path.open("rb") as f:
        data = pickle.load(f)
    return {
        "acc": np.asarray(data["test_prec"], dtype=float) * 100.0,
        "loss": np.asarray(data["test_loss"], dtype=float),
    }


def result_path(method_name, seed):
    info = METHODS[method_name]
    if info["kind"] == "tuned":
        return BASE_DIR / f"cifar100_tune_r18_e300_sam_mind020_asyncsam_rms_seed{seed}.pkl"
    return BASE_DIR / f"cifar100_resnet18_e300_ra_re_ls_base_{info['key']}_seed{seed}.pkl"


def load_group(method_name):
    results = []
    missing = []
    for seed in range(1, 11):
        item = load_result(result_path(method_name, seed))
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
        return None, None
    min_len = min(len(r[metric]) for r in results)
    stacked = np.stack([r[metric][:min_len] for r in results], axis=0)
    return np.arange(1, min_len + 1), stacked.mean(axis=0)


def build_rows():
    rows = []
    all_results = {}
    for method_name in METHODS:
        results, missing = load_group(method_name)
        all_results[method_name] = results
        row = {"method": method_name}
        row.update(summarize(results))
        row["missing_seeds"] = " ".join(map(str, missing))
        rows.append(row)
    return rows, all_results


def write_summary(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "summary.csv"
    fieldnames = [
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
    width, height = 1900, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw_center(
        draw,
        width / 2,
        18,
        "CIFAR-100 / ResNet18 / Base IID: Tuned AsyncSAM+RMS vs Original Baselines",
        font,
    )
    draw_center(
        draw,
        width / 2,
        42,
        "AsyncSAM+RMS uses the tuned mind020 recipe; AsyncSGD / FedAsync / FedBuff use the original shared-protocol 10-seed runs.",
        font,
    )

    legend_x, legend_y = 160, 74
    for method_name, info in METHODS.items():
        draw.rectangle(
            [legend_x, legend_y, legend_x + 18, legend_y + 18],
            fill=hex_to_rgb(info["color"]),
            outline="black",
        )
        draw.text((legend_x + 26, legend_y + 2), method_name, font=font, fill="black")
        legend_x += 320 if "AsyncSAM" in method_name else 220

    metrics = [
        ("final_acc", "Final Accuracy (%)"),
        ("best_acc", "Best Accuracy (%)"),
        ("final_loss", "Final Test Loss"),
    ]
    finite_acc = [float(r["best_acc"]) for r in rows if np.isfinite(float(r["best_acc"]))]
    finite_loss = [float(r["final_loss"]) for r in rows if np.isfinite(float(r["final_loss"]))]
    acc_min = np.floor((min(finite_acc) - 1.2) * 2.0) / 2.0
    acc_max = np.ceil((max(finite_acc) + 0.6) * 2.0) / 2.0
    loss_min = np.floor((min(finite_loss) - 0.08) * 20.0) / 20.0
    loss_max = np.ceil((max(finite_loss) + 0.08) * 20.0) / 20.0
    limits = {
        "final_acc": (acc_min, acc_max),
        "best_acc": (acc_min, acc_max),
        "final_loss": (loss_min, loss_max),
    }

    margin_left, margin_top, margin_bottom, gap = 90, 120, 105, 45
    panel_width = (width - 2 * margin_left - 2 * gap) // 3
    panel_height = height - margin_top - margin_bottom
    for panel_idx, (metric, title) in enumerate(metrics):
        x0 = margin_left + panel_idx * (panel_width + gap)
        y0 = margin_top
        x1 = x0 + panel_width
        y1 = y0 + panel_height
        axis_min, axis_max = limits[metric]
        draw.rectangle([x0, y0, x1, y1], outline="black")
        draw.text((x0 + 8, y0 - 24), title, font=font, fill="black")
        for tick in range(6):
            ratio = tick / 5.0
            ty = y1 - int(ratio * panel_height)
            draw.line([x0, ty, x1, ty], fill="#e0e0e0")
            value = axis_min + (axis_max - axis_min) * ratio
            label = f"{value:.2f}" if "loss" in metric else f"{value:.1f}"
            draw.text((x0 - 45, ty - 6), label, font=font, fill="black")
        group_width = panel_width / len(rows)
        bar_width = group_width / 2.2
        for idx, row in enumerate(rows):
            center = x0 + (idx + 0.5) * group_width
            draw_center(draw, center, y1 + 12, row["method"].replace("AsyncSAM+RMS ", "SAM+RMS "), font)
            value = float(row[metric])
            std = float(row[f"{metric}_std"])
            ratio = (value - axis_min) / (axis_max - axis_min)
            ratio = max(0.0, min(1.0, ratio))
            bx0 = center - bar_width / 2
            bx1 = center + bar_width / 2
            by0 = y1 - ratio * panel_height
            draw.rectangle(
                [int(bx0), int(by0), int(bx1), y1],
                fill=hex_to_rgb(METHODS[row["method"]]["color"]),
                outline="black",
            )
            std_low = max(axis_min, value - std)
            std_high = min(axis_max, value + std)
            py_low = y1 - ((std_low - axis_min) / (axis_max - axis_min)) * panel_height
            py_high = y1 - ((std_high - axis_min) / (axis_max - axis_min)) * panel_height
            draw.line([int(center), int(py_high), int(center), int(py_low)], fill="black", width=1)
            draw.line([int(center - 6), int(py_high), int(center + 6), int(py_high)], fill="black", width=1)
            draw.line([int(center - 6), int(py_low), int(center + 6), int(py_low)], fill="black", width=1)

    png_path = OUT_DIR / "cifar100_resnet18_base_iid_tuned_asyncsam_vs_baselines_summary.png"
    pdf_path = OUT_DIR / "cifar100_resnet18_base_iid_tuned_asyncsam_vs_baselines_summary.pdf"
    save_image(image, png_path, pdf_path)
    return png_path, pdf_path


def plot_curves(all_results):
    width, height = 1820, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw_center(
        draw,
        width / 2,
        18,
        "CIFAR-100 / ResNet18 / Base IID Mean Curves (10 Seeds)",
        font,
    )
    draw_center(
        draw,
        width / 2,
        42,
        "Tuned AsyncSAM+RMS is compared against the original baseline runs on the same IID task.",
        font,
    )

    legend_x, legend_y = 160, 74
    for method_name, info in METHODS.items():
        draw.line([legend_x, legend_y + 9, legend_x + 35, legend_y + 9], fill=hex_to_rgb(info["color"]), width=4)
        draw.text((legend_x + 45, legend_y + 2), method_name, font=font, fill="black")
        legend_x += 320 if "AsyncSAM" in method_name else 220

    panels = [
        ("acc", "Test Accuracy (%)"),
        ("loss", "Test Loss"),
    ]
    margin_left, margin_top, margin_bottom, gap = 90, 120, 95, 70
    panel_width = (width - 2 * margin_left - gap) // 2
    panel_height = height - margin_top - margin_bottom

    acc_curves = {}
    loss_curves = {}
    max_epoch = 1
    loss_max = 0.0
    loss_min = float("inf")
    for method_name in METHODS:
        epochs, mean_acc = curve_stats(all_results[method_name], "acc")
        _, mean_loss = curve_stats(all_results[method_name], "loss")
        if epochs is None:
            continue
        acc_curves[method_name] = (epochs, mean_acc)
        loss_curves[method_name] = (epochs, mean_loss)
        max_epoch = max(max_epoch, int(epochs[-1]))
        loss_max = max(loss_max, float(np.nanmax(mean_loss)))
        loss_min = min(loss_min, float(np.nanmin(mean_loss)))
    loss_min = min(loss_min, 0.95)
    loss_max = max(loss_max, 1.30)
    loss_pad = 0.06 * (loss_max - loss_min)
    loss_min -= loss_pad
    loss_max += loss_pad

    for panel_idx, (metric, title) in enumerate(panels):
        x0 = margin_left + panel_idx * (panel_width + gap)
        y0 = margin_top
        x1 = x0 + panel_width
        y1 = y0 + panel_height
        draw.rectangle([x0, y0, x1, y1], outline="black")
        draw.text((x0 + 8, y0 - 24), title, font=font, fill="black")
        y_min, y_max = (0.0, 80.0) if metric == "acc" else (loss_min, loss_max)
        for tick in range(6):
            ratio = tick / 5.0
            ty = y1 - int(ratio * panel_height)
            draw.line([x0, ty, x1, ty], fill="#e0e0e0")
            value = y_min + (y_max - y_min) * ratio
            label = f"{value:.1f}" if metric == "loss" else f"{value:.0f}"
            draw.text((x0 - 45, ty - 6), label, font=font, fill="black")
        for tick in range(5):
            ratio = tick / 4.0
            tx = x0 + int(ratio * panel_width)
            draw.line([tx, y0, tx, y1], fill="#f0f0f0")
            draw_center(draw, tx, y1 + 8, f"{int(max_epoch * ratio):d}", font)
        curves = acc_curves if metric == "acc" else loss_curves
        for method_name, (epochs, mean) in curves.items():
            points = []
            for epoch, value in zip(epochs, mean):
                px = x0 + (float(epoch) / max_epoch) * panel_width
                py = y1 - ((float(value) - y_min) / (y_max - y_min)) * panel_height
                points.append((int(px), int(py)))
            if len(points) >= 2:
                draw.line(points, fill=hex_to_rgb(METHODS[method_name]["color"]), width=3)

    png_path = OUT_DIR / "cifar100_resnet18_base_iid_tuned_asyncsam_vs_baselines_curves.png"
    pdf_path = OUT_DIR / "cifar100_resnet18_base_iid_tuned_asyncsam_vs_baselines_curves.pdf"
    save_image(image, png_path, pdf_path)
    return png_path, pdf_path


def main():
    rows, all_results = build_rows()
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
