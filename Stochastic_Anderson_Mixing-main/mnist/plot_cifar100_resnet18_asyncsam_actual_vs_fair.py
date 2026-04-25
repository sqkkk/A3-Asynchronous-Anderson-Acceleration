import csv
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path("/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist/experiments")
FAIR_SUMMARY_CSV = (
    BASE_DIR
    / "cifar100_resnet18_e300_ra_re_ls_four_settings_10seed_fig"
    / "summary.csv"
)
OUT_DIR = BASE_DIR / "cifar100_resnet18_asyncsam_actual_fig"

SETTINGS = [
    ("Base IID", "base"),
    ("Label-Sorted Non-IID", "labelsorted"),
    ("Dirichlet Non-IID", "dirichlet"),
    ("High Delay IID", "highdelay"),
]

FAIR_COLOR = "#98a4b3"
TUNED_COLOR = "#0b5d7a"
DELTA_COLOR = "#b24a2f"


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def load_fair_rows():
    rows = {}
    with FAIR_SUMMARY_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["method"] != "AsyncSAM+RMS":
                continue
            rows[row["setting"]] = {
                "final_acc": float(row["final_acc"]),
                "final_acc_std": float(row["final_acc_std"]),
                "best_acc": float(row["best_acc"]),
                "best_acc_std": float(row["best_acc_std"]),
            }
    return rows


def tuned_result_path(setting_key, seed):
    if setting_key == "base":
        return BASE_DIR / f"cifar100_tune_r18_e300_sam_mind020_asyncsam_rms_seed{seed}.pkl"
    return BASE_DIR / f"cifar100_tune_r18_e300_{setting_key}_sam_mind020_asyncsam_rms_seed{seed}.pkl"


def load_tuned_group(setting_key):
    acc_final = []
    acc_best = []
    loss_final = []
    for seed in range(1, 11):
        path = tuned_result_path(setting_key, seed)
        if not path.exists():
            continue
        with path.open("rb") as f:
            data = pickle.load(f)
        acc = np.asarray(data["test_prec"], dtype=float) * 100.0
        loss = np.asarray(data["test_loss"], dtype=float)
        acc_final.append(float(acc[-1]))
        acc_best.append(float(np.max(acc)))
        loss_final.append(float(loss[-1]))
    return {
        "num_seeds": len(acc_final),
        "final_acc": float(np.mean(acc_final)),
        "final_acc_std": float(np.std(acc_final)),
        "best_acc": float(np.mean(acc_best)),
        "best_acc_std": float(np.std(acc_best)),
        "final_loss": float(np.mean(loss_final)),
    }


def load_peak_result():
    best_peak = None
    best_name = None
    for path in sorted(BASE_DIR.glob("cifar100_tune_r18_e300_sam_mind020_asyncsam_rms_seed*.pkl")):
        with path.open("rb") as f:
            data = pickle.load(f)
        acc = np.asarray(data["test_prec"], dtype=float) * 100.0
        peak = float(np.max(acc))
        if best_peak is None or peak > best_peak:
            best_peak = peak
            best_name = path.name
    return best_name, best_peak


def build_rows():
    fair = load_fair_rows()
    rows = []
    for setting_name, setting_key in SETTINGS:
        tuned = load_tuned_group(setting_key)
        fair_row = fair[setting_name]
        rows.append(
            {
                "setting": setting_name,
                "fair_final_acc": fair_row["final_acc"],
                "fair_final_acc_std": fair_row["final_acc_std"],
                "fair_best_acc": fair_row["best_acc"],
                "fair_best_acc_std": fair_row["best_acc_std"],
                "tuned_final_acc": tuned["final_acc"],
                "tuned_final_acc_std": tuned["final_acc_std"],
                "tuned_best_acc": tuned["best_acc"],
                "tuned_best_acc_std": tuned["best_acc_std"],
                "tuned_final_loss": tuned["final_loss"],
                "delta_final_acc": tuned["final_acc"] - fair_row["final_acc"],
                "delta_best_acc": tuned["best_acc"] - fair_row["best_acc"],
                "num_seeds": tuned["num_seeds"],
            }
        )
    return rows


def write_summary(rows, peak_name, peak_value):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "summary.csv"
    fieldnames = [
        "setting",
        "num_seeds",
        "fair_final_acc",
        "fair_final_acc_std",
        "fair_best_acc",
        "fair_best_acc_std",
        "tuned_final_acc",
        "tuned_final_acc_std",
        "tuned_best_acc",
        "tuned_best_acc_std",
        "tuned_final_loss",
        "delta_final_acc",
        "delta_best_acc",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    notes_path = OUT_DIR / "peak_note.txt"
    notes_path.write_text(f"Base IID single-seed peak: {peak_value:.2f}% ({peak_name})\n")
    return csv_path, notes_path


def draw_center(draw, x, y, text, font, fill="black"):
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y), text, font=font, fill=fill)


def save_image(image, png_path, pdf_path):
    image.save(png_path)
    image.save(pdf_path, "PDF", resolution=200.0)


def draw_bars(
    draw,
    rows,
    panel_box,
    metric_prefix,
    title,
    axis_min,
    axis_max,
    font,
):
    x0, y0, x1, y1 = panel_box
    panel_width = x1 - x0
    panel_height = y1 - y0
    draw.rectangle([x0, y0, x1, y1], outline="black")
    draw.text((x0 + 8, y0 - 24), title, font=font, fill="black")
    for tick in range(6):
        ratio = tick / 5.0
        ty = y1 - int(ratio * panel_height)
        draw.line([x0, ty, x1, ty], fill="#e2e2e2")
        value = axis_min + (axis_max - axis_min) * ratio
        draw.text((x0 - 42, ty - 6), f"{value:.1f}", font=font, fill="black")
    group_width = panel_width / len(rows)
    bar_width = group_width / 3.3
    for idx, row in enumerate(rows):
        center = x0 + (idx + 0.5) * group_width
        label = row["setting"].replace(" Non-IID", "")
        label = label.replace("Label-Sorted", "Label")
        label = label.replace("High Delay IID", "High Delay")
        draw_center(draw, center, y1 + 10, label, font)
        entries = [
            (
                row[f"fair_{metric_prefix}"],
                row[f"fair_{metric_prefix}_std"],
                FAIR_COLOR,
                "Fair",
            ),
            (
                row[f"tuned_{metric_prefix}"],
                row[f"tuned_{metric_prefix}_std"],
                TUNED_COLOR,
                "Tuned",
            ),
        ]
        for bar_idx, (value, std, color, _) in enumerate(entries):
            bx0 = center - group_width / 3 + bar_idx * (bar_width * 1.35)
            bx1 = bx0 + bar_width
            ratio = (value - axis_min) / (axis_max - axis_min)
            ratio = max(0.0, min(1.0, ratio))
            by0 = y1 - ratio * panel_height
            draw.rectangle(
                [int(bx0), int(by0), int(bx1), y1],
                fill=hex_to_rgb(color),
                outline="black",
            )
            std_low = max(axis_min, value - std)
            std_high = min(axis_max, value + std)
            py_low = y1 - ((std_low - axis_min) / (axis_max - axis_min)) * panel_height
            py_high = y1 - ((std_high - axis_min) / (axis_max - axis_min)) * panel_height
            midx = (bx0 + bx1) / 2
            draw.line([int(midx), int(py_high), int(midx), int(py_low)], fill="black", width=1)
            draw.line([int(midx - 5), int(py_high), int(midx + 5), int(py_high)], fill="black", width=1)
            draw.line([int(midx - 5), int(py_low), int(midx + 5), int(py_low)], fill="black", width=1)
        delta = row[f"delta_{metric_prefix}"]
        draw_center(
            draw,
            center,
            max(y0 + 6, y1 - ((row[f'tuned_{metric_prefix}'] - axis_min) / (axis_max - axis_min)) * panel_height - 16),
            f"+{delta:.2f}",
            font,
            fill=DELTA_COLOR,
        )


def plot_summary(rows, peak_name, peak_value):
    width, height = 1800, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw_center(
        draw,
        width / 2,
        18,
        "CIFAR-100 / ResNet18 / AsyncSAM+RMS: Fair Shared Protocol vs Tuned Actual Performance",
        font,
    )
    subtitle = (
        "Fair bars use the 11.17 shared-protocol 10-seed table; "
        "tuned bars use the 11.18 AsyncSAM+RMS-only tuned recipe (mind020)."
    )
    draw_center(draw, width / 2, 40, subtitle, font)
    peak_text = f"Base IID single-seed peak: {peak_value:.2f}% ({peak_name})"
    draw_center(draw, width / 2, 62, peak_text, font, fill=DELTA_COLOR)

    legend_x, legend_y = 160, 86
    legend_items = [("Fair Shared", FAIR_COLOR), ("Tuned Actual", TUNED_COLOR), ("Gain label", DELTA_COLOR)]
    for label, color in legend_items:
        if label == "Gain label":
            draw.text((legend_x, legend_y + 2), "+0.00", font=font, fill=hex_to_rgb(color))
            legend_x += 110
        else:
            draw.rectangle([legend_x, legend_y, legend_x + 18, legend_y + 18], fill=hex_to_rgb(color), outline="black")
            draw.text((legend_x + 26, legend_y + 2), label, font=font, fill="black")
            legend_x += 190

    final_values = [r["fair_final_acc"] for r in rows] + [r["tuned_final_acc"] for r in rows]
    best_values = [r["fair_best_acc"] for r in rows] + [r["tuned_best_acc"] for r in rows]
    final_min = np.floor((min(final_values) - 0.4) * 2.0) / 2.0
    final_max = np.ceil((max(final_values) + 0.4) * 2.0) / 2.0
    best_min = np.floor((min(best_values) - 0.4) * 2.0) / 2.0
    best_max = np.ceil((max(best_values) + 0.4) * 2.0) / 2.0

    margin_left, margin_top, margin_bottom, gap = 90, 130, 110, 80
    panel_width = (width - 2 * margin_left - gap) // 2
    panel_height = height - margin_top - margin_bottom
    draw_bars(
        draw,
        rows,
        (margin_left, margin_top, margin_left + panel_width, margin_top + panel_height),
        "final_acc",
        "Final Accuracy (%)",
        final_min,
        final_max,
        font,
    )
    draw_bars(
        draw,
        rows,
        (
            margin_left + panel_width + gap,
            margin_top,
            margin_left + 2 * panel_width + gap,
            margin_top + panel_height,
        ),
        "best_acc",
        "Best Accuracy (%)",
        best_min,
        best_max,
        font,
    )

    png_path = OUT_DIR / "cifar100_resnet18_asyncsam_actual_vs_fair.png"
    pdf_path = OUT_DIR / "cifar100_resnet18_asyncsam_actual_vs_fair.pdf"
    save_image(image, png_path, pdf_path)
    return png_path, pdf_path


def main():
    rows = build_rows()
    peak_name, peak_value = load_peak_result()
    csv_path, notes_path = write_summary(rows, peak_name, peak_value)
    png_path, pdf_path = plot_summary(rows, peak_name, peak_value)
    print(csv_path)
    print(notes_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
