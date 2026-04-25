import csv
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(
    "/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist/experiments/by_family/cifar100/resnext29"
)
OUT_DIR = BASE_DIR / "resnext29_8x16_lr003_e200_base_iid_seed1_bestfedbuff_compare_fig"

METHODS = {
    "AsyncSAM+RMS": {
        "path": BASE_DIR / "resnext29_8x16_lr003_e200_base_asyncsam_rms_seed1.pkl",
        "color": "#0b5d7a",
    },
    "FedAsync": {
        "path": BASE_DIR / "resnext29_8x16_lr003_e200_base_fedasync_tuned_seed1.pkl",
        "color": "#6c757d",
    },
    "AsyncSGD": {
        "path": BASE_DIR / "resnext29_8x16_lr003_e200_base_asyncsgd_seed1.pkl",
        "color": "#2b8a3e",
    },
    "FedBuff (Best Paper-Style)": {
        "path": BASE_DIR
        / "baseline_tune_base_iid_fedbuff_paper_rescue2"
        / "fedbuff_paper_pure_k2_etag3p9_seed1.pkl",
        "color": "#9467bd",
    },
}


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def load_result(path):
    with path.open("rb") as f:
        data = pickle.load(f)
    return {
        "acc": np.asarray(data["test_prec"], dtype=float) * 100.0,
        "loss": np.asarray(data["test_loss"], dtype=float),
    }


def summarize(result):
    return {
        "final_acc": float(result["acc"][-1]),
        "best_acc": float(np.max(result["acc"])),
        "final_loss": float(result["loss"][-1]),
        "best_loss": float(np.min(result["loss"])),
    }


def draw_center(draw, x, y, text, font, fill="black"):
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y), text, font=font, fill=fill)


def draw_panel(draw, box, title, ymin, ymax, ylabel, font, tick_fmt, xlabel="Epoch"):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline="black")
    draw.text((x0 + 8, y0 - 24), title, font=font, fill="black")
    for tick in range(6):
        ratio = tick / 5.0
        ty = y1 - int(ratio * (y1 - y0))
        draw.line([x0, ty, x1, ty], fill="#dddddd")
        value = ymin + (ymax - ymin) * ratio
        draw.text((x0 - 50, ty - 6), tick_fmt(value), font=font, fill="black")
    if xlabel:
        draw_center(draw, (x0 + x1) / 2, y1 + 26, xlabel, font)
    if ylabel:
        draw.text((x0 - 56, y0 - 8), ylabel, font=font, fill="black")


def value_to_xy(step, value, box, n_steps, ymin, ymax):
    x0, y0, x1, y1 = box
    x = x0 + (step - 1) / max(1, n_steps - 1) * (x1 - x0)
    clipped = max(ymin, min(ymax, float(value)))
    y = y1 - (clipped - ymin) / max(1e-9, (ymax - ymin)) * (y1 - y0)
    return int(x), int(y)


def write_summary(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "summary.csv"
    fieldnames = [
        "method",
        "source_path",
        "final_acc",
        "best_acc",
        "final_loss",
        "best_loss",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def plot(rows, all_results):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 1820, 1240
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw_center(
        draw,
        width / 2,
        20,
        "CIFAR-100 / ResNeXt29-8x16d / Base IID / 200 Epochs / Seed 1",
        font,
    )
    draw_center(
        draw,
        width / 2,
        42,
        "Four-method comparison with the current best paper-style FedBuff: k=2, etag=3.9",
        font,
        fill="#444444",
    )

    legend_x, legend_y = 74, 74
    for method, info in METHODS.items():
        color = hex_to_rgb(info["color"])
        draw.line([legend_x, legend_y + 8, legend_x + 34, legend_y + 8], fill=color, width=5)
        draw.text((legend_x + 42, legend_y), method, font=font, fill="black")
        legend_x += 315 if "FedBuff" in method else 250

    acc_values = np.concatenate([all_results[m]["acc"] for m in METHODS])
    loss_values = np.concatenate([all_results[m]["loss"] for m in METHODS])
    acc_min = np.floor((float(acc_values.min()) - 1.0) * 2.0) / 2.0
    acc_max = np.ceil((float(acc_values.max()) + 0.8) * 2.0) / 2.0
    loss_min = np.floor((float(loss_values.min()) - 0.04) * 50.0) / 50.0
    loss_max = np.ceil((float(loss_values.max()) + 0.05) * 50.0) / 50.0

    acc_box = (92, 140, 850, 590)
    loss_box = (980, 140, 1710, 590)
    draw_panel(draw, acc_box, "Test Accuracy Curve", acc_min, acc_max, "Acc (%)", font, lambda v: f"{v:.1f}")
    draw_panel(draw, loss_box, "Test Loss Curve", loss_min, loss_max, "Loss", font, lambda v: f"{v:.2f}")

    for box in [acc_box, loss_box]:
        x0, y0, x1, y1 = box
        for tick in range(5):
            ratio = tick / 4.0
            tx = x0 + int(ratio * (x1 - x0))
            epoch = int(1 + ratio * 199)
            draw.line([tx, y1, tx, y1 + 4], fill="black")
            draw_center(draw, tx, y1 + 8, str(epoch), font)

    for method, info in METHODS.items():
        color = hex_to_rgb(info["color"])
        result = all_results[method]
        for metric, box, ymin, ymax in [
            ("acc", acc_box, acc_min, acc_max),
            ("loss", loss_box, loss_min, loss_max),
        ]:
            values = result[metric]
            points = [
                value_to_xy(step, value, box, len(values), ymin, ymax)
                for step, value in enumerate(values, start=1)
            ]
            if len(points) >= 2:
                draw.line(points, fill=color, width=4)

    methods = list(METHODS.keys())
    by_method = {row["method"]: row for row in rows}

    final_acc_values = [float(by_method[m]["final_acc"]) for m in methods]
    best_acc_values = [float(by_method[m]["best_acc"]) for m in methods]
    final_loss_values = [float(by_method[m]["final_loss"]) for m in methods]
    final_acc_bar_min = np.floor((min(final_acc_values) - 1.0) * 2.0) / 2.0
    final_acc_bar_max = np.ceil((max(final_acc_values) + 0.5) * 2.0) / 2.0
    best_acc_bar_min = np.floor((min(best_acc_values) - 1.0) * 2.0) / 2.0
    best_acc_bar_max = np.ceil((max(best_acc_values) + 0.5) * 2.0) / 2.0
    final_loss_bar_min = np.floor((min(final_loss_values) - 0.03) * 100.0) / 100.0
    final_loss_bar_max = np.ceil((max(final_loss_values) + 0.03) * 100.0) / 100.0

    bar_boxes = [
        ("Final Accuracy (%)", "final_acc", final_acc_bar_min, final_acc_bar_max, (92, 720, 555, 1115)),
        ("Best Accuracy (%)", "best_acc", best_acc_bar_min, best_acc_bar_max, (665, 720, 1128, 1115)),
        ("Final Test Loss", "final_loss", final_loss_bar_min, final_loss_bar_max, (1245, 720, 1710, 1115)),
    ]
    for title, metric, ymin, ymax, box in bar_boxes:
        x0, y0, x1, y1 = box
        draw_panel(
            draw,
            box,
            title,
            ymin,
            ymax,
            "",
            font,
            lambda v: f"{v:.2f}" if "Loss" in title else f"{v:.1f}",
            xlabel="",
        )
        group_width = (x1 - x0) / len(methods)
        for idx, method in enumerate(methods):
            row = by_method[method]
            value = float(row[metric])
            bx0 = x0 + idx * group_width + group_width * 0.22
            bx1 = x0 + idx * group_width + group_width * 0.78
            by0 = y1 - (value - ymin) / max(1e-9, (ymax - ymin)) * (y1 - y0)
            draw.rectangle(
                [int(bx0), int(by0), int(bx1), y1],
                fill=hex_to_rgb(METHODS[method]["color"]),
                outline="black",
            )
            label = method.replace("Async", "A.").replace(" (Best Paper-Style)", "")
            draw_center(draw, int((bx0 + bx1) / 2), y1 + 12, label, font)

    footer = (
        "This figure is a same-setting, same-seed comparison. FedBuff uses the best current paper-style "
        "buffered configuration from the rescue sweep."
    )
    draw.text((28, height - 38), footer, font=font, fill="#444444")

    png_path = OUT_DIR / "resnext29_8x16_lr003_e200_base_iid_seed1_bestfedbuff_compare.png"
    pdf_path = OUT_DIR / "resnext29_8x16_lr003_e200_base_iid_seed1_bestfedbuff_compare.pdf"
    image.save(png_path)
    image.save(pdf_path, "PDF", resolution=200.0)
    return png_path, pdf_path


def main():
    rows = []
    all_results = {}
    for method, info in METHODS.items():
        result = load_result(info["path"])
        all_results[method] = result
        row = {"method": method, "source_path": str(info["path"])}
        row.update(summarize(result))
        rows.append(row)
    csv_path = write_summary(rows)
    png_path, pdf_path = plot(rows, all_results)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
