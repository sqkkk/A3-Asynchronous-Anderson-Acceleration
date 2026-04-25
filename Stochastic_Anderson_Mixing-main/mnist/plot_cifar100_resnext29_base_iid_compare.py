import csv
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(
    "/mnt/liuyx_data/shiqk/Stochastic_Anderson_Mixing-main/mnist/experiments/by_family/cifar100/resnext29"
)
OUT_DIR = BASE_DIR / "resnext29_8x16_lr003_e200_base_iid_compare_fig"

METHODS = {
    "AsyncSAM+RMS": "asyncsam_rms",
    # Tuned algorithm-specific server parameters; shared protocol is unchanged.
    "FedAsync": "fedasync_tuned",
    "AsyncSGD": "asyncsgd",
    "FedBuff": "fedbuff_tuned",
}

COLORS = {
    "AsyncSAM+RMS": "#0b5d7a",
    "FedAsync": "#6c757d",
    "AsyncSGD": "#2b8a3e",
    "FedBuff": "#9467bd",
}


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def result_path(method_key, seed):
    return BASE_DIR / f"resnext29_8x16_lr003_e200_base_{method_key}_seed{seed}.pkl"


def load_result(path):
    with path.open("rb") as f:
        data = pickle.load(f)
    return {
        "acc": np.asarray(data["test_prec"], dtype=float) * 100.0,
        "loss": np.asarray(data["test_loss"], dtype=float),
    }


def load_group(method_key):
    results = []
    missing = []
    for seed in range(1, 11):
        path = result_path(method_key, seed)
        if path.exists() and path.stat().st_size > 0:
            results.append(load_result(path))
        else:
            missing.append(seed)
    return results, missing


def summarize(results):
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
    min_len = min(len(r[metric]) for r in results)
    stacked = np.stack([r[metric][:min_len] for r in results], axis=0)
    return np.arange(1, min_len + 1), stacked.mean(axis=0), stacked.std(axis=0)


def draw_center(draw, x, y, text, font, fill="black"):
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y), text, font=font, fill=fill)


def draw_panel(draw, box, title, ymin, ymax, ylabel, font):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline="black")
    draw.text((x0 + 8, y0 - 24), title, font=font, fill="black")
    for tick in range(6):
        ratio = tick / 5.0
        ty = y1 - int(ratio * (y1 - y0))
        draw.line([x0, ty, x1, ty], fill="#dddddd")
        value = ymin + (ymax - ymin) * ratio
        label = f"{value:.1f}" if ymax - ymin < 10 else f"{value:.0f}"
        draw.text((x0 - 54, ty - 6), label, font=font, fill="black")
    draw_center(draw, (x0 + x1) / 2, y1 + 30, "Epoch", font)
    draw.text((x0 - 58, y0 - 8), ylabel, font=font, fill="black")


def value_to_xy(step, value, box, n_steps, ymin, ymax):
    x0, y0, x1, y1 = box
    x = x0 + (step - 1) / max(1, n_steps - 1) * (x1 - x0)
    clipped = max(ymin, min(ymax, float(value)))
    y = y1 - (clipped - ymin) / (ymax - ymin) * (y1 - y0)
    return int(x), int(y)


def write_summary(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "summary.csv"
    fields = [
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
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot(rows, all_results):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 1800, 1260
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    methods = list(METHODS.keys())
    by_method = {row["method"]: row for row in rows}

    draw_center(
        draw,
        width / 2,
        22,
        "CIFAR-100 / ResNeXt29-8x16d / Base IID / 200 Epochs / 10 Seeds",
        font,
    )

    legend_x, legend_y = 82, 58
    for method in methods:
        color = hex_to_rgb(COLORS[method])
        draw.line([legend_x, legend_y + 9, legend_x + 36, legend_y + 9], fill=color, width=5)
        draw.text((legend_x + 44, legend_y + 1), method, font=font, fill="black")
        legend_x += 270

    acc_box = (92, 125, 850, 590)
    loss_box = (980, 125, 1710, 590)
    loss_values = np.concatenate([r["loss"] for results in all_results.values() for r in results])
    loss_min = float(loss_values.min())
    loss_max = float(loss_values.max())
    loss_pad = max(0.05, 0.08 * (loss_max - loss_min))
    loss_ymin = max(0.0, loss_min - loss_pad)
    loss_ymax = loss_max + loss_pad
    # Use zero-based axes for the main figure to avoid visually exaggerating gaps.
    draw_panel(draw, acc_box, "Mean Test Accuracy Curve", 0.0, 80.0, "Acc (%)", font)
    draw_panel(draw, loss_box, "Mean Test Loss Curve", loss_ymin, loss_ymax, "Loss", font)

    for box in [acc_box, loss_box]:
        x0, y0, x1, y1 = box
        for tick in range(5):
            ratio = tick / 4.0
            tx = x0 + int(ratio * (x1 - x0))
            epoch = int(1 + ratio * 199)
            draw.line([tx, y1, tx, y1 + 4], fill="black")
            draw_center(draw, tx, y1 + 8, str(epoch), font)

    for method in methods:
        color = hex_to_rgb(COLORS[method])
        for metric, box, ymin, ymax in [
            ("acc", acc_box, 0.0, 80.0),
            ("loss", loss_box, loss_ymin, loss_ymax),
        ]:
            steps, mean, _ = curve_stats(all_results[method], metric)
            points = [value_to_xy(s, v, box, len(steps), ymin, ymax) for s, v in zip(steps, mean)]
            if len(points) >= 2:
                draw.line(points, fill=color, width=4)

    # Compact metric bars under the curves.
    bar_boxes = [
        ("Final Accuracy (%)", "final_acc", "final_acc_std", 0.0, 80.0, (92, 725, 555, 1120)),
        ("Best Accuracy (%)", "best_acc", "best_acc_std", 0.0, 80.0, (665, 725, 1128, 1120)),
        ("Final Test Loss", "final_loss", "final_loss_std", 0.9, 1.35, (1245, 725, 1710, 1120)),
    ]
    for title, metric, std_metric, ymin, ymax, box in bar_boxes:
        x0, y0, x1, y1 = box
        draw_panel(draw, box, title, ymin, ymax, "", font)
        group_width = (x1 - x0) / len(methods)
        for idx, method in enumerate(methods):
            row = by_method[method]
            value = float(row[metric])
            error = float(row[std_metric])
            bx0 = x0 + idx * group_width + group_width * 0.22
            bx1 = x0 + idx * group_width + group_width * 0.78
            clipped = max(ymin, min(ymax, value))
            by0 = y1 - (clipped - ymin) / (ymax - ymin) * (y1 - y0)
            draw.rectangle(
                [int(bx0), int(by0), int(bx1), y1],
                fill=hex_to_rgb(COLORS[method]),
                outline="black",
            )
            err_low = max(ymin, value - error)
            err_high = min(ymax, value + error)
            ey_low = y1 - (err_low - ymin) / (ymax - ymin) * (y1 - y0)
            ey_high = y1 - (err_high - ymin) / (ymax - ymin) * (y1 - y0)
            ex = int((bx0 + bx1) / 2)
            draw.line([ex, int(ey_low), ex, int(ey_high)], fill="black", width=1)
            draw.line([ex - 4, int(ey_low), ex + 4, int(ey_low)], fill="black", width=1)
            draw.line([ex - 4, int(ey_high), ex + 4, int(ey_high)], fill="black", width=1)
            label = method.replace("Async", "A.").replace("+RMS", "+RMS")
            draw_center(draw, ex, y1 + 12, label, font)

    png_path = OUT_DIR / "resnext29_8x16_lr003_e200_base_iid_compare.png"
    pdf_path = OUT_DIR / "resnext29_8x16_lr003_e200_base_iid_compare.pdf"
    image.save(png_path)
    image.save(pdf_path, "PDF", resolution=200.0)
    return png_path, pdf_path


def main():
    rows = []
    all_results = {}
    for method_name, method_key in METHODS.items():
        results, missing = load_group(method_key)
        all_results[method_name] = results
        row = {"method": method_name}
        row.update(summarize(results))
        row["missing_seeds"] = " ".join(map(str, missing))
        rows.append(row)
    csv_path = write_summary(rows)
    png_path, pdf_path = plot(rows, all_results)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
