import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


SERIES = [
    ("asyncsgd", "Async SGD", "#4C72B0"),
    ("asyncsam", "Async SAM", "#8172B2"),
]


def load_result(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    test_prec = np.asarray(data["test_prec"], dtype=float) * 100.0
    train_prec = np.asarray(data["train_prec"], dtype=float) * 100.0
    train_loss = np.asarray(data["train_loss"], dtype=float)
    test_loss = np.asarray(data["test_loss"], dtype=float)
    return {
        "epochs": np.arange(1, len(test_prec) + 1),
        "test_prec": test_prec,
        "train_prec": train_prec,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "final_test": float(test_prec[-1]),
        "best_test": float(test_prec.max()),
        "final_test_loss": float(test_loss[-1]),
        "best_test_loss": float(test_loss.min()),
        "avg_staleness": data.get("avg_staleness", []),
        "max_staleness": data.get("max_staleness", []),
    }


def _hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _lighten(color, alpha=0.45):
    r, g, b = _hex_to_rgb(color)
    return (
        int(255 - (255 - r) * alpha),
        int(255 - (255 - g) * alpha),
        int(255 - (255 - b) * alpha),
    )


def save_pil_figure(run_data, output_dir, title):
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1600, 700
    margin = 70
    panel_gap = 70
    left_w = 860
    right_w = width - margin * 2 - panel_gap - left_w
    panel_top = 110
    panel_h = 500
    left_x0 = margin
    left_x1 = left_x0 + left_w
    right_x0 = left_x1 + panel_gap
    right_x1 = width - margin
    y0 = panel_top
    y1 = panel_top + panel_h

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((width // 2, 35), title, font=font, fill=(20, 20, 20), anchor="mm")

    draw.rectangle([left_x0, y0, left_x1, y1], outline=(180, 180, 180), width=2)
    draw.line([left_x0, y1, left_x1, y1], fill=(90, 90, 90), width=2)
    draw.line([left_x0, y0, left_x0, y1], fill=(90, 90, 90), width=2)

    epochs_max = max(int(run_data[name]["epochs"][-1]) for name, _, _ in SERIES)
    acc_max = max(float(run_data[name]["test_prec"].max()) for name, _, _ in SERIES)
    acc_max = max(100.0, acc_max)

    for tick in range(1, epochs_max + 1):
        x = left_x0 + int((left_x1 - left_x0) * (tick - 1) / max(epochs_max - 1, 1))
        draw.line([x, y0, x, y1], fill=(235, 235, 235), width=1)
        draw.text((x, y1 + 18), str(tick), font=font, fill=(80, 80, 80), anchor="mm")

    for tick in range(0, int(acc_max) + 1, 10):
        y = y1 - int((y1 - y0) * tick / max(acc_max, 1))
        draw.line([left_x0, y, left_x1, y], fill=(235, 235, 235), width=1)
        draw.text((left_x0 - 15, y), str(tick), font=font, fill=(80, 80, 80), anchor="rm")

    draw.text((left_x0 + 10, y0 - 18), "Test Accuracy vs Epoch", font=font, fill=(30, 30, 30))

    for name, label, color in SERIES:
        rgb = _hex_to_rgb(color)
        pts = []
        for epoch, acc in zip(run_data[name]["epochs"], run_data[name]["test_prec"]):
            x = left_x0 + int((left_x1 - left_x0) * (epoch - 1) / max(epochs_max - 1, 1))
            y = y1 - int((y1 - y0) * acc / max(acc_max, 1))
            pts.append((x, y))
        if len(pts) > 1:
            draw.line(pts, fill=rgb, width=4)

    legend_x = left_x0 + 10
    legend_y = y0 + 10
    for _, label, color in SERIES:
        rgb = _hex_to_rgb(color)
        draw.line([(legend_x, legend_y + 8), (legend_x + 56, legend_y + 8)], fill=rgb, width=4)
        draw.text((legend_x + 65, legend_y + 8), label, font=font, fill=(40, 40, 40), anchor="lm")
        legend_y += 22

    draw.rectangle([right_x0, y0, right_x1, y1], outline=(180, 180, 180), width=2)
    draw.line([right_x0, y1, right_x1, y1], fill=(90, 90, 90), width=2)
    draw.line([right_x0, y0, right_x0, y1], fill=(90, 90, 90), width=2)
    draw.text((right_x0 + 10, y0 - 18), "Final Summary", font=font, fill=(30, 30, 30))

    bar_max = max(max(run_data[name]["final_test"] for name, _, _ in SERIES),
                  max(run_data[name]["best_test"] for name, _, _ in SERIES),
                  100.0)
    for tick in range(0, int(bar_max) + 1, 10):
        y = y1 - int((y1 - y0) * tick / max(bar_max, 1))
        draw.line([right_x0, y, right_x1, y], fill=(235, 235, 235), width=1)
        draw.text((right_x0 - 15, y), str(tick), font=font, fill=(80, 80, 80), anchor="rm")

    group_w = (right_x1 - right_x0) / max(len(SERIES), 1)
    bar_w = max(int(group_w * 0.22), 18)
    for idx, (name, label, color) in enumerate(SERIES):
        center = right_x0 + group_w * (idx + 0.5)
        final_h = int((y1 - y0) * run_data[name]["final_test"] / max(bar_max, 1))
        best_h = int((y1 - y0) * run_data[name]["best_test"] / max(bar_max, 1))
        base_rgb = _hex_to_rgb(color)
        lite_rgb = _lighten(color)
        draw.rectangle([center - bar_w - 4, y1 - final_h, center - 4, y1], fill=base_rgb)
        draw.rectangle([center + 4, y1 - best_h, center + bar_w + 4, y1], fill=lite_rgb)
        draw.text((center, y1 + 18), label, font=font, fill=(40, 40, 40), anchor="mm")

    draw.text(((right_x0 + right_x1) // 2, y1 + 45), "Left: Final Test Acc  |  Right: Best Test Acc", font=font, fill=(30, 30, 30), anchor="mm")

    image.save(output_dir / "async_distributed_compare.png")


def save_pil_loss_figure(run_data, output_dir, title):
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1600, 700
    margin = 70
    panel_gap = 70
    left_w = 860
    right_w = width - margin * 2 - panel_gap - left_w
    panel_top = 110
    panel_h = 500
    left_x0 = margin
    left_x1 = left_x0 + left_w
    right_x0 = left_x1 + panel_gap
    right_x1 = width - margin
    y0 = panel_top
    y1 = panel_top + panel_h

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((width // 2, 35), title, font=font, fill=(20, 20, 20), anchor="mm")

    draw.rectangle([left_x0, y0, left_x1, y1], outline=(180, 180, 180), width=2)
    draw.line([left_x0, y1, left_x1, y1], fill=(90, 90, 90), width=2)
    draw.line([left_x0, y0, left_x0, y1], fill=(90, 90, 90), width=2)

    epochs_max = max(int(run_data[name]["epochs"][-1]) for name, _, _ in SERIES)
    loss_max = max(float(run_data[name]["train_loss"].max()) for name, _, _ in SERIES)
    loss_max = max(loss_max, max(float(run_data[name]["test_loss"].max()) for name, _, _ in SERIES))
    loss_max = max(loss_max, 1.0)

    for tick in range(1, epochs_max + 1):
        x = left_x0 + int((left_x1 - left_x0) * (tick - 1) / max(epochs_max - 1, 1))
        draw.line([x, y0, x, y1], fill=(235, 235, 235), width=1)
        draw.text((x, y1 + 18), str(tick), font=font, fill=(80, 80, 80), anchor="mm")

    for tick_id in range(0, 11):
        tick = loss_max * tick_id / 10.0
        y = y1 - int((y1 - y0) * tick / max(loss_max, 1e-8))
        draw.line([left_x0, y, left_x1, y], fill=(235, 235, 235), width=1)
        draw.text((left_x0 - 15, y), f"{tick:.2f}", font=font, fill=(80, 80, 80), anchor="rm")

    draw.text((left_x0 + 10, y0 - 18), "Loss vs Epoch", font=font, fill=(30, 30, 30))

    for name, label, color in SERIES:
        rgb = _hex_to_rgb(color)
        test_pts = []
        train_pts = []
        for epoch, loss in zip(run_data[name]["epochs"], run_data[name]["test_loss"]):
            x = left_x0 + int((left_x1 - left_x0) * (epoch - 1) / max(epochs_max - 1, 1))
            y = y1 - int((y1 - y0) * loss / max(loss_max, 1e-8))
            test_pts.append((x, y))
        for epoch, loss in zip(run_data[name]["epochs"], run_data[name]["train_loss"]):
            x = left_x0 + int((left_x1 - left_x0) * (epoch - 1) / max(epochs_max - 1, 1))
            y = y1 - int((y1 - y0) * loss / max(loss_max, 1e-8))
            train_pts.append((x, y))
        if len(test_pts) > 1:
            draw.line(test_pts, fill=rgb, width=4)
        if len(train_pts) > 1:
            lite_rgb = _lighten(color, alpha=0.65)
            for idx in range(len(train_pts) - 1):
                if idx % 2 == 0:
                    draw.line([train_pts[idx], train_pts[idx + 1]], fill=lite_rgb, width=2)

    legend_x = left_x0 + 10
    legend_y = y0 + 10
    for _, label, color in SERIES:
        rgb = _hex_to_rgb(color)
        lite_rgb = _lighten(color, alpha=0.65)
        draw.line([(legend_x, legend_y + 8), (legend_x + 56, legend_y + 8)], fill=rgb, width=4)
        draw.text((legend_x + 65, legend_y + 8), f"{label} test", font=font, fill=(40, 40, 40), anchor="lm")
        legend_y += 18
        draw.line([(legend_x, legend_y + 8), (legend_x + 56, legend_y + 8)], fill=lite_rgb, width=2)
        draw.text((legend_x + 65, legend_y + 8), f"{label} train", font=font, fill=(40, 40, 40), anchor="lm")
        legend_y += 22

    draw.rectangle([right_x0, y0, right_x1, y1], outline=(180, 180, 180), width=2)
    draw.line([right_x0, y1, right_x1, y1], fill=(90, 90, 90), width=2)
    draw.line([right_x0, y0, right_x0, y1], fill=(90, 90, 90), width=2)
    draw.text((right_x0 + 10, y0 - 18), "Loss Summary", font=font, fill=(30, 30, 30))

    bar_max = max(
        max(run_data[name]["final_test_loss"] for name, _, _ in SERIES),
        max(run_data[name]["best_test_loss"] for name, _, _ in SERIES),
        1.0,
    )
    for tick_id in range(0, 11):
        tick = bar_max * tick_id / 10.0
        y = y1 - int((y1 - y0) * tick / max(bar_max, 1e-8))
        draw.line([right_x0, y, right_x1, y], fill=(235, 235, 235), width=1)
        draw.text((right_x0 - 15, y), f"{tick:.2f}", font=font, fill=(80, 80, 80), anchor="rm")

    group_w = (right_x1 - right_x0) / max(len(SERIES), 1)
    bar_w = max(int(group_w * 0.22), 18)
    for idx, (name, label, color) in enumerate(SERIES):
        center = right_x0 + group_w * (idx + 0.5)
        final_h = int((y1 - y0) * run_data[name]["final_test_loss"] / max(bar_max, 1e-8))
        best_h = int((y1 - y0) * run_data[name]["best_test_loss"] / max(bar_max, 1e-8))
        base_rgb = _hex_to_rgb(color)
        lite_rgb = _lighten(color)
        draw.rectangle([center - bar_w - 4, y1 - final_h, center - 4, y1], fill=base_rgb)
        draw.rectangle([center + 4, y1 - best_h, center + bar_w + 4, y1], fill=lite_rgb)
        draw.text((center, y1 + 18), label, font=font, fill=(40, 40, 40), anchor="mm")

    draw.text(((right_x0 + right_x1) // 2, y1 + 45), "Left: Final Test Loss  |  Right: Best Test Loss", font=font, fill=(30, 30, 30), anchor="mm")

    image.save(output_dir / "async_distributed_loss_compare.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--title", type=str, default="Async Distributed SAM MNIST Comparison")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_data = {}
    for name, _, _ in SERIES:
        run_data[name] = load_result(args.input_dir / f"{name}_seed{args.seed}.pkl")

    with (args.output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "optimizer",
            "final_test_acc",
            "best_test_acc",
            "final_test_loss",
            "best_test_loss",
            "avg_staleness_last_epoch",
            "max_staleness_last_epoch",
        ])
        for name, label, _ in SERIES:
            avg_stale = run_data[name]["avg_staleness"][-1] if run_data[name]["avg_staleness"] else 0.0
            max_stale = run_data[name]["max_staleness"][-1] if run_data[name]["max_staleness"] else 0
            writer.writerow([
                label,
                f"{run_data[name]['final_test']:.4f}",
                f"{run_data[name]['best_test']:.4f}",
                f"{run_data[name]['final_test_loss']:.4f}",
                f"{run_data[name]['best_test_loss']:.4f}",
                f"{avg_stale:.4f}",
                f"{max_stale}",
            ])

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for name, label, color in SERIES:
            axes[0].plot(run_data[name]["epochs"], run_data[name]["test_prec"], label=label, color=color, linewidth=2)

        axes[0].set_title("Test Accuracy vs Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend()

        labels = [label for _, label, _ in SERIES]
        finals = [run_data[name]["final_test"] for name, _, _ in SERIES]
        bests = [run_data[name]["best_test"] for name, _, _ in SERIES]
        xs = np.arange(len(labels))
        axes[1].bar(xs - 0.18, finals, width=0.36, label="Final", color="#4C72B0")
        axes[1].bar(xs + 0.18, bests, width=0.36, label="Best", color="#55A868")
        axes[1].set_xticks(xs)
        axes[1].set_xticklabels(labels, rotation=15)
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Final Summary")
        axes[1].grid(True, axis="y", alpha=0.25)
        axes[1].legend()

        fig.suptitle(args.title)
        fig.tight_layout()
        fig.savefig(args.output_dir / "async_distributed_compare.png", dpi=200, bbox_inches="tight")
        fig.savefig(args.output_dir / "async_distributed_compare.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for name, label, color in SERIES:
            axes[0].plot(run_data[name]["epochs"], run_data[name]["test_loss"], label=f"{label} test", color=color, linewidth=2)
            axes[0].plot(run_data[name]["epochs"], run_data[name]["train_loss"], label=f"{label} train", color=color, linewidth=1.8, linestyle="--", alpha=0.75)

        axes[0].set_title("Loss vs Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend()

        labels = [label for _, label, _ in SERIES]
        finals = [run_data[name]["final_test_loss"] for name, _, _ in SERIES]
        bests = [run_data[name]["best_test_loss"] for name, _, _ in SERIES]
        xs = np.arange(len(labels))
        axes[1].bar(xs - 0.18, finals, width=0.36, label="Final test loss", color="#C44E52")
        axes[1].bar(xs + 0.18, bests, width=0.36, label="Best test loss", color="#55A868")
        axes[1].set_xticks(xs)
        axes[1].set_xticklabels(labels, rotation=15)
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Loss Summary")
        axes[1].grid(True, axis="y", alpha=0.25)
        axes[1].legend()

        fig.suptitle(args.title + " - Loss")
        fig.tight_layout()
        fig.savefig(args.output_dir / "async_distributed_loss_compare.png", dpi=200, bbox_inches="tight")
        fig.savefig(args.output_dir / "async_distributed_loss_compare.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception:
        save_pil_figure(run_data, args.output_dir, args.title)
        save_pil_loss_figure(run_data, args.output_dir, args.title + " - Loss")


if __name__ == "__main__":
    main()
