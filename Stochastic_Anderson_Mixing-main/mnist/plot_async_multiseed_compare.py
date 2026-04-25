import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


PALETTE = [
    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#CCB974",
    "#64B5CD",
]


def load_result(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    return {
        "epochs": np.arange(1, len(data["test_prec"]) + 1),
        "test_prec": np.asarray(data["test_prec"], dtype=float) * 100.0,
        "test_loss": np.asarray(data["test_loss"], dtype=float),
    }


def parse_group(spec: str):
    label, path_blob = spec.split("=", 1)
    paths = [Path(p) for p in path_blob.split(",") if p.strip()]
    return label, paths


def aggregate_runs(paths):
    runs = [load_result(path) for path in paths]
    min_len = min(len(run["epochs"]) for run in runs)
    acc = np.stack([run["test_prec"][:min_len] for run in runs], axis=0)
    loss = np.stack([run["test_loss"][:min_len] for run in runs], axis=0)
    epochs = runs[0]["epochs"][:min_len]
    return {
        "epochs": epochs,
        "acc_mean": acc.mean(axis=0),
        "acc_std": acc.std(axis=0),
        "loss_mean": loss.mean(axis=0),
        "loss_std": loss.std(axis=0),
        "final_acc_mean": float(acc[:, -1].mean()),
        "final_acc_std": float(acc[:, -1].std()),
        "best_acc_mean": float(acc.max(axis=1).mean()),
        "best_acc_std": float(acc.max(axis=1).std()),
        "final_loss_mean": float(loss[:, -1].mean()),
        "final_loss_std": float(loss[:, -1].std()),
        "best_loss_mean": float(loss.min(axis=1).mean()),
        "best_loss_std": float(loss.min(axis=1).std()),
    }


def _hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def draw_pil(groups, output_dir: Path, title: str, acc_ymax=None, loss_ymax=None):
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1800, 760
    margin = 70
    gap = 60
    top = 95
    panel_h = 520
    left_w = 800
    right_w = 800

    x0_acc = margin
    x1_acc = x0_acc + left_w
    x0_loss = x1_acc + gap
    x1_loss = x0_loss + right_w
    y0 = top
    y1 = top + panel_h

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((width // 2, 35), title, font=font, fill=(20, 20, 20), anchor="mm")

    for x0, x1 in [(x0_acc, x1_acc), (x0_loss, x1_loss)]:
        draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180), width=2)
        draw.line([x0, y1, x1, y1], fill=(90, 90, 90), width=2)
        draw.line([x0, y0, x0, y1], fill=(90, 90, 90), width=2)

    epochs_max = max(int(group["data"]["epochs"][-1]) for group in groups)
    acc_max = max(float(group["data"]["acc_mean"].max() + group["data"]["acc_std"].max()) for group in groups)
    acc_max = max(acc_max, 100.0)
    loss_max = max(float(group["data"]["loss_mean"].max() + group["data"]["loss_std"].max()) for group in groups)
    loss_max = max(loss_max, 1.0)
    if acc_ymax is not None:
        acc_max = float(acc_ymax)
    if loss_ymax is not None:
        loss_max = float(loss_ymax)

    for tick in range(1, epochs_max + 1):
        x_acc = x0_acc + int((x1_acc - x0_acc) * (tick - 1) / max(epochs_max - 1, 1))
        x_loss = x0_loss + int((x1_loss - x0_loss) * (tick - 1) / max(epochs_max - 1, 1))
        draw.line([x_acc, y0, x_acc, y1], fill=(235, 235, 235), width=1)
        draw.line([x_loss, y0, x_loss, y1], fill=(235, 235, 235), width=1)

    for tick in range(0, int(acc_max) + 1, 10):
        y = y1 - int((y1 - y0) * tick / max(acc_max, 1))
        draw.line([x0_acc, y, x1_acc, y], fill=(235, 235, 235), width=1)
        draw.text((x0_acc - 15, y), str(tick), font=font, fill=(80, 80, 80), anchor="rm")

    for tick_id in range(0, 11):
        tick = loss_max * tick_id / 10.0
        y = y1 - int((y1 - y0) * tick / max(loss_max, 1e-8))
        draw.line([x0_loss, y, x1_loss, y], fill=(235, 235, 235), width=1)
        draw.text((x0_loss - 15, y), f"{tick:.2f}", font=font, fill=(80, 80, 80), anchor="rm")

    draw.text((x0_acc + 10, y0 - 18), "Mean Test Accuracy", font=font, fill=(30, 30, 30))
    draw.text((x0_loss + 10, y0 - 18), "Mean Test Loss", font=font, fill=(30, 30, 30))

    legend_x = margin + 10
    legend_y = 60
    for group in groups:
        rgb = _hex_to_rgb(group["color"])
        draw.line([(legend_x, legend_y), (legend_x + 55, legend_y)], fill=rgb, width=4)
        draw.text((legend_x + 65, legend_y), group["label"], font=font, fill=(40, 40, 40), anchor="lm")
        legend_y += 18

    for group in groups:
        rgb = _hex_to_rgb(group["color"])
        acc_pts = []
        loss_pts = []
        for epoch, acc in zip(group["data"]["epochs"], group["data"]["acc_mean"]):
            x = x0_acc + int((x1_acc - x0_acc) * (epoch - 1) / max(epochs_max - 1, 1))
            y = y1 - int((y1 - y0) * acc / max(acc_max, 1))
            acc_pts.append((x, y))
        for epoch, loss in zip(group["data"]["epochs"], group["data"]["loss_mean"]):
            x = x0_loss + int((x1_loss - x0_loss) * (epoch - 1) / max(epochs_max - 1, 1))
            y = y1 - int((y1 - y0) * loss / max(loss_max, 1e-8))
            loss_pts.append((x, y))
        if len(acc_pts) > 1:
            draw.line(acc_pts, fill=rgb, width=4)
        if len(loss_pts) > 1:
            draw.line(loss_pts, fill=rgb, width=4)

    image.save(output_dir / "multiseed_compare.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", action="append", required=True, help="label=path1,path2,path3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Async Multi-Seed Comparison")
    parser.add_argument("--acc-ymax", type=float, default=None)
    parser.add_argument("--loss-ymax", type=float, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups = []
    for idx, spec in enumerate(args.group):
        label, paths = parse_group(spec)
        groups.append({
            "label": label,
            "paths": paths,
            "color": PALETTE[idx % len(PALETTE)],
            "data": aggregate_runs(paths),
        })

    with (args.output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label",
            "final_acc_mean",
            "final_acc_std",
            "best_acc_mean",
            "best_acc_std",
            "final_loss_mean",
            "final_loss_std",
            "best_loss_mean",
            "best_loss_std",
            "paths",
        ])
        for group in groups:
            d = group["data"]
            writer.writerow([
                group["label"],
                f"{d['final_acc_mean']:.4f}",
                f"{d['final_acc_std']:.4f}",
                f"{d['best_acc_mean']:.4f}",
                f"{d['best_acc_std']:.4f}",
                f"{d['final_loss_mean']:.4f}",
                f"{d['final_loss_std']:.4f}",
                f"{d['best_loss_mean']:.4f}",
                f"{d['best_loss_std']:.4f}",
                ";".join(str(path) for path in group["paths"]),
            ])

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for group in groups:
            d = group["data"]
            color = group["color"]
            axes[0].plot(d["epochs"], d["acc_mean"], label=group["label"], color=color, linewidth=2)
            axes[0].fill_between(
                d["epochs"],
                d["acc_mean"] - d["acc_std"],
                d["acc_mean"] + d["acc_std"],
                color=color,
                alpha=0.18,
            )
            axes[1].plot(d["epochs"], d["loss_mean"], label=group["label"], color=color, linewidth=2)
            axes[1].fill_between(
                d["epochs"],
                d["loss_mean"] - d["loss_std"],
                d["loss_mean"] + d["loss_std"],
                color=color,
                alpha=0.18,
            )

        axes[0].set_title("Test Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy (%)")
        if args.acc_ymax is not None:
            axes[0].set_ylim(top=args.acc_ymax)
        axes[0].grid(True, alpha=0.25)
        axes[0].legend()

        axes[1].set_title("Test Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_ylim(bottom=0.0)
        if args.loss_ymax is not None:
            axes[1].set_ylim(top=args.loss_ymax)
        axes[1].grid(True, alpha=0.25)
        axes[1].legend()

        fig.suptitle(args.title)
        fig.tight_layout()
        fig.savefig(args.output_dir / "multiseed_compare.png", dpi=220, bbox_inches="tight")
        fig.savefig(args.output_dir / "multiseed_compare.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception:
        draw_pil(groups, args.output_dir, args.title, acc_ymax=args.acc_ymax, loss_ymax=args.loss_ymax)


if __name__ == "__main__":
    main()
