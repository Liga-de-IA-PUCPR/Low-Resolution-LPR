"""
Load benchmark_results.csv and produce comparison plots + summary table.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_CSV = Path("reports/benchmark_results.csv")
PLOTS_DIR = Path("reports/plots")

MODEL_ORDER = ["yolov8n", "yolov9n", "yolo11n", "yolo12n"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    # Keep consistent model order
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df.sort_values("model")


def plot_map_comparison(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    width = 0.35
    bars1 = ax.bar([i - width / 2 for i in x], df["mAP50"], width, label="mAP@0.5", color=COLORS, alpha=0.85)
    bars2 = ax.bar([i + width / 2 for i in x], df["mAP50_95"], width, label="mAP@0.5:0.95",
                   color=COLORS, alpha=0.5, hatch="//")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"].tolist())
    ax.set_ylabel("mAP")
    ax.set_title("YOLO Benchmark — mAP Comparison (LRLPR Scenario-A val set)")
    ax.legend()
    ax.set_ylim(0, min(1.05, df["mAP50"].max() * 1.25))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "map_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved map_comparison.png")


def plot_speed_accuracy(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, row in df.iterrows():
        ax.scatter(row["ms_per_image"], row["mAP50_95"], color=COLORS[MODEL_ORDER.index(row["model"])],
                   s=120, zorder=5)
        ax.annotate(row["model"], (row["ms_per_image"], row["mAP50_95"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_xlabel("Inference time (ms / image)")
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_title("Speed vs Accuracy tradeoff")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "speed_accuracy_tradeoff.png", dpi=150)
    plt.close(fig)
    print("Saved speed_accuracy_tradeoff.png")


def print_summary(df: pd.DataFrame):
    print("\n=== Benchmark Summary ===")
    display_cols = ["model", "mAP50", "mAP50_95", "precision", "recall",
                    "ms_per_image", "model_size_mb", "params_M"]
    print(df[display_cols].to_string(index=False))
    best = df.loc[df["mAP50_95"].idxmax(), "model"]
    fastest = df.loc[df["ms_per_image"].idxmin(), "model"]
    print(f"\nBest accuracy: {best}  |  Fastest: {fastest}")


def main():
    assert RESULTS_CSV.exists(), f"Run benchmark.py first — {RESULTS_CSV} not found"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    print_summary(df)
    plot_map_comparison(df)
    plot_speed_accuracy(df)


if __name__ == "__main__":
    main()
