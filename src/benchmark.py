"""
Fine-tune and evaluate 4 YOLO models on the LRLPR dataset.
Saves per-model metrics to reports/benchmark_results.csv.
"""
import csv
import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO

DATASET_YAML = Path("data/yolo_dataset/dataset.yaml")
RUNS_DIR = Path("runs/benchmark")
REPORTS_DIR = Path("reports")

MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolov9n": "yolov9t.pt",   # ultralytics uses 'yolov9t' for the tiny/nano tier
    "yolo11n": "yolo11n.pt",
    "yolo12n": "yolo12n.pt",
}

TRAIN_CFG = dict(
    data=str(DATASET_YAML.resolve()),
    epochs=100,
    imgsz=320,
    batch=64,
    workers=4,
    patience=20,
    verbose=False,
    exist_ok=True,
)

VAL_IMAGE_LIMIT = 500  # images sampled for inference-time measurement


def detect_device() -> str | int:
    if torch.cuda.is_available():
        return 0
    return "cpu"


def train_model(model_name: str, weights: str, device) -> Path:
    print(f"\n{'='*60}")
    print(f"Training {model_name} ({weights})")
    print(f"{'='*60}")
    model = YOLO(weights)
    model.train(
        **TRAIN_CFG,
        device=device,
        project=str(RUNS_DIR),
        name=model_name,
    )
    best = RUNS_DIR / model_name / "weights" / "best.pt"
    return best


def evaluate_model(model_name: str, best_weights: Path, device) -> dict:
    model = YOLO(str(best_weights))

    # Validation metrics
    val_results = model.val(
        data=str(DATASET_YAML.resolve()),
        imgsz=320,
        device=device,
        verbose=False,
    )
    metrics = val_results.results_dict

    # Inference time on a sample of val images
    val_images = sorted((Path("data/yolo_dataset/val/images")).glob("*.png"))[:VAL_IMAGE_LIMIT]
    if val_images:
        start = time.perf_counter()
        model.predict(
            [str(p) for p in val_images],
            imgsz=320,
            device=device,
            verbose=False,
            save=False,
        )
        elapsed = time.perf_counter() - start
        ms_per_image = elapsed / len(val_images) * 1000
    else:
        ms_per_image = float("nan")

    # Model size and parameter count
    model_size_mb = os.path.getsize(best_weights) / 1e6
    n_params = sum(p.numel() for p in model.model.parameters()) / 1e6

    return {
        "model": model_name,
        "mAP50": round(metrics.get("metrics/mAP50(B)", float("nan")), 4),
        "mAP50_95": round(metrics.get("metrics/mAP50-95(B)", float("nan")), 4),
        "precision": round(metrics.get("metrics/precision(B)", float("nan")), 4),
        "recall": round(metrics.get("metrics/recall(B)", float("nan")), 4),
        "ms_per_image": round(ms_per_image, 2),
        "model_size_mb": round(model_size_mb, 2),
        "params_M": round(n_params, 2),
        "device": str(device),
    }


def save_results(results: list[dict]):
    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "benchmark_results.csv"
    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {out_path}")


def main():
    assert DATASET_YAML.exists(), f"Run prepare_dataset.py first — {DATASET_YAML} not found"
    device = detect_device()
    print(f"Using device: {device}")

    all_results = []
    for model_name, weights in MODELS.items():
        best_weights = train_model(model_name, weights, device)
        row = evaluate_model(model_name, best_weights, device)
        all_results.append(row)
        print(f"{model_name}: mAP50={row['mAP50']}, mAP50-95={row['mAP50_95']}, "
              f"P={row['precision']}, R={row['recall']}, {row['ms_per_image']}ms/img")

    save_results(all_results)


if __name__ == "__main__":
    main()
