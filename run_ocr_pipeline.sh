#!/usr/bin/env bash
# LRLPR OCR pipeline, end to end: uv sync -> prepare manifest -> CPU smoke test ->
# full training (auto multi-GPU via DataParallel if more than one CUDA device is
# visible) -> Recognition Rate evaluation on data/test.
#
# Usage:
#   ./run_ocr_pipeline.sh                              # default run
#   SKIP_SMOKE=1 ./run_ocr_pipeline.sh                  # skip the CPU smoke test
#   RUN_NAME=my_run ./run_ocr_pipeline.sh --epochs 100 --batch-size 128
#     (any extra arguments are forwarded to train_ocr.py, e.g. to size the batch
#      for two RTX 3090s)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

RUN_NAME="${RUN_NAME:-run}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

echo "==> uv sync"
uv sync

echo "==> Preparing OCR manifest"
uv run python src/prepare_ocr_dataset.py

if [[ "$SKIP_SMOKE" != "1" ]]; then
  echo "==> Smoke test (fast CPU sanity check before committing GPU time)"
  uv run python src/train_ocr.py --smoke-test
fi

echo "==> Full training run (name=$RUN_NAME)"
uv run python src/train_ocr.py --name "$RUN_NAME" "$@"

echo "==> Evaluating on the held-out test split"
uv run python src/evaluate_ocr.py --checkpoint "runs/ocr/$RUN_NAME/best.pt" --split test

echo "==> Done. Predictions: reports/ocr_eval_results.csv"
