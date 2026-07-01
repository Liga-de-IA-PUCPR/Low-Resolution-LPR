# Low-Resolution LPR — YOLO Benchmark

```bash
uv run python src/prepare_dataset.py   # converte annotations → YOLO format (90/10 split)
uv run python src/benchmark.py         # fine-tuning dos 4 modelos (YOLOv8n, v9n, 11n, 12n)
uv run python src/analyze_results.py   # gera reports/benchmark_results.csv e plots
```
