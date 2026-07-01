# Low-Resolution LPR

Dataset oficial do ICPR 2026 LRLPR Challenge. A tarefa oficial é **OCR** (Recognition
Rate = tracks com todos os 7 caracteres corretos) — as placas já chegam recortadas,
então detecção não é o gargalo.

## Pipeline OCR (teacher-student SR + OCR)

Reproduz a receita do time vencedor (DLmath, 82.13%): super-resolução e OCR treinados
juntos, com um teacher EMA (alimentado pelo HR rebaixado) supervisionando o student
(alimentado pelo LR bruto).

```bash
uv run python src/prepare_ocr_dataset.py   # gera data/ocr_manifest.csv (train/val/test)
uv run python src/train_ocr.py --smoke-test  # valida o pipeline em CPU antes do treino real
uv run python src/train_ocr.py             # treino completo (usa todas as GPUs CUDA visíveis)
uv run python src/evaluate_ocr.py --checkpoint runs/ocr/run/best.pt  # Recognition Rate em data/test
```

Ou rode tudo em sequência com `./run_ocr_pipeline.sh` (ver flags com `-h`).

Principais flags de `train_ocr.py`: `--frame-fusion {none,logit-sum}`, `--no-teacher`
(ablation sem o teacher), `--no-multi-gpu` (desliga DataParallel), `--epochs`,
`--batch-size`. Roda em múltiplas GPUs automaticamente via `DataParallel` quando mais
de uma é detectada.

## Benchmark YOLO (detecção — trabalho anterior, mantido para referência)

```bash
uv run python src/prepare_dataset.py   # converte annotations → YOLO format (90/10 split)
uv run python src/benchmark.py         # fine-tuning dos 4 modelos (YOLOv8n, v9n, 11n, 12n)
uv run python src/analyze_results.py   # gera reports/benchmark_results.csv e plots
```
