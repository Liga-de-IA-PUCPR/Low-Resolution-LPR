"""
Evaluate Recognition Rate (exact-match track accuracy, the official LRLPR metric)
on a held-out split using the student weights only (LR frames only, no HR/teacher).
"""
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from ocr.data import PlateTrackDataset
from ocr.model import SRPlateNet
from ocr.teacher_student import fuse_logits_sum
from ocr.vocab import ctc_greedy_decode

MANIFEST_CSV = "data/ocr_manifest.csv"
REPORTS_DIR = Path("reports")
N_FRAMES = 5


def detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--frame-fusion", choices=["logit-sum", "none"], default="logit-sum")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def load_student(checkpoint_path: str, device: str) -> SRPlateNet:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SRPlateNet(ckpt["sr_cfg"], ckpt["ocr_cfg"])
    model.load_state_dict(ckpt["student"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_track_texts(model: SRPlateNet, lr: torch.Tensor, frame_fusion: str) -> list[str]:
    b = lr.shape[0]
    lr_flat = lr.reshape(b * N_FRAMES, *lr.shape[2:])
    _, logits = model(lr_flat)  # [B*5,T,C] raw, batch-first
    t, c = logits.shape[1], logits.shape[2]
    logits = logits.reshape(b, N_FRAMES, t, c)

    if frame_fusion == "logit-sum":
        scored = fuse_logits_sum(logits)  # [B,T,C], all 5 frames
    else:
        scored = logits[:, 0, :, :]  # single-frame ablation: first frame only

    preds = scored.argmax(dim=-1)  # [B,T]
    return [ctc_greedy_decode(preds[i].tolist()) for i in range(b)]


def save_results(rows: list[dict]) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "ocr_eval_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "layout", "plate_text", "prediction", "correct"])
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def print_summary(rows: list[dict], split: str, frame_fusion: str):
    total = len(rows)
    correct = sum(r["correct"] for r in rows)
    rr = correct / total if total else 0.0
    print(f"\nSplit: {split}  frame_fusion: {frame_fusion}")
    print(f"Recognition Rate: {rr:.4f} ({correct}/{total})")

    for layout in sorted({r["layout"] for r in rows}):
        sub = [r for r in rows if r["layout"] == layout]
        sub_correct = sum(r["correct"] for r in sub)
        sub_total = len(sub)
        sub_rr = sub_correct / sub_total if sub_total else 0.0
        print(f"  {layout}: {sub_rr:.4f} ({sub_correct}/{sub_total})")


def main():
    args = parse_args()
    device = args.device or detect_device()
    print(f"Using device: {device}")

    model = load_student(args.checkpoint, device)

    ds = PlateTrackDataset(MANIFEST_CSV, split=args.split, augment=False)
    if args.limit:
        ds = Subset(ds, range(min(args.limit, len(ds))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Evaluating on {len(ds)} {args.split} tracks")

    rows = []
    for batch in loader:
        lr = batch["lr"].to(device)
        preds = predict_track_texts(model, lr, args.frame_fusion)
        for i, pred in enumerate(preds):
            rows.append(
                {
                    "track_id": batch["track_id"][i],
                    "layout": batch["layout"][i],
                    "plate_text": batch["text"][i],
                    "prediction": pred,
                    "correct": int(pred == batch["text"][i]),
                }
            )

    out_path = save_results(rows)
    print(f"Results saved to {out_path}")
    print_summary(rows, args.split, args.frame_fusion)


if __name__ == "__main__":
    main()
