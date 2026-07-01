"""
Train the teacher-student SR+OCR model on the LRLPR-26 dataset.

Run src/prepare_ocr_dataset.py first to generate data/ocr_manifest.csv.
Use --smoke-test for a fast CPU-only correctness check before a real run.
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ocr.data import LR_SIZE, PlateTrackDataset
from ocr.model import OCRConfig, SRConfig
from ocr.teacher_student import (
    TeacherStudentOCR,
    ctc_loss_fn,
    distillation_loss_fn,
    fuse_logits_sum,
    sr_recon_loss_fn,
)
from ocr.vocab import ctc_greedy_decode

MANIFEST_CSV = "data/ocr_manifest.csv"
RUNS_DIR = Path("runs/ocr")
N_FRAMES = 5


def detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="run")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--lambda-sr", type=float, default=0.1)
    p.add_argument("--distill-max-weight", type=float, default=0.5)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--mask-prob", type=float, default=0.3)
    p.add_argument("--frame-fusion", choices=["none", "logit-sum"], default="none")
    p.add_argument("--no-teacher", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument(
        "--no-multi-gpu", action="store_true",
        help="Disable DataParallel even if multiple CUDA devices are visible.",
    )
    return p.parse_args()


def build_model(smoke: bool) -> tuple[TeacherStudentOCR, SRConfig, OCRConfig]:
    sr_cfg = SRConfig.tiny() if smoke else SRConfig()
    ocr_cfg = OCRConfig.tiny() if smoke else OCRConfig()
    return TeacherStudentOCR(sr_cfg, ocr_cfg), sr_cfg, ocr_cfg


def unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, torch.nn.DataParallel) else module


def wrap_multi_gpu(model: TeacherStudentOCR, device: str, disabled: bool) -> TeacherStudentOCR:
    """Wrap student/teacher in DataParallel when multiple CUDA devices are visible.

    SRPlateNet's outputs are batch-first (dim 0), so DataParallel's default gather
    (concat along dim 0) reassembles them correctly with no further changes needed.
    """
    if disabled or device != "cuda" or torch.cuda.device_count() < 2:
        return model
    n = torch.cuda.device_count()
    print(f"Using {n} GPUs via DataParallel")
    model.student = torch.nn.DataParallel(model.student)
    model.teacher = torch.nn.DataParallel(model.teacher)
    return model


def build_datasets(args: argparse.Namespace) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    train_ds = PlateTrackDataset(MANIFEST_CSV, split="train", augment=True, mask_prob=args.mask_prob)
    val_ds = PlateTrackDataset(MANIFEST_CSV, split="val", augment=False)
    if args.smoke_test:
        train_ds = Subset(train_ds, range(min(32, len(train_ds))))
        val_ds = Subset(val_ds, range(min(16, len(val_ds))))
    return train_ds, val_ds


def distill_weight(step: int, warmup_steps: int, max_weight: float) -> float:
    if warmup_steps <= 0:
        return max_weight
    return max_weight * min(1.0, step / warmup_steps)


def compute_losses(
    model: TeacherStudentOCR, batch: dict, args: argparse.Namespace, global_step: int, device: str
) -> tuple[torch.Tensor, dict]:
    lr = batch["lr"].to(device)  # [B,5,3,H,W]
    hr = batch["hr"].to(device)  # [B,5,3,H,W]
    has_hr = batch["has_hr"].to(device)  # [B]
    target = batch["target"].to(device)  # [B,7]

    b = lr.shape[0]
    lr_flat = lr.reshape(b * N_FRAMES, *lr.shape[2:])
    hr_flat = hr.reshape(b * N_FRAMES, *hr.shape[2:])
    has_hr_flat = has_hr.repeat_interleave(N_FRAMES)

    sr_s, logits_s = model.student(lr_flat)  # sr_s: [B*5,3,H,W], logits_s: [B*5,T,C] raw, batch-first

    logs = {}
    if args.frame_fusion == "logit-sum":
        t, c = logits_s.shape[1], logits_s.shape[2]
        logits_fused = fuse_logits_sum(logits_s.reshape(b, N_FRAMES, t, c))  # [B,T,C]
        l_ctc = ctc_loss_fn(logits_fused, target)
    else:
        target_flat = target.repeat_interleave(N_FRAMES, dim=0)  # [B*5,7]
        l_ctc = ctc_loss_fn(logits_s, target_flat)
    total = l_ctc
    logs["ctc"] = l_ctc.item()

    if has_hr_flat.any():
        hr_sub = hr_flat[has_hr_flat]
        sr_s_sub = sr_s[has_hr_flat]
        l_sr = sr_recon_loss_fn(sr_s_sub, hr_sub)
        total = total + args.lambda_sr * l_sr
        logs["sr"] = l_sr.item()

        if not args.no_teacher:
            hr_down = F.interpolate(
                hr_sub, size=(LR_SIZE[1], LR_SIZE[0]), mode="bilinear", align_corners=False
            )
            with torch.no_grad():
                _, logits_t = model.teacher(hr_down)
            logits_s_sub = logits_s[has_hr_flat]
            w = distill_weight(global_step, args.warmup_steps, args.distill_max_weight)
            l_distill = distillation_loss_fn(logits_s_sub, logits_t)
            total = total + w * l_distill
            logs["distill"] = l_distill.item()

    return total, logs


@torch.no_grad()
def evaluate_recognition_rate(model: TeacherStudentOCR, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        lr = batch["lr"].to(device)  # [B,5,3,H,W]
        texts = batch["text"]
        b = lr.shape[0]
        lr_flat = lr.reshape(b * N_FRAMES, *lr.shape[2:])
        _, logits = model.student(lr_flat)  # [B*5,T,C]
        t, c = logits.shape[1], logits.shape[2]
        fused = fuse_logits_sum(logits.reshape(b, N_FRAMES, t, c))  # [B,T,C]
        preds = fused.argmax(dim=-1)  # [B,T]
        for i in range(b):
            pred_text = ctc_greedy_decode(preds[i].tolist())
            correct += int(pred_text == texts[i])
            total += 1
    model.train()
    return correct / total if total else 0.0


def save_checkpoint(
    model: TeacherStudentOCR, sr_cfg: SRConfig, ocr_cfg: OCRConfig, optimizer, epoch: int, path: Path
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student": unwrap(model.student).state_dict(),
            "teacher": unwrap(model.teacher).state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "sr_cfg": sr_cfg,
            "ocr_cfg": ocr_cfg,
        },
        path,
    )


def main():
    args = parse_args()
    if args.smoke_test:
        args.epochs = 2
        args.batch_size = 4
        args.frame_fusion = "none"
        args.device = "cpu"
        args.num_workers = 0
        args.val_every = 1
        if args.name == "run":
            args.name = "smoke"

    device = args.device or detect_device()
    print(f"Using device: {device}")

    train_ds, val_ds = build_datasets(args)
    print(f"train tracks: {len(train_ds)}, val tracks: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model, sr_cfg, ocr_cfg = build_model(args.smoke_test)
    model = model.to(device)
    model = wrap_multi_gpu(model, device, args.no_multi_gpu)
    optimizer = torch.optim.AdamW(model.student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(1, int(0.05 * total_steps))
    warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
    )

    run_dir = RUNS_DIR / args.name

    global_step = 0
    best_rr = -1.0
    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.perf_counter()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, logs = compute_losses(model, batch, args, global_step, device)
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.ema_update(args.ema_decay)
            running_loss += loss.item()
            global_step += 1

        avg_loss = running_loss / steps_per_epoch
        elapsed = time.perf_counter() - epoch_start
        print(
            f"epoch {epoch + 1}/{args.epochs} loss={avg_loss:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e} ({elapsed:.1f}s)"
        )

        is_last = epoch == args.epochs - 1
        if (epoch + 1) % args.val_every == 0 or is_last:
            rr = evaluate_recognition_rate(model, val_loader, device)
            print(f"  val recognition_rate={rr:.4f}")
            save_checkpoint(model, sr_cfg, ocr_cfg, optimizer, epoch, run_dir / f"checkpoint_epoch{epoch + 1}.pt")
            if rr > best_rr:
                best_rr = rr
                save_checkpoint(model, sr_cfg, ocr_cfg, optimizer, epoch, run_dir / "best.pt")

    print(f"Training complete. Best val recognition_rate={best_rr:.4f}")


if __name__ == "__main__":
    main()
