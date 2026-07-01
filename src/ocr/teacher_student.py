"""Teacher-student wrapper: an EMA teacher (fed downsampled HR) supervises the LR student."""
import copy

import torch
import torch.nn.functional as F
from torch import nn

from ocr.data import LR_SIZE
from ocr.model import OCRConfig, SRConfig, SRPlateNet
from ocr.vocab import BLANK_IDX


class TeacherStudentOCR(nn.Module):
    def __init__(self, sr_cfg: SRConfig, ocr_cfg: OCRConfig):
        super().__init__()
        self.student = SRPlateNet(sr_cfg, ocr_cfg)
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def forward(self, lr_img: torch.Tensor, hr_img: torch.Tensor | None = None) -> dict:
        sr_s, logits_s = self.student(lr_img)
        out = {"sr_student": sr_s, "logits_student": logits_s}
        if hr_img is not None:
            hr_down = F.interpolate(
                hr_img, size=(LR_SIZE[1], LR_SIZE[0]), mode="bilinear", align_corners=False
            )
            with torch.no_grad():
                sr_t, logits_t = self.teacher(hr_down)
            out["sr_teacher"] = sr_t
            out["logits_teacher"] = logits_t
        return out

    @torch.no_grad()
    def ema_update(self, decay: float):
        for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
            pt.mul_(decay).add_(ps, alpha=1 - decay)
        for bt, bs in zip(self.teacher.buffers(), self.student.buffers()):
            bt.copy_(bs)


def ctc_loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits: [B,T,C] raw (pre-softmax), batch-first. targets: [B,L] encoded plate text (fixed length L)."""
    log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # CTC requires time-major [T,B,C]
    t, b, _ = log_probs.shape
    input_lengths = torch.full((b,), t, dtype=torch.long, device=logits.device)
    target_lengths = torch.full((b,), targets.shape[1], dtype=torch.long, device=logits.device)
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=BLANK_IDX, zero_infinity=True)


def distillation_loss_fn(logits_student: torch.Tensor, logits_teacher: torch.Tensor) -> torch.Tensor:
    """Per-timestep mean-teacher KL divergence between student and (detached) teacher distributions.
    Both inputs: [B,T,C] raw (pre-softmax), batch-first."""
    c = logits_student.shape[-1]
    student_log_probs = F.log_softmax(logits_student.reshape(-1, c), dim=-1)
    teacher_probs = F.softmax(logits_teacher.reshape(-1, c), dim=-1).detach()
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def fuse_logits_sum(logits_per_frame: torch.Tensor) -> torch.Tensor:
    """logits_per_frame: [B,n_frames,T,C] raw logits -> [B,T,C] summed across frames (pre-softmax)."""
    return logits_per_frame.sum(dim=1)


def sr_recon_loss_fn(sr_student: torch.Tensor, hr_target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(sr_student, hr_target)
