"""Dataset and augmentation for the LRLPR teacher-student OCR pipeline."""
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ocr.vocab import encode_text

LR_SIZE = (100, 32)  # (W, H) canonical size, ~mean aspect ratio of raw LR crops
SR_SCALE = 2
HR_SIZE = (LR_SIZE[0] * SR_SCALE, LR_SIZE[1] * SR_SCALE)  # (200, 64)


def _load_image_tensor(path: str, size: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB").resize(size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3]
    return torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W]


def random_occlusion(img: torch.Tensor, prob: float = 0.3) -> torch.Tensor:
    """Random vertical or horizontal occlusion bar, applied in-place on a copy."""
    if random.random() >= prob:
        return img
    img = img.clone()
    _, h, w = img.shape
    fill = img.mean()
    if random.random() < 0.5:  # vertical bar
        bar_w = int(random.uniform(0.1, 0.3) * w)
        x0 = random.randint(0, max(0, w - bar_w))
        img[:, :, x0:x0 + bar_w] = fill
    else:  # horizontal bar
        bar_h = int(random.uniform(0.1, 0.3) * h)
        y0 = random.randint(0, max(0, h - bar_h))
        img[:, y0:y0 + bar_h, :] = fill
    return img


class PlateTrackDataset(Dataset):
    """One item = one track: 5 LR frames (+ 5 HR frames if available)."""

    def __init__(self, manifest_csv: str, split: str, augment: bool = False, mask_prob: float = 0.3):
        df = pd.read_csv(manifest_csv)
        self.rows = df[df["split"] == split].reset_index(drop=True)
        self.augment = augment
        self.mask_prob = mask_prob

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]

        lr_paths = row["lr_frames"].split(";")
        lr_frames = [_load_image_tensor(p, LR_SIZE) for p in lr_paths]
        if self.augment:
            lr_frames = [random_occlusion(f, self.mask_prob) for f in lr_frames]
        lr = torch.stack(lr_frames)  # [5,3,H,W]

        has_hr = bool(row["has_hr"])
        if has_hr:
            hr_paths = row["hr_frames"].split(";")
            hr = torch.stack([_load_image_tensor(p, HR_SIZE) for p in hr_paths])  # [5,3,H,W]
        else:
            hr = torch.zeros(5, 3, HR_SIZE[1], HR_SIZE[0])

        target = torch.tensor(encode_text(row["plate_text"]), dtype=torch.long)  # [7]

        return {
            "lr": lr,
            "hr": hr,
            "has_hr": has_hr,
            "target": target,
            "text": row["plate_text"],
            "layout": row["layout"],
            "track_id": row["track_id"],
        }
