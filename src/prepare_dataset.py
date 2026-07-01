"""
Convert LRLPR Scenario-A corners annotations to YOLO bbox format.
Produces data/yolo_dataset/{train,val}/{images,labels}/ with 90/10 split.
"""
import json
import os
import random
import shutil
from pathlib import Path

import yaml

SCENARIO_A_ROOT = Path("data/train/Scenario-A")
OUTPUT_ROOT = Path("data/yolo_dataset")
LAYOUTS = ["Brazilian", "Mercosur"]
FRAMES = [f"lr-{i:03d}.png" for i in range(1, 6)]
VAL_RATIO = 0.10
SEED = 42


def corners_to_yolo_bbox(corners: dict, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    tl = corners["top-left"]
    tr = corners["top-right"]
    br = corners["bottom-right"]
    bl = corners["bottom-left"]

    x_min = min(tl[0], bl[0])
    x_max = max(tr[0], br[0])
    y_min = min(tl[1], tr[1])
    y_max = max(bl[1], br[1])

    cx = (x_min + x_max) / 2 / img_w
    cy = (y_min + y_max) / 2 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h

    return cx, cy, w, h


def collect_tracks() -> dict[str, list[Path]]:
    """Return {'Brazilian': [...track_paths], 'Mercosur': [...track_paths]}"""
    tracks = {layout: [] for layout in LAYOUTS}
    for layout in LAYOUTS:
        layout_dir = SCENARIO_A_ROOT / layout
        for track in sorted(layout_dir.iterdir()):
            if track.is_dir():
                tracks[layout].append(track)
    return tracks


def split_tracks(tracks: dict[str, list[Path]], val_ratio: float, seed: int):
    rng = random.Random(seed)
    train_tracks, val_tracks = [], []
    for layout in LAYOUTS:
        layout_tracks = tracks[layout][:]
        rng.shuffle(layout_tracks)
        n_val = max(1, round(len(layout_tracks) * val_ratio))
        val_tracks.extend(layout_tracks[:n_val])
        train_tracks.extend(layout_tracks[n_val:])
    return train_tracks, val_tracks


def process_track(track_path: Path, split: str) -> int:
    ann_file = track_path / "annotations.json"
    with open(ann_file) as f:
        ann = json.load(f)

    corners_map = ann.get("corners", {})
    layout = track_path.parent.name
    track_id = track_path.name
    written = 0

    for frame in FRAMES:
        img_src = track_path / frame
        if not img_src.exists():
            continue
        if frame not in corners_map:
            continue

        from PIL import Image
        with Image.open(img_src) as img:
            img_w, img_h = img.size

        cx, cy, w, h = corners_to_yolo_bbox(corners_map[frame], img_w, img_h)

        # Clamp to [0, 1] to guard against sub-pixel annotation drift
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        stem = f"{layout}_{track_id}_{frame.replace('.png', '')}"
        img_dst = OUTPUT_ROOT / split / "images" / f"{stem}.png"
        lbl_dst = OUTPUT_ROOT / split / "labels" / f"{stem}.txt"

        shutil.copy2(img_src, img_dst)
        lbl_dst.write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        written += 1

    return written


def write_dataset_yaml():
    cfg = {
        "path": str(OUTPUT_ROOT.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["plate"],
    }
    with open(OUTPUT_ROOT / "dataset.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def main():
    for split in ("train", "val"):
        (OUTPUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)

    tracks = collect_tracks()
    print(f"Tracks found — Brazilian: {len(tracks['Brazilian'])}, Mercosur: {len(tracks['Mercosur'])}")

    train_tracks, val_tracks = split_tracks(tracks, VAL_RATIO, SEED)
    print(f"Split → train: {len(train_tracks)}, val: {len(val_tracks)}")

    for split_name, split_tracks_ in [("train", train_tracks), ("val", val_tracks)]:
        total = 0
        for track in split_tracks_:
            total += process_track(track, split_name)
        print(f"{split_name}: {total} images written")

    write_dataset_yaml()
    print(f"dataset.yaml written to {OUTPUT_ROOT / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
