"""
Build data/ocr_manifest.csv from LRLPR-26 tracks (Scenario-A, Scenario-B, test).

Scenario-A -> split "train". Scenario-B -> stratified 90/10 split (by layout)
into "train"/"val", mirroring the DLmath winning-team split (10,000 A +
9,000 B train, 1,000 B held-out val). data/test -> split "test", reserved
for the final Recognition Rate benchmark only.
"""
import csv
import json
import random
from pathlib import Path

TRAIN_ROOT = Path("data/train")
TEST_ROOT = Path("data/test")
OUTPUT_CSV = Path("data/ocr_manifest.csv")

LAYOUTS = ["Brazilian", "Mercosur"]
ALPHABET = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
PLATE_LEN = 7
VAL_RATIO = 0.10  # applied only within Scenario-B
SEED = 42

FIELDNAMES = [
    "track_id", "scenario", "layout", "plate_text", "split",
    "has_hr", "lr_frames", "hr_frames",
]


def find_frames(track_path: Path, prefix: str) -> list[Path]:
    frames = []
    for i in range(1, 6):
        for ext in ("png", "jpg"):
            p = track_path / f"{prefix}-{i:03d}.{ext}"
            if p.exists():
                frames.append(p)
                break
    return frames


def load_track(track_path: Path, scenario: str) -> dict | None:
    ann_file = track_path / "annotations.json"
    with open(ann_file) as f:
        ann = json.load(f)

    text = ann.get("plate_text", "").strip()
    layout = ann.get("plate_layout", "")
    if len(text) != PLATE_LEN or not set(text) <= ALPHABET:
        print(f"Skipping {track_path}: invalid plate_text {ann.get('plate_text')!r}")
        return None

    lr_frames = find_frames(track_path, "lr")
    if len(lr_frames) != 5:
        print(f"Skipping {track_path}: found {len(lr_frames)} lr frames, expected 5")
        return None
    hr_frames = find_frames(track_path, "hr")
    has_hr = len(hr_frames) == 5

    return {
        "track_id": track_path.name,
        "scenario": scenario,
        "layout": layout,
        "plate_text": text,
        "has_hr": has_hr,
        "lr_frames": ";".join(str(p) for p in lr_frames),
        "hr_frames": ";".join(str(p) for p in hr_frames) if has_hr else "",
    }


def collect_scenario(scenario: str) -> dict[str, list[dict]]:
    rows = {layout: [] for layout in LAYOUTS}
    for layout in LAYOUTS:
        layout_dir = TRAIN_ROOT / scenario / layout
        for track_path in sorted(layout_dir.iterdir()):
            if not track_path.is_dir():
                continue
            row = load_track(track_path, scenario)
            if row is not None:
                rows[layout].append(row)
    return rows


def collect_test() -> list[dict]:
    rows = []
    for track_path in sorted(TEST_ROOT.iterdir()):
        if not track_path.is_dir():
            continue
        row = load_track(track_path, "test")
        if row is not None:
            rows.append(row)
    return rows


def split_scenario_b(rows_by_layout: dict[str, list[dict]], val_ratio: float, seed: int) -> list[dict]:
    """Stratified per-layout shuffle+split, same pattern as prepare_dataset.py's split_tracks()."""
    rng = random.Random(seed)
    out = []
    for layout in LAYOUTS:
        layout_rows = rows_by_layout[layout][:]
        rng.shuffle(layout_rows)
        n_val = max(1, round(len(layout_rows) * val_ratio))
        for row in layout_rows[:n_val]:
            row["split"] = "val"
            out.append(row)
        for row in layout_rows[n_val:]:
            row["split"] = "train"
            out.append(row)
    return out


def write_manifest(rows: list[dict]):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    scenario_a = collect_scenario("Scenario-A")
    rows_a = [row for layout in LAYOUTS for row in scenario_a[layout]]
    for row in rows_a:
        row["split"] = "train"
    print(f"Scenario-A: {len(rows_a)} tracks -> train")

    scenario_b = collect_scenario("Scenario-B")
    rows_b = split_scenario_b(scenario_b, VAL_RATIO, SEED)
    n_train_b = sum(1 for r in rows_b if r["split"] == "train")
    n_val_b = sum(1 for r in rows_b if r["split"] == "val")
    print(f"Scenario-B: {n_train_b} tracks -> train, {n_val_b} tracks -> val")

    rows_test = collect_test()
    for row in rows_test:
        row["split"] = "test"
    print(f"Test: {len(rows_test)} tracks -> test")

    all_rows = rows_a + rows_b + rows_test
    write_manifest(all_rows)
    print(f"\nTotal: {len(all_rows)} tracks written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
