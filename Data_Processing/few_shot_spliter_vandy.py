#!/usr/bin/env python3
"""Split Vandy patches into train/val/test CSVs for few-shot runs."""

import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


ROOT = Path("./Glom_Patches_0207_Cropped_vandy_1024")
OUTDIR = Path("./Glom_Patches_0207_train_val_test_vandy")

TRAIN_SLIDES = [
    "22861_2017-04-08 12_12_09",
    "26835_2018-08-09 09_21_15",
]
VAL_SLIDES = [
    "25118_2017-04-07 23_48_31",
    "36551_2017-04-08 02_24_05",
]
TEST_SLIDES = "auto"

RUNS = 10
SHOTS = [1, 2, 4, 8, 16, 32]
TARGET_PER_CLASS = 32
SEED = 42
SMALL_SHOT_NO_OVERLAP_UPTO = 4

SLIDE_ID_PATTERN = re.compile(r"^(\d+_\d{4}-\d{2}-\d{2} \d{2}_\d{2}_\d{2})")


def parse_slide_id(name: str):
    """Extract the slide identifier from a patch filename."""
    match = SLIDE_ID_PATTERN.match(name)
    if match:
        return match.group(1)
    return None


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def scan_dataset(root: Path):
    """Return class directories grouped by slide."""
    classes = sorted(directory.name for directory in root.iterdir() if directory.is_dir())
    by_class_slide: Dict[str, Dict[str, List[Path]]] = {
        class_name: defaultdict(list) for class_name in classes
    }
    all_slides = set()

    for class_name in classes:
        class_dir = root / class_name
        for file_path in class_dir.iterdir():
            if not file_path.is_file():
                continue
            slide_id = parse_slide_id(file_path.name)
            if not slide_id:
                continue
            by_class_slide[class_name][slide_id].append(file_path)
            all_slides.add(slide_id)

    return classes, by_class_slide, sorted(all_slides)


def round_robin_slices(pool, runs, seed):
    """Split a shuffled pool into non-overlapping pieces across runs."""
    rng = random.Random(seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    if total == 0:
        return [[] for _ in range(runs)]

    chunk = (total + runs - 1) // runs
    return [
        shuffled[index * chunk:min((index + 1) * chunk, total)]
        for index in range(runs)
    ]


def write_split_csv(rows, path: Path):
    ensure_dir(path.parent)
    header = ["path", "fname", "slide_id", "class_name"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rng_master = random.Random(SEED)

    if not ROOT.exists():
        sys.exit(f"[ERROR] ROOT does not exist: {ROOT}")

    classes, by_class_slide, slides_all = scan_dataset(ROOT)

    train_slide_set = set(TRAIN_SLIDES)
    val_slide_set = set(VAL_SLIDES)
    overlap = train_slide_set & val_slide_set
    if overlap:
        sys.exit(f"[ERROR] TRAIN_SLIDES and VAL_SLIDES overlap: {overlap}")

    for slide_id in TRAIN_SLIDES:
        if slide_id not in slides_all:
            sys.exit(f"[ERROR] Train slide not found in dataset: {slide_id}")

    for slide_id in VAL_SLIDES:
        if slide_id not in slides_all:
            sys.exit(f"[ERROR] Val slide not found in dataset: {slide_id}")

    if TEST_SLIDES == "auto":
        test_slides = [
            slide_id for slide_id in slides_all if slide_id not in train_slide_set | val_slide_set
        ]
    else:
        test_slides = list(TEST_SLIDES)

    for class_name in classes:
        total_train = sum(len(by_class_slide[class_name].get(slide_id, [])) for slide_id in TRAIN_SLIDES)
        if total_train < TARGET_PER_CLASS:
            sys.exit(
                f"[ERROR] Class '{class_name}' has {total_train} patches in train slides, "
                f"need {TARGET_PER_CLASS}"
            )

    val_rows = []
    test_rows = []
    for class_name in classes:
        for slide_id in val_slide_set:
            for source_path in sorted(by_class_slide[class_name].get(slide_id, [])):
                val_rows.append(
                    {
                        "path": str(source_path),
                        "fname": source_path.name,
                        "slide_id": slide_id,
                        "class_name": class_name,
                    }
                )
        for slide_id in test_slides:
            for source_path in sorted(by_class_slide[class_name].get(slide_id, [])):
                test_rows.append(
                    {
                        "path": str(source_path),
                        "fname": source_path.name,
                        "slide_id": slide_id,
                        "class_name": class_name,
                    }
                )

    write_split_csv(val_rows, OUTDIR / "val.csv")
    write_split_csv(test_rows, OUTDIR / "test.csv")

    train_pool_by_class = {}
    rr_slices_by_class = {}
    for class_name in classes:
        pool = []
        for slide_id in TRAIN_SLIDES:
            pool.extend(by_class_slide[class_name].get(slide_id, []))
        rng_master.shuffle(pool)
        train_pool_by_class[class_name] = pool

        class_seed = SEED + classes.index(class_name)
        rr_slices_by_class[class_name] = round_robin_slices(pool, RUNS, seed=class_seed)

    for run_index in range(RUNS):
        run_dir = OUTDIR / f"run_{run_index + 1:02d}"
        ensure_dir(run_dir)

        per_class_32 = {}
        for class_name in classes:
            pool = train_pool_by_class[class_name][:]
            rr_piece = rr_slices_by_class[class_name][run_index]
            picked = rr_piece[:min(len(rr_piece), TARGET_PER_CLASS)]

            if len(picked) < TARGET_PER_CLASS:
                seen = set(picked)
                for patch_path in pool:
                    if patch_path in seen:
                        continue
                    picked.append(patch_path)
                    if len(picked) >= TARGET_PER_CLASS:
                        break

            per_class_32[class_name] = picked

        for shot in sorted(SHOTS):
            rows = []
            for class_name in classes:
                base = per_class_32[class_name][:]
                rng = random.Random(f"{SEED}-{run_index}-{classes.index(class_name)}-{shot}")
                rng.shuffle(base)
                take = min(shot, len(base))
                subset = base[:take]

                if shot <= SMALL_SHOT_NO_OVERLAP_UPTO:
                    prefer = rr_slices_by_class[class_name][run_index][
                        :min(shot, len(rr_slices_by_class[class_name][run_index]))
                    ]
                    seen = set(prefer)
                    subset = prefer + [item for item in subset if item not in seen]
                    subset = subset[:take]

                for source_path in subset:
                    rows.append(
                        {
                            "path": str(source_path),
                            "fname": source_path.name,
                            "slide_id": parse_slide_id(source_path.name),
                            "class_name": class_name,
                        }
                    )

            write_split_csv(rows, run_dir / f"train_shot_{shot}.csv")


if __name__ == "__main__":
    main()
