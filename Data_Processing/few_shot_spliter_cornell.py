#!/usr/bin/env python3
"""Split Cornell patches into train/val/test CSVs for few-shot runs."""

import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


ROOT = Path("./Glom_Patches_0207_Cropped_cornell_1024")
OUTDIR = Path("./Glom_Patches_0207_train_val_test_cornell")

TRAIN_WSIS = ["2", "18"]
VAL_WSIS = ["17", "19"]
TEST_WSIS = "auto"

RUNS = 10
SHOTS = [1, 2, 4, 8, 16, 32]
TARGET_PER_CLASS = 32
SEED = 42
SMALL_SHOT_NO_OVERLAP_UPTO = 4

WSI_PATTERNS = [
    re.compile(r"^patch_(\d+)_\d+_\d+_image\.(png|jpg|jpeg|tif|tiff)$", re.IGNORECASE),
    re.compile(r"^patch_(\d+)[_.-]"),
]


def parse_wsi_id(name: str):
    """Extract the WSI identifier from a patch filename."""
    for pattern in WSI_PATTERNS:
        match = pattern.match(name)
        if match:
            return match.group(1)
    return None


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def scan_dataset(root: Path):
    """Return class directories grouped by WSI."""
    classes = sorted(directory.name for directory in root.iterdir() if directory.is_dir())
    by_class_wsi: Dict[str, Dict[str, List[Path]]] = {
        class_name: defaultdict(list) for class_name in classes
    }
    all_wsis = set()

    for class_name in classes:
        class_dir = root / class_name
        for file_path in class_dir.iterdir():
            if not file_path.is_file():
                continue
            wsi_id = parse_wsi_id(file_path.name)
            if not wsi_id:
                continue
            by_class_wsi[class_name][wsi_id].append(file_path)
            all_wsis.add(wsi_id)

    return classes, by_class_wsi, sorted(all_wsis, key=lambda value: (len(value), value))


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
    header = ["path", "fname", "wsi_id", "class_name"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rng_master = random.Random(SEED)

    if not ROOT.exists():
        sys.exit(f"[ERROR] ROOT does not exist: {ROOT}")

    classes, by_class_wsi, wsis_all = scan_dataset(ROOT)

    train_wsi_set = set(TRAIN_WSIS)
    val_wsi_set = set(VAL_WSIS)
    overlap = train_wsi_set & val_wsi_set
    if overlap:
        sys.exit(f"[ERROR] TRAIN_WSIS and VAL_WSIS overlap: {overlap}")

    if TEST_WSIS == "auto":
        test_wsis = [
            wsi_id for wsi_id in wsis_all if wsi_id not in train_wsi_set | val_wsi_set
        ]
    else:
        test_wsis = list(TEST_WSIS)

    for class_name in classes:
        total_train = sum(len(by_class_wsi[class_name].get(wsi_id, [])) for wsi_id in TRAIN_WSIS)
        if total_train < TARGET_PER_CLASS:
            sys.exit(
                f"[ERROR] Class '{class_name}' has {total_train} patches in train WSIs "
                f"{TRAIN_WSIS}, need {TARGET_PER_CLASS}"
            )

    val_rows = []
    test_rows = []
    for class_name in classes:
        for wsi_id in val_wsi_set:
            for source_path in sorted(by_class_wsi[class_name].get(wsi_id, [])):
                val_rows.append(
                    {
                        "path": str(source_path),
                        "fname": source_path.name,
                        "wsi_id": wsi_id,
                        "class_name": class_name,
                    }
                )
        for wsi_id in test_wsis:
            for source_path in sorted(by_class_wsi[class_name].get(wsi_id, [])):
                test_rows.append(
                    {
                        "path": str(source_path),
                        "fname": source_path.name,
                        "wsi_id": wsi_id,
                        "class_name": class_name,
                    }
                )

    write_split_csv(val_rows, OUTDIR / "val.csv")
    write_split_csv(test_rows, OUTDIR / "test.csv")

    train_pool_by_class = {}
    rr_slices_by_class = {}
    for class_name in classes:
        pool = []
        for wsi_id in TRAIN_WSIS:
            pool.extend(by_class_wsi[class_name].get(wsi_id, []))
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
                            "wsi_id": parse_wsi_id(source_path.name),
                            "class_name": class_name,
                        }
                    )

            write_split_csv(rows, run_dir / f"train_shot_{shot}.csv")


if __name__ == "__main__":
    main()
