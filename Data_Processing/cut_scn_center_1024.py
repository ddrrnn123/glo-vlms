#!/usr/bin/env python3

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import openslide


ID_PATTERN = re.compile(r"^(\d+_\d{4}-\d{2}-\d{2} \d{2}_\d{2}_\d{2})")
COORD_PATTERN = re.compile(r"-x-([0-9]+(?:\.[0-9]+)?)")
MISSING_LIST_PATH = "./missing_wsi_ids.txt"
PATCH_SIZE = 1024


def extract_slide_id(filename: str) -> str | None:
    match = ID_PATTERN.search(filename)
    return match.group(1) if match else None


def parse_coords_from_filename(filename: str) -> tuple[int, int, int, int]:
    tokens = COORD_PATTERN.findall(filename)
    if len(tokens) < 4:
        raise ValueError(
            f"Expected at least 4 numeric tokens after '-x-' in filename, got {len(tokens)}"
        )

    x_str, y_str, w_str, h_str = tokens[-4:]
    x = int(float(x_str))
    y = int(float(y_str))
    w = int(float(w_str))
    h = int(float(h_str))
    return x, y, w, h


def load_missing_prefixes(path: str) -> set[str]:
    missing = set()
    missing_path = Path(path)
    if not missing_path.exists():
        print(f"[WARN] Missing list not found: {missing_path} (continuing without it)")
        return missing

    for line in missing_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        missing.add(line)

    return missing


def build_wsi_index(wsi_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in wsi_root.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() != ".scn":
            continue

        slide_id = extract_slide_id(path.name)
        if slide_id is None:
            continue

        index[slide_id] = path

    return index


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-cut SCN patches around the original box center using a fixed 1024x1024 crop."
    )
    parser.add_argument(
        "--patch-root",
        default="./Glom_patch_vandy",
        help="Root directory of the existing patch dataset.",
    )
    parser.add_argument(
        "--wsi-root",
        default="./vandy_WSI",
        help="Directory containing SCN files.",
    )
    parser.add_argument(
        "--out",
        default="./Glom_patch_vandy_center1024",
        help="Output directory for the re-cut dataset.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip patches whose output files already exist.",
    )
    return parser


def clamp_origin(origin: int, max_origin: int) -> int:
    if max_origin < 0:
        return 0
    if origin < 0:
        return 0
    if origin > max_origin:
        return max_origin
    return origin


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    patch_root = Path(args.patch_root)
    wsi_root = Path(args.wsi_root)
    out_root = Path(args.out)

    if not patch_root.exists():
        print(f"[ERROR] Patch root not found: {patch_root}", file=sys.stderr)
        return 1
    if not wsi_root.exists():
        print(f"[ERROR] WSI root not found: {wsi_root}", file=sys.stderr)
        return 1

    out_root.mkdir(parents=True, exist_ok=True)

    missing_prefixes = load_missing_prefixes(MISSING_LIST_PATH)
    wsi_index = build_wsi_index(wsi_root)

    patches_by_slide: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in patch_root.rglob("*.png"):
        if not path.is_file():
            continue

        slide_id = extract_slide_id(path.name)
        if slide_id is None:
            continue

        try:
            x, y, w, h = parse_coords_from_filename(path.name)
        except ValueError:
            continue

        patches_by_slide[slide_id].append(
            {
                "src_path": path,
                "class_name": path.parent.name,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        )

    for slide_id, records in patches_by_slide.items():
        prefix = slide_id.split(" ")[0]
        if prefix in missing_prefixes or slide_id in missing_prefixes:
            continue

        wsi_path = wsi_index.get(slide_id)
        if wsi_path is None:
            continue

        try:
            slide = openslide.OpenSlide(str(wsi_path))
        except Exception as exc:
            print(f"[ERROR] OpenSlide cannot open: {wsi_path} ({exc})")
            continue

        slide_width, slide_height = slide.dimensions
        max_x = max(0, slide_width - PATCH_SIZE)
        max_y = max(0, slide_height - PATCH_SIZE)

        for record in records:
            x = int(record["x"])
            y = int(record["y"])
            width = int(record["w"])
            height = int(record["h"])
            if width <= 0 or height <= 0:
                continue

            center_x = x + (width / 2.0)
            center_y = y + (height / 2.0)
            origin_x = int(round(center_x - (PATCH_SIZE / 2.0)))
            origin_y = int(round(center_y - (PATCH_SIZE / 2.0)))

            origin_x = clamp_origin(origin_x, max_x)
            origin_y = clamp_origin(origin_y, max_y)

            class_name = str(record["class_name"])
            out_dir = out_root / class_name
            out_dir.mkdir(parents=True, exist_ok=True)

            source_path = Path(record["src_path"])
            out_path = out_dir / source_path.name

            if args.skip_existing and out_path.exists():
                continue

            try:
                patch = slide.read_region((origin_x, origin_y), 0, (PATCH_SIZE, PATCH_SIZE))
            except Exception as exc:
                print(f"[ERROR] read_region failed for {source_path.name}: {exc}")
                continue

            patch.convert("RGB").save(out_path, format="PNG")

        slide.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
