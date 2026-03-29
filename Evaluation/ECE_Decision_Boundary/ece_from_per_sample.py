#!/usr/bin/env python3
"""
Compute Top-1 ECE summaries from per-sample prediction CSV files.

Accepted inputs:
  - {per_sample_root}/{dataset}/{method}/{model}/runNN_shotNN.csv
  - {ssl_root}/{dataset}/predictions_{dataset}_runNN_shotNN.csv
    (loaded as method=ssl, model=ssl when --include_ssl is enabled)

Run-level output:
  - {output_root}/run_shot_ece.csv

Shot-level output:
  - {output_root}/shot_statistics_ece.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_PER_SAMPLE_ROOT = "/Data3/Daniel/fewshot/result_0207/inference_results/per_sample"
DEFAULT_SSL_ROOT = "/Data3/Daniel/fewshot/result_0207/ssl"
DEFAULT_OUTPUT_ROOT = "/Data3/Daniel/fewshot/result_0207/inference_results/ece"

KNOWN_DATASETS = ["cornell", "vandy"]
CANONICAL_MODELS = ["clip", "plip", "conch", "ssl", "resnet18"]
MODEL_CHOICES = CANONICAL_MODELS
KNOWN_METHODS = ["lora", "adapter", "vanilla", "classifier", "linear_probe", "ssl", "baseline"]


def parse_run_shot_from_filename(path: str) -> Optional[Dict[str, int]]:
    match = re.search(r"run(\d+)_shot(\d+)\.csv$", os.path.basename(path))
    if not match:
        return None
    return {"run_id": int(match.group(1)), "shot": int(match.group(2))}


def normalize_model_name(model: str) -> str:
    return str(model).strip().lower()


def discover_per_sample_files(per_sample_root: str) -> List[Dict[str, object]]:
    pattern = os.path.join(per_sample_root, "*", "*", "*", "run*_shot*.csv")
    rows: List[Dict[str, object]] = []
    for path in sorted(glob.glob(pattern)):
        rel = os.path.relpath(path, per_sample_root)
        parts = rel.split(os.sep)
        if len(parts) != 4:
            continue

        dataset, method, model, _ = parts
        parsed = parse_run_shot_from_filename(path)
        if parsed is None:
            continue

        rows.append(
            {
                "dataset": dataset.lower(),
                "method": method.lower(),
                "model": normalize_model_name(model),
                "run_id": parsed["run_id"],
                "shot": parsed["shot"],
                "path": path,
                "source_type": "per_sample",
            }
        )
    return rows


def discover_ssl_files(ssl_root: str) -> List[Dict[str, object]]:
    pattern = os.path.join(ssl_root, "*", "predictions_*_run*_shot*.csv")
    rows: List[Dict[str, object]] = []
    for path in sorted(glob.glob(pattern)):
        rel = os.path.relpath(path, ssl_root)
        parts = rel.split(os.sep)
        if len(parts) != 2:
            continue

        dataset = parts[0].lower()
        parsed = parse_run_shot_from_filename(path)
        if parsed is None:
            continue

        rows.append(
            {
                "dataset": dataset,
                "method": "ssl",
                "model": "ssl",
                "run_id": parsed["run_id"],
                "shot": parsed["shot"],
                "path": path,
                "source_type": "ssl",
            }
        )
    return rows


def compute_top1_ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
    if confidence.size == 0:
        return 0.0

    confidence = np.clip(confidence, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(confidence.size)
    ece = 0.0

    for idx in range(n_bins):
        low = edges[idx]
        high = edges[idx + 1]
        if idx == 0:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence > low) & (confidence <= high)

        count = int(np.sum(mask))
        if count == 0:
            continue

        bin_acc = float(np.mean(correct[mask]))
        bin_conf = float(np.mean(confidence[mask]))
        ece += (count / total) * abs(bin_acc - bin_conf)

    return float(ece)


def summarize_one_csv(
    path: str,
    dataset: str,
    model: str,
    method: str,
    run_id: int,
    shot: int,
    n_bins: int,
) -> Optional[Dict[str, object]]:
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to read CSV: {path} | {exc}")
        return None

    if df.empty:
        print(f"[WARN] Empty CSV: {path}")
        return None

    if "true_label" not in df.columns:
        print(f"[WARN] Missing true_label column: {path}")
        return None

    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    if not prob_cols:
        print(f"[WARN] Missing prob_* columns: {path}")
        return None

    probs = df[prob_cols].to_numpy(dtype=np.float64)
    class_names = [col[len("prob_"):] for col in prob_cols]
    pred_indices = np.argmax(probs, axis=1)
    pred_from_prob = np.array([class_names[idx] for idx in pred_indices], dtype=object)
    confidence = np.max(probs, axis=1)
    true_labels = df["true_label"].astype(str).to_numpy()
    correct = (pred_from_prob == true_labels).astype(np.float64)

    mismatch_count = 0
    if "pred_label" in df.columns:
        pred_label = df["pred_label"].astype(str).to_numpy()
        mismatch_count = int(np.sum(pred_from_prob != pred_label))
        if mismatch_count > 0:
            print(f"[WARN] pred_label mismatch: {path} | mismatch_count={mismatch_count}")

    ece_top1 = compute_top1_ece(confidence, correct, n_bins=n_bins)
    num_images = int(len(df))
    mismatch_rate = float(mismatch_count / num_images) if num_images > 0 else 0.0

    return {
        "dataset": dataset,
        "model": model,
        "method": method,
        "run_id": int(run_id),
        "shot": int(shot),
        "num_images": num_images,
        "num_classes": int(len(class_names)),
        "ECE_TOP1": float(ece_top1),
        "ACC_from_prob": float(np.mean(correct)) if num_images > 0 else 0.0,
        "mean_confidence": float(np.mean(confidence)) if num_images > 0 else 0.0,
        "pred_mismatch_count": mismatch_count,
        "pred_mismatch_rate": mismatch_rate,
        "source_file": path,
    }


def aggregate_by_shot(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()

    group_cols = ["dataset", "model", "method", "shot"]
    metric_cols = ["ECE_TOP1", "ACC_from_prob", "mean_confidence", "pred_mismatch_rate"]
    grouped = run_df.groupby(group_cols, as_index=False)

    rows: List[Dict[str, object]] = []
    for keys, grp in grouped:
        dataset, model, method, shot = keys
        row: Dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "method": method,
            "shot": int(shot),
            "n_runs": int(grp["run_id"].nunique()),
            "run_ids": ",".join(str(int(x)) for x in sorted(grp["run_id"].unique())),
        }
        for col in metric_cols:
            values = grp[col].to_numpy(dtype=np.float64)
            row[f"{col}_mean"] = float(np.mean(values)) if values.size else 0.0
            row[f"{col}_std"] = float(np.std(values, ddof=0)) if values.size else 0.0
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["dataset", "model", "method", "shot"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Top-1 ECE from per-sample prediction CSV files.")
    parser.add_argument("--per_sample_root", default=DEFAULT_PER_SAMPLE_ROOT)
    parser.add_argument("--ssl_root", default=DEFAULT_SSL_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", default="all", choices=KNOWN_DATASETS + ["all"])
    parser.add_argument("--model", default="all", choices=MODEL_CHOICES + ["all"])
    parser.add_argument("--method", default="all", choices=KNOWN_METHODS + ["all"])
    parser.add_argument("--include_ssl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n_bins", type=int, default=15)
    parser.add_argument("--binning", choices=["uniform"], default="uniform")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round_digits", type=int, default=6)
    args = parser.parse_args()

    if args.n_bins <= 0:
        raise ValueError("--n_bins must be > 0")

    np.random.seed(args.seed)

    all_files = discover_per_sample_files(args.per_sample_root)
    if args.include_ssl:
        all_files.extend(discover_ssl_files(args.ssl_root))

    datasets = KNOWN_DATASETS if args.dataset == "all" else [args.dataset]
    models = CANONICAL_MODELS if args.model == "all" else [normalize_model_name(args.model)]
    methods = KNOWN_METHODS if args.method == "all" else [args.method]

    selected = [
        row
        for row in all_files
        if row["dataset"] in datasets and row["model"] in models and row["method"] in methods
    ]

    if not selected:
        print("[WARN] No input CSV files found after filtering.")
        return

    run_rows: List[Dict[str, object]] = []
    skipped = 0
    total_mismatch = 0
    for item in selected:
        result = summarize_one_csv(
            path=str(item["path"]),
            dataset=str(item["dataset"]),
            model=str(item["model"]),
            method=str(item["method"]),
            run_id=int(item["run_id"]),
            shot=int(item["shot"]),
            n_bins=args.n_bins,
        )
        if result is None:
            skipped += 1
            continue
        total_mismatch += int(result["pred_mismatch_count"])
        run_rows.append(result)

    if not run_rows:
        print("[WARN] No valid run-shot rows produced.")
        return

    run_df = pd.DataFrame(run_rows).sort_values(
        ["dataset", "model", "method", "shot", "run_id"]
    ).reset_index(drop=True)
    shot_df = aggregate_by_shot(run_df)

    run_float_cols = ["ECE_TOP1", "ACC_from_prob", "mean_confidence", "pred_mismatch_rate"]
    shot_float_cols = [col for col in shot_df.columns if col.endswith("_mean") or col.endswith("_std")]

    run_df[run_float_cols] = run_df[run_float_cols].round(args.round_digits)
    if not shot_df.empty and shot_float_cols:
        shot_df[shot_float_cols] = shot_df[shot_float_cols].round(args.round_digits)

    os.makedirs(args.output_root, exist_ok=True)
    run_path = os.path.join(args.output_root, "run_shot_ece.csv")
    shot_path = os.path.join(args.output_root, "shot_statistics_ece.csv")
    run_df.to_csv(run_path, index=False)
    shot_df.to_csv(shot_path, index=False)

    print(
        f"[OK] processed={len(selected)} valid={len(run_rows)} skipped={skipped} "
        f"mismatch_total={total_mismatch}"
    )
    print(f"[OK] run-shot ECE -> {run_path}")
    print(f"[OK] shot statistics -> {shot_path}")


if __name__ == "__main__":
    main()
