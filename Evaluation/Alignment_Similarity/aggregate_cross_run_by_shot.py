#!/usr/bin/env python3
"""
Aggregate alignment and similarity summaries across runs for each shot.

Inputs:
  - {summary_dir}/{dataset}_summary.csv
  - {summary_dir}/{dataset}_baseline_summary.csv

Output:
  - {output_dir}/{dataset}_statistics_summary.csv
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_SUMMARY_DIR = "/Data3/Daniel/fewshot/result_0207/inference_results/alignment_similarity/summary"
DEFAULT_OUTPUT_DIR = "/Data3/Daniel/fewshot/result_0207/inference_results/alignment_similarity/run_comparison"
KNOWN_DATASETS = ["cornell", "vandy"]


def load_summary(path: str, dataset: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[WARN] Missing summary file: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        print(f"[WARN] Empty summary file: {path}")
        return df

    if "dataset" not in df.columns:
        df["dataset"] = dataset
    else:
        df["dataset"] = df["dataset"].astype(str).str.lower()

    return df


def coerce_numeric_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_dataset_aggregate(dataset_df: pd.DataFrame, round_digits: int) -> pd.DataFrame:
    group_cols = ["dataset", "model", "method", "shot"]
    excluded_cols = set(group_cols + ["run_id"])
    numeric_cols = [
        col
        for col in dataset_df.columns
        if col not in excluded_cols and pd.api.types.is_numeric_dtype(dataset_df[col])
    ]

    def std0(values: pd.Series) -> float:
        return float(np.std(values, ddof=0)) if len(values) else 0.0

    std0.__name__ = "std"
    agg_spec = {col: ["mean", std0] for col in numeric_cols}

    grouped = dataset_df.groupby(group_cols, as_index=False)
    stats = grouped.agg(agg_spec)
    stats.columns = [
        "_".join([part for part in col if part]).rstrip("_") if isinstance(col, tuple) else col
        for col in stats.columns
    ]

    run_meta = grouped["run_id"].agg(
        n_runs=lambda s: int(pd.Series(s).nunique()),
        run_ids=lambda s: ",".join(str(int(x)) for x in sorted(pd.Series(s).dropna().unique())),
    ).reset_index()
    run_meta = run_meta.drop(columns=["index"], errors="ignore")

    out = stats.merge(run_meta, on=group_cols, how="left")
    std_cols = [col for col in out.columns if col.endswith("_std")]
    out[std_cols] = out[std_cols].fillna(0.0)

    metric_cols = [col for col in out.columns if col.endswith("_mean") or col.endswith("_std")]
    if metric_cols:
        out[metric_cols] = out[metric_cols].round(round_digits)

    return out.sort_values(["model", "method", "shot"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate alignment and similarity summaries across runs by shot.")
    parser.add_argument("--summary_dir", default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", default="all", choices=["cornell", "vandy", "all"])
    parser.add_argument("--round", type=int, default=4, dest="round_digits")
    args = parser.parse_args()

    datasets = KNOWN_DATASETS if args.dataset == "all" else [args.dataset]
    os.makedirs(args.output_dir, exist_ok=True)

    for dataset in datasets:
        finetune_path = os.path.join(args.summary_dir, f"{dataset}_summary.csv")
        baseline_path = os.path.join(args.summary_dir, f"{dataset}_baseline_summary.csv")

        parts = []
        finetune_df = load_summary(finetune_path, dataset)
        baseline_df = load_summary(baseline_path, dataset)

        if not finetune_df.empty:
            parts.append(finetune_df)
        if not baseline_df.empty:
            parts.append(baseline_df)

        if not parts:
            print(f"[WARN] No data for dataset={dataset}")
            continue

        merged = pd.concat(parts, ignore_index=True, sort=False)
        merged["dataset"] = merged["dataset"].astype(str).str.lower()
        merged["model"] = merged["model"].astype(str).str.lower()
        merged["method"] = merged["method"].astype(str).str.lower()
        merged["shot"] = pd.to_numeric(merged["shot"], errors="coerce").fillna(-1).astype(int)
        merged["run_id"] = pd.to_numeric(merged["run_id"], errors="coerce").fillna(0).astype(int)

        numeric_candidates = [col for col in merged.columns if col not in ["dataset", "model", "method", "shot", "run_id"]]
        merged = coerce_numeric_cols(merged, numeric_candidates)

        out = build_dataset_aggregate(merged, args.round_digits)
        output_path = os.path.join(args.output_dir, f"{dataset}_statistics_summary.csv")
        out.to_csv(output_path, index=False)

        print(f"[OK] {dataset}: groups={len(out)} -> {output_path}")


if __name__ == "__main__":
    main()
