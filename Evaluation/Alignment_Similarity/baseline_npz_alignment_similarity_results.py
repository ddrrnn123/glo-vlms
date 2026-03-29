#!/usr/bin/env python3
"""
Compute baseline run-level alignment and similarity summaries from NPZ inference outputs.

Input NPZ layout:
  {npz_root}/{model}_basemodel/{dataset}/runNN_shotNN/
    - images.npz
    - texts.npz
    - metadata.csv

Summary CSV output:
  {output_root}/summary/{dataset}_baseline_summary.csv
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_NPZ_ROOT = "/Data3/Daniel/fewshot/result_0207/baseline_npz"
DEFAULT_OUTPUT_ROOT = "/Data3/Daniel/fewshot/result_0207/inference_results/alignment_similarity"
KNOWN_DATASETS = ["cornell", "vandy"]
KNOWN_MODELS = ["clip", "plip", "conch", "ssl", "resnet18"]


def find_npz_dirs(npz_root: str) -> List[str]:
    pattern = os.path.join(npz_root, "**", "run*_shot*")
    candidates = glob.glob(pattern, recursive=True)
    valid = []
    for directory in candidates:
        if os.path.exists(os.path.join(directory, "images.npz")) and os.path.exists(os.path.join(directory, "texts.npz")):
            valid.append(directory)
    return sorted(valid)


def parse_npz_dir(npz_dir: str, npz_root: str) -> Tuple[str, str, str, int, int]:
    rel = os.path.relpath(npz_dir, npz_root)
    parts = rel.split(os.sep)
    if len(parts) < 3:
        raise ValueError(f"NPZ path too short: {npz_dir}")

    model_method = parts[-3]
    dataset = parts[-2]
    run_shot = parts[-1]

    match = re.search(r"run(\d+)_shot(\d+)", run_shot)
    if not match:
        raise ValueError(f"Cannot parse run/shot: {run_shot}")

    run_id = int(match.group(1))
    shot = int(match.group(2))

    mm_parts = model_method.split("_")
    model = mm_parts[0]
    method = "_".join(mm_parts[1:]) if len(mm_parts) > 1 else ""
    if not model or not method:
        raise ValueError(f"Cannot parse model/method: {model_method}")

    return model, method, dataset, run_id, shot


def load_npz_features(npz_dir: str) -> Dict[str, np.ndarray]:
    images = np.load(os.path.join(npz_dir, "images.npz"), allow_pickle=True)
    texts = np.load(os.path.join(npz_dir, "texts.npz"), allow_pickle=True)
    return {
        "image_features": images["X"],
        "image_labels": images["y"],
        "text_features": texts["X"],
        "text_labels": texts["y"],
        "class_names": texts["class_names"],
    }


def load_metadata(npz_dir: str) -> Dict[str, object]:
    path = os.path.join(npz_dir, "metadata.csv")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        if df.empty:
            return {}
        return df.iloc[0].to_dict()
    except Exception:
        return {}


def compute_similarity_gap(similarity: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    pos = []
    neg = []
    for idx in range(len(labels)):
        class_idx = int(labels[idx])
        sims = similarity[idx]
        pos.append(float(sims[class_idx]))
        neg.append(float(np.mean([sims[j] for j in range(len(sims)) if j != class_idx])))

    mean_pos = float(np.mean(pos)) if pos else 0.0
    mean_neg = float(np.mean(neg)) if neg else 0.0
    return mean_pos - mean_neg, mean_pos, mean_neg


def parse_fc_bias(metadata: Dict[str, object], num_classes: int) -> np.ndarray:
    raw = metadata.get("ssl_fc_bias")
    if raw is None:
        raw = metadata.get("fc_bias")
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.zeros((num_classes,), dtype=np.float32)

    values: Optional[List[float]] = None
    if isinstance(raw, str):
        txt = raw.strip()
        if not txt:
            values = []
        else:
            try:
                parsed = ast.literal_eval(txt)
                if isinstance(parsed, list):
                    values = [float(x) for x in parsed]
            except Exception:
                values = None
    elif isinstance(raw, list):
        values = [float(x) for x in raw]

    if values is None or len(values) != num_classes:
        return np.zeros((num_classes,), dtype=np.float32)

    return np.array(values, dtype=np.float32)


def process_npz_dir(npz_dir: str, npz_root: str) -> Optional[Dict[str, object]]:
    model, method, dataset, run_id, shot = parse_npz_dir(npz_dir, npz_root)
    if method != "basemodel":
        return None

    features = load_npz_features(npz_dir)
    metadata = load_metadata(npz_dir)

    class_names = [str(c) for c in features["class_names"]]
    image_labels = features["image_labels"].astype(int)
    image_features = features["image_features"]
    text_features = features["text_features"]

    if model in {"ssl", "resnet18"}:
        _ = parse_fc_bias(metadata, len(class_names))
        similarity_gap = 0.0
        mean_pos = 0.0
        mean_neg = 0.0
        alignment_score = 0.0
        logit_scale_used = 0.0
    else:
        similarity = image_features.dot(text_features.T)
        similarity_gap, mean_pos, mean_neg = compute_similarity_gap(similarity, image_labels)
        alignment_score = mean_pos
        logit_scale_used = 1.0

    return {
        "dataset": dataset,
        "model": model,
        "method": "baseline",
        "run_id": run_id,
        "shot": shot,
        "num_images": int(len(image_labels)),
        "num_classes": int(len(class_names)),
        "alignment_score": float(alignment_score),
        "similarity_gap": float(similarity_gap),
        "mean_cosine_pos": float(mean_pos),
        "mean_cosine_neg": float(mean_neg),
        "logit_scale_used": float(logit_scale_used),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute baseline alignment and similarity summaries from NPZ inference outputs.")
    parser.add_argument("--npz_root", default=DEFAULT_NPZ_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", default="all", choices=["cornell", "vandy", "all"])
    parser.add_argument("--model", default="all", choices=["clip", "plip", "conch", "ssl", "resnet18", "all"])
    args = parser.parse_args()

    datasets = KNOWN_DATASETS if args.dataset == "all" else [args.dataset]
    models = KNOWN_MODELS if args.model == "all" else [args.model]

    npz_dirs = find_npz_dirs(args.npz_root)
    summary_by_dataset: Dict[str, List[Dict[str, object]]] = {dataset: [] for dataset in datasets}

    for npz_dir in npz_dirs:
        try:
            model, method, dataset, _, _ = parse_npz_dir(npz_dir, args.npz_root)
        except Exception:
            continue

        if dataset not in datasets:
            continue
        if model not in models:
            continue
        if method != "basemodel":
            continue

        result = process_npz_dir(npz_dir, args.npz_root)
        if result is not None:
            summary_by_dataset[dataset].append(result)

    summary_dir = os.path.join(args.output_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    for dataset, rows in summary_by_dataset.items():
        if not rows:
            continue
        df = pd.DataFrame(rows).sort_values(["model", "run_id", "shot"]).reset_index(drop=True)
        out_path = os.path.join(summary_dir, f"{dataset}_baseline_summary.csv")
        df.to_csv(out_path, index=False)
        print(f"[OK] {dataset}: {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
