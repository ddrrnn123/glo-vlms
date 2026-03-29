#!/usr/bin/env python3
"""
Compute run-level alignment and similarity summaries from NPZ inference outputs.

Input NPZ layout:
  {npz_root}/{model}_{method}/{dataset}/runNN_shotNN/
    - images.npz
    - texts.npz
    - metadata.csv

Summary CSV output:
  {output_root}/summary/{dataset}_summary.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_NPZ_ROOT = "/Data3/Daniel/fewshot/result_0207"
DEFAULT_OUTPUT_ROOT = "/Data3/Daniel/fewshot/result_0207/inference_results/alignment_similarity"


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

    if "_shot" not in run_shot:
        raise ValueError(f"Bad run/shot segment: {run_shot}")

    match = re.search(r"run(\d+)_shot(\d+)", run_shot)
    if not match:
        raise ValueError(f"Cannot parse run/shot: {run_shot}")

    run_id = int(match.group(1))
    shot = int(match.group(2))

    mm_parts = model_method.split("_")
    model = mm_parts[0]
    method = "_".join(mm_parts[1:]) if len(mm_parts) > 1 else ""
    if not method:
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


def load_metadata(npz_dir: str) -> Dict[str, Optional[float]]:
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


def is_classifier_method(method: str, text_features: np.ndarray) -> bool:
    if "classifier" in method:
        return True
    return np.allclose(text_features, 0)


def get_logit_scale(metadata: Dict[str, Optional[float]]) -> float:
    value = metadata.get("logit_scale_exp")
    try:
        if value is not None and not pd.isna(value):
            return float(value)
    except Exception:
        pass
    return 100.0


def is_resnet_logits_npz(
    model: str,
    method: str,
    image_features: np.ndarray,
    text_features: np.ndarray,
    metadata: Dict[str, Optional[float]],
) -> bool:
    if model != "resnet18":
        return False

    feature_type = str(metadata.get("feature_type", "")).lower()
    if "logits" in feature_type:
        return True

    if image_features.ndim == 2 and text_features.ndim == 2:
        n_cls = text_features.shape[0]
        if image_features.shape[1] == n_cls:
            eye = np.eye(n_cls, dtype=text_features.dtype)
            if text_features.shape == eye.shape and np.allclose(text_features, eye):
                return True

    return method in {"linear_probe", "resnet18_linear_probe"}


def compute_similarity_gap(similarity: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    pos = []
    neg = []
    for idx in range(len(labels)):
        class_idx = int(labels[idx])
        sims = similarity[idx]
        pos.append(sims[class_idx])
        neg.append(np.mean([sims[j] for j in range(len(sims)) if j != class_idx]))

    mean_pos = float(np.mean(pos)) if pos else 0.0
    mean_neg = float(np.mean(neg)) if neg else 0.0
    return mean_pos - mean_neg, mean_pos, mean_neg


def process_npz_dir(npz_dir: str, npz_root: str) -> Optional[Dict[str, object]]:
    model, method, dataset, run_id, shot = parse_npz_dir(npz_dir, npz_root)
    features = load_npz_features(npz_dir)
    metadata = load_metadata(npz_dir)

    class_names = [str(c) for c in features["class_names"]]
    image_labels = features["image_labels"].astype(int)
    image_features = features["image_features"]
    text_features = features["text_features"]

    is_classifier = is_classifier_method(method, text_features)
    is_resnet_logits = is_resnet_logits_npz(model, method, image_features, text_features, metadata)

    if is_resnet_logits or is_classifier:
        similarity_gap = 0.0
        mean_pos = 0.0
        mean_neg = 0.0
        alignment_score = 0.0
        logit_scale_used = 1.0 if is_resnet_logits else 0.0
    else:
        similarity = image_features.dot(text_features.T)
        similarity_gap, mean_pos, mean_neg = compute_similarity_gap(similarity, image_labels)
        alignment_score = mean_pos
        logit_scale_used = get_logit_scale(metadata)

    return {
        "dataset": dataset,
        "model": model,
        "method": method,
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


def parse_ssl_csv(path: str) -> Optional[Dict[str, object]]:
    match = re.search(r"run(\d+)_shot(\d+)", os.path.basename(path))
    if not match:
        return None

    run_id = int(match.group(1))
    shot = int(match.group(2))

    df = pd.read_csv(path)
    if df.empty:
        return None

    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    class_names = [col[len("prob_"):] for col in prob_cols]

    return {
        "run_id": run_id,
        "shot": shot,
        "num_images": int(len(df)),
        "num_classes": int(len(class_names)),
    }


def process_ssl(npz_root: str, dataset: str) -> List[Dict[str, object]]:
    results = []
    ssl_dir = os.path.join(npz_root, "ssl", dataset)
    if not os.path.exists(ssl_dir):
        return results

    for path in sorted(glob.glob(os.path.join(ssl_dir, "predictions_*_run*_shot*.csv"))):
        parsed = parse_ssl_csv(path)
        if parsed is None:
            continue
        results.append(
            {
                "dataset": dataset,
                "model": "resnet18",
                "method": "ssl",
                "run_id": parsed["run_id"],
                "shot": parsed["shot"],
                "num_images": parsed["num_images"],
                "num_classes": parsed["num_classes"],
                "alignment_score": 0.0,
                "similarity_gap": 0.0,
                "mean_cosine_pos": 0.0,
                "mean_cosine_neg": 0.0,
                "logit_scale_used": 0.0,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute alignment and similarity summaries from NPZ inference outputs.")
    parser.add_argument("--npz_root", default=DEFAULT_NPZ_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", default="all", choices=["cornell", "vandy", "all"])
    parser.add_argument("--model", default="all", choices=["clip", "plip", "conch", "resnet18", "all"])
    parser.add_argument(
        "--method",
        default="all",
        choices=["lora", "adapter", "vanilla", "classifier", "linear_probe", "all"],
    )
    parser.add_argument("--include_ssl", action="store_true", default=True)
    args = parser.parse_args()

    datasets = ["cornell", "vandy"] if args.dataset == "all" else [args.dataset]
    models = ["clip", "plip", "conch", "resnet18"] if args.model == "all" else [args.model]
    methods = ["lora", "adapter", "vanilla", "classifier", "linear_probe"] if args.method == "all" else [args.method]

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
        if method not in methods:
            continue

        result = process_npz_dir(npz_dir, args.npz_root)
        if result is not None:
            summary_by_dataset[dataset].append(result)

    if args.include_ssl:
        for dataset in datasets:
            summary_by_dataset[dataset].extend(process_ssl(args.npz_root, dataset))

    summary_dir = os.path.join(args.output_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    for dataset, rows in summary_by_dataset.items():
        if not rows:
            continue
        df = pd.DataFrame(rows).sort_values(["model", "method", "run_id", "shot"]).reset_index(drop=True)
        out_path = os.path.join(summary_dir, f"{dataset}_summary.csv")
        df.to_csv(out_path, index=False)
        print(f"[OK] {dataset}: {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
