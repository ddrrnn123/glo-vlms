#!/usr/bin/env python3
"""
Compute per-sample predictions and classification summaries from baseline NPZ inference outputs.

Per-sample CSV output:
  {output_root}/per_sample/{dataset}/baseline/{model}/runNN_shotNN.csv

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

try:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import label_binarize
except Exception:  # noqa: BLE001
    accuracy_score = None
    f1_score = None
    roc_auc_score = None
    label_binarize = None


DEFAULT_NPZ_ROOT = "/Data3/Daniel/fewshot/result_0207/baseline_npz"
DEFAULT_OUTPUT_ROOT = "/Data3/Daniel/fewshot/result_0207/inference_results"
KNOWN_DATASETS = ["cornell", "vandy"]
KNOWN_MODELS = ["clip", "plip", "conch", "ssl", "resnet18"]


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    expv = np.exp(logits)
    return expv / np.sum(expv, axis=1, keepdims=True)


def find_npz_dirs(npz_root: str) -> List[str]:
    pattern = os.path.join(npz_root, "**", "run*_shot*")
    candidates = glob.glob(pattern, recursive=True)
    valid = []
    for d in candidates:
        if os.path.exists(os.path.join(d, "images.npz")) and os.path.exists(os.path.join(d, "texts.npz")):
            valid.append(d)
    return sorted(valid)


def parse_npz_dir(npz_dir: str, npz_root: str) -> Tuple[str, str, str, int, int]:
    rel = os.path.relpath(npz_dir, npz_root)
    parts = rel.split(os.sep)
    if len(parts) < 3:
        raise ValueError(f"NPZ path too short: {npz_dir}")

    model_method = parts[-3]
    dataset = parts[-2]
    run_shot = parts[-1]

    m = re.search(r"run(\d+)_shot(\d+)", run_shot)
    if not m:
        raise ValueError(f"Cannot parse run/shot: {run_shot}")
    run_id = int(m.group(1))
    shot = int(m.group(2))

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
        "image_paths": images["paths"],
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
    for i in range(len(labels)):
        idx = int(labels[i])
        sims = similarity[i]
        pos.append(float(sims[idx]))
        neg.append(float(np.mean([sims[j] for j in range(len(sims)) if j != idx])))
    mean_pos = float(np.mean(pos)) if pos else 0.0
    mean_neg = float(np.mean(neg)) if neg else 0.0
    return mean_pos - mean_neg, mean_pos, mean_neg


def compute_metrics(
    true_labels: List[str],
    pred_labels: List[str],
    prob_matrix: Optional[np.ndarray],
    class_names: List[str],
) -> Dict[str, float]:
    if accuracy_score is None:
        return {"ACC": 0.0, "AUC": 0.0, "F1": 0.0}

    acc = accuracy_score(true_labels, pred_labels)
    try:
        f1 = f1_score(true_labels, pred_labels, labels=class_names, average="macro", zero_division=0)
    except Exception:
        f1 = 0.0

    auc = 0.0
    if prob_matrix is not None and label_binarize is not None:
        try:
            y_bin = label_binarize(true_labels, classes=class_names)
            if y_bin.ndim == 1:
                y_bin = y_bin.reshape(-1, 1)
            aucs = []
            for i in range(len(class_names)):
                if len(np.unique(y_bin[:, i])) > 1:
                    aucs.append(roc_auc_score(y_bin[:, i], prob_matrix[:, i]))
            auc = float(np.mean(aucs)) if aucs else 0.0
        except Exception:
            auc = 0.0

    return {"ACC": round(acc, 4), "AUC": round(auc, 4), "F1": round(f1, 4)}


def build_per_sample_df(
    image_paths: np.ndarray,
    true_labels: List[str],
    pred_labels: List[str],
    probs: np.ndarray,
    cosine_sim: np.ndarray,
    class_names: List[str],
) -> pd.DataFrame:
    rows = []
    for i in range(len(image_paths)):
        row = {
            "image_path": image_paths[i],
            "true_label": true_labels[i],
            "pred_label": pred_labels[i],
        }
        for j, cls in enumerate(class_names):
            row[f"prob_{cls}"] = float(probs[i, j])
            row[f"cosine_sim_{cls}"] = float(cosine_sim[i, j])
        rows.append(row)
    return pd.DataFrame(rows)


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

    if values is None:
        return np.zeros((num_classes,), dtype=np.float32)

    if len(values) != num_classes:
        return np.zeros((num_classes,), dtype=np.float32)

    return np.array(values, dtype=np.float32)


def process_npz_dir(npz_dir: str, npz_root: str, output_root: str, overwrite: bool) -> Optional[Dict[str, object]]:
    model, method, dataset, run_id, shot = parse_npz_dir(npz_dir, npz_root)
    if method != "basemodel":
        return None

    features = load_npz_features(npz_dir)
    metadata = load_metadata(npz_dir)

    class_names = [str(c) for c in features["class_names"]]
    image_labels = features["image_labels"].astype(int)
    true_labels = [class_names[idx] for idx in image_labels]

    image_features = features["image_features"]
    text_features = features["text_features"]

    if model in {"ssl", "resnet18"}:
        bias = parse_fc_bias(metadata, len(class_names))
        logits = image_features.dot(text_features.T) + bias.reshape(1, -1)
        probs = softmax_np(logits)
        cosine_sim = np.zeros_like(probs)
        pred_indices = np.argmax(probs, axis=1)
        pred_labels = [class_names[i] for i in pred_indices]
        similarity_gap = 0.0
        mean_pos = 0.0
        mean_neg = 0.0
        alignment_score = 0.0
        scale_used = 0.0
    else:
        similarity = image_features.dot(text_features.T)
        probs = softmax_np(similarity)
        cosine_sim = similarity
        pred_indices = np.argmax(probs, axis=1)
        pred_labels = [class_names[i] for i in pred_indices]
        similarity_gap, mean_pos, mean_neg = compute_similarity_gap(similarity, image_labels)
        alignment_score = mean_pos
        scale_used = 1.0

    per_sample_dir = os.path.join(output_root, "per_sample", dataset, "baseline", model)
    os.makedirs(per_sample_dir, exist_ok=True)
    per_sample_path = os.path.join(per_sample_dir, f"run{run_id:02d}_shot{shot:02d}.csv")
    if overwrite or not os.path.exists(per_sample_path):
        df = build_per_sample_df(features["image_paths"], true_labels, pred_labels, probs, cosine_sim, class_names)
        df.to_csv(per_sample_path, index=False)

    metrics = compute_metrics(true_labels, pred_labels, probs, class_names)
    return {
        "dataset": dataset,
        "model": model,
        "method": "baseline",
        "run_id": run_id,
        "shot": shot,
        "num_images": int(len(true_labels)),
        "num_classes": int(len(class_names)),
        "ACC": metrics["ACC"],
        "AUC": metrics["AUC"],
        "F1": metrics["F1"],
        "alignment_score": float(alignment_score),
        "similarity_gap": float(similarity_gap),
        "mean_cosine_pos": float(mean_pos),
        "mean_cosine_neg": float(mean_neg),
        "logit_scale_used": float(scale_used),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute classification summaries from baseline NPZ inference outputs.")
    parser.add_argument("--npz_root", default=DEFAULT_NPZ_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", default="all", choices=["cornell", "vandy", "all"])
    parser.add_argument("--model", default="all", choices=["clip", "plip", "conch", "ssl", "resnet18", "all"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets = KNOWN_DATASETS if args.dataset == "all" else [args.dataset]
    models = KNOWN_MODELS if args.model == "all" else [args.model]

    npz_dirs = find_npz_dirs(args.npz_root)
    summary_by_dataset: Dict[str, List[Dict[str, object]]] = {d: [] for d in datasets}

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

        result = process_npz_dir(npz_dir, args.npz_root, args.output_root, args.overwrite)
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


if __name__ == "__main__":
    main()
