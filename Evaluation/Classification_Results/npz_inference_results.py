#!/usr/bin/env python3
"""
Compute per-sample predictions and classification summaries from NPZ inference outputs.

Per-sample CSV output:
  {output_root}/per_sample/{dataset}/{method}/{model}/runNN_shotNN.csv

Summary CSV output:
  {output_root}/summary/{dataset}_summary.csv
"""

import argparse
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


DEFAULT_NPZ_ROOT = "/Data3/Daniel/fewshot/result_0207"
DEFAULT_OUTPUT_ROOT = "/Data3/Daniel/fewshot/result_0207/inference_results"


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

    if "_shot" not in run_shot:
        raise ValueError(f"Bad run/shot segment: {run_shot}")

    m = re.search(r"run(\d+)_shot(\d+)", run_shot)
    if not m:
        raise ValueError(f"Cannot parse run/shot: {run_shot}")

    run_id = int(m.group(1))
    shot = int(m.group(2))

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
        "image_paths": images["paths"],
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
        row = df.iloc[0].to_dict()
        return row
    except Exception:
        return {}


def is_classifier_method(method: str, text_features: np.ndarray) -> bool:
    if "classifier" in method:
        return True
    return np.allclose(text_features, 0)


def get_logit_scale(metadata: Dict[str, Optional[float]]) -> float:
    val = metadata.get("logit_scale_exp")
    try:
        if val is not None and not pd.isna(val):
            return float(val)
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
    """Detect ResNet NPZ layout where images.npz stores logits directly."""
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
    for i in range(len(labels)):
        idx = int(labels[i])
        sims = similarity[i]
        pos.append(sims[idx])
        neg.append(np.mean([sims[j] for j in range(len(sims)) if j != idx]))
    mean_pos = float(np.mean(pos)) if pos else 0.0
    mean_neg = float(np.mean(neg)) if neg else 0.0
    return mean_pos - mean_neg, mean_pos, mean_neg


def compute_metrics(
    true_labels: List[str],
    pred_labels: List[str],
    prob_matrix: np.ndarray,
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


def load_classifier_path(
    dataset: str,
    model: str,
    run_id: int,
    shot: int,
    metadata: Dict[str, Optional[float]],
) -> Optional[str]:
    weight_path = metadata.get("weight_path")
    if isinstance(weight_path, str) and weight_path:
        candidate = os.path.join(weight_path, "classifier.pth")
        if os.path.exists(candidate):
            return candidate

    root = f"/Data3/Daniel/fewshot/model_weight_0122_{dataset}"
    pattern = os.path.join(root, f"{model}_classifier", "run*_*", "shot*_*", "best_model", "classifier.pth")
    for p in glob.glob(pattern):
        parts = p.split(os.sep)
        run_dir = parts[-4]
        shot_dir = parts[-3]
        m_run = re.search(r"run(\d+)", run_dir)
        m_shot = re.search(r"shot(\d+)", shot_dir)
        if m_run and m_shot:
            if int(m_run.group(1)) == run_id and int(m_shot.group(1)) == shot:
                return p
    return None


def classifier_probs_from_hidden(hidden: np.ndarray, classifier_path: str) -> np.ndarray:
    import torch

    state = torch.load(classifier_path, map_location="cpu")
    weight = state.get("classifier.4.weight")
    bias = state.get("classifier.4.bias")
    if weight is None or bias is None:
        raise ValueError(f"Missing classifier.4.* in {classifier_path}")

    w = weight.numpy()
    b = bias.numpy()
    logits = hidden.dot(w.T) + b
    return softmax_np(logits)


def process_npz_dir(npz_dir: str, npz_root: str, output_root: str, overwrite: bool) -> Optional[Dict[str, object]]:
    model, method, dataset, run_id, shot = parse_npz_dir(npz_dir, npz_root)
    features = load_npz_features(npz_dir)
    metadata = load_metadata(npz_dir)

    class_names = [str(c) for c in features["class_names"]]
    image_labels = features["image_labels"].astype(int)
    true_labels = [class_names[idx] for idx in image_labels]

    image_features = features["image_features"]
    text_features = features["text_features"]
    is_classifier = is_classifier_method(method, text_features)
    is_resnet_logits = is_resnet_logits_npz(model, method, image_features, text_features, metadata)

    if is_resnet_logits:
        logits = image_features
        probs = softmax_np(logits)
        cosine_sim = np.zeros_like(probs)
        pred_indices = np.argmax(probs, axis=1)
        pred_labels = [class_names[i] for i in pred_indices]
        similarity_gap = 0.0
        mean_pos = 0.0
        mean_neg = 0.0
        alignment_score = 0.0
        logit_scale_used = 1.0
    elif is_classifier:
        classifier_path = load_classifier_path(dataset, model, run_id, shot, metadata)
        if classifier_path is None:
            raise FileNotFoundError(f"Classifier weights not found for {model} {dataset} run{run_id} shot{shot}")
        probs = classifier_probs_from_hidden(image_features, classifier_path)
        cosine_sim = np.zeros_like(probs)
        pred_indices = np.argmax(probs, axis=1)
        pred_labels = [class_names[i] for i in pred_indices]
        similarity_gap = 0.0
        mean_pos = 0.0
        mean_neg = 0.0
        alignment_score = 0.0
        logit_scale_used = 0.0
    else:
        similarity = image_features.dot(text_features.T)
        logit_scale_used = get_logit_scale(metadata)
        probs = softmax_np(similarity * logit_scale_used)
        cosine_sim = similarity
        pred_indices = np.argmax(probs, axis=1)
        pred_labels = [class_names[i] for i in pred_indices]
        similarity_gap, mean_pos, mean_neg = compute_similarity_gap(similarity, image_labels)
        alignment_score = mean_pos

    per_sample_dir = os.path.join(output_root, "per_sample", dataset, method, model)
    os.makedirs(per_sample_dir, exist_ok=True)
    per_sample_path = os.path.join(per_sample_dir, f"run{run_id:02d}_shot{shot:02d}.csv")
    if overwrite or not os.path.exists(per_sample_path):
        df = build_per_sample_df(features["image_paths"], true_labels, pred_labels, probs, cosine_sim, class_names)
        df.to_csv(per_sample_path, index=False)

    metrics = compute_metrics(true_labels, pred_labels, probs, class_names)
    return {
        "dataset": dataset,
        "model": model,
        "method": method,
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
        "logit_scale_used": float(logit_scale_used),
    }


def parse_ssl_csv(path: str) -> Optional[Dict[str, object]]:
    m = re.search(r"run(\d+)_shot(\d+)", os.path.basename(path))
    if not m:
        return None
    run_id = int(m.group(1))
    shot = int(m.group(2))

    df = pd.read_csv(path)
    if df.empty:
        return None

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    class_names = [c[len("prob_"):] for c in prob_cols]
    true_labels = df["true_label"].astype(str).tolist()
    pred_labels = df["pred_label"].astype(str).tolist()
    probs = df[prob_cols].to_numpy() if prob_cols else None

    metrics = (
        compute_metrics(true_labels, pred_labels, probs, class_names)
        if probs is not None
        else {"ACC": 0.0, "AUC": 0.0, "F1": 0.0}
    )

    return {
        "run_id": run_id,
        "shot": shot,
        "num_images": int(len(df)),
        "num_classes": int(len(class_names)),
        "ACC": metrics["ACC"],
        "AUC": metrics["AUC"],
        "F1": metrics["F1"],
    }


def process_ssl(dataset: str, output_root: str) -> List[Dict[str, object]]:
    results = []
    ssl_dir = os.path.join(DEFAULT_NPZ_ROOT, "ssl", dataset)
    if not os.path.exists(ssl_dir):
        return results

    for p in sorted(glob.glob(os.path.join(ssl_dir, "predictions_*_run*_shot*.csv"))):
        parsed = parse_ssl_csv(p)
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
                "ACC": parsed["ACC"],
                "AUC": parsed["AUC"],
                "F1": parsed["F1"],
                "alignment_score": 0.0,
                "similarity_gap": 0.0,
                "mean_cosine_pos": 0.0,
                "mean_cosine_neg": 0.0,
                "logit_scale_used": 0.0,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute classification summaries from NPZ inference outputs.")
    parser.add_argument("--npz_root", default=DEFAULT_NPZ_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", default="all", choices=["cornell", "vandy", "all"])
    parser.add_argument("--model", default="all", choices=["clip", "plip", "conch", "resnet18", "all"])
    parser.add_argument(
        "--method",
        default="all",
        choices=["lora", "adapter", "vanilla", "classifier", "linear_probe", "all"],
    )
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--include_ssl", action="store_true", default=True)
    args = parser.parse_args()

    datasets = ["cornell", "vandy"] if args.dataset == "all" else [args.dataset]
    models = ["clip", "plip", "conch", "resnet18"] if args.model == "all" else [args.model]
    methods = ["lora", "adapter", "vanilla", "classifier", "linear_probe"] if args.method == "all" else [args.method]

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
        if method not in methods:
            continue

        result = process_npz_dir(npz_dir, args.npz_root, args.output_root, args.overwrite)
        if result is not None:
            summary_by_dataset[dataset].append(result)

    if args.include_ssl:
        for dataset in datasets:
            summary_by_dataset[dataset].extend(process_ssl(dataset, args.output_root))

    summary_dir = os.path.join(args.output_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    for dataset, rows in summary_by_dataset.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        out_path = os.path.join(summary_dir, f"{dataset}_summary.csv")
        df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
