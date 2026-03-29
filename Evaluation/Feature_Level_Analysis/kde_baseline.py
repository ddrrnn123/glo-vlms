#!/usr/bin/env python3
"""
Run baseline image-text KDE alignment analysis from baseline NPZ features.
"""

import os

import numpy as np
import pandas as pd

from kde import calculate_alignment_metrics, plot_kde_text_alignment


MODELS = ["clip", "conch", "plip"]
DATASETS = ["cornell", "vandy"]
BASELINE_ROOT = "/Data3/Daniel/fewshot/result_0207/baseline_npz"
OUTPUT_ROOT = "/Data3/Daniel/fewshot/result_0207/inference_results"


def resolve_datasets():
    if isinstance(DATASETS, str):
        if DATASETS.lower() == "all":
            return ["cornell", "vandy"]
        return [DATASETS]
    return [str(dataset) for dataset in DATASETS]


def resolve_models():
    if isinstance(MODELS, str):
        if MODELS.lower() == "all":
            discovered = []
            if os.path.isdir(BASELINE_ROOT):
                for name in sorted(os.listdir(BASELINE_ROOT)):
                    full_path = os.path.join(BASELINE_ROOT, name)
                    if name.endswith("_basemodel") and os.path.isdir(full_path):
                        discovered.append(name.replace("_basemodel", ""))
            return discovered
        return [MODELS]
    return [str(model) for model in MODELS]


def load_baseline_npz_features(model_name, dataset):
    npz_dir = os.path.join(BASELINE_ROOT, f"{model_name}_basemodel", dataset, "run00_shot00")
    if not os.path.isdir(npz_dir):
        raise FileNotFoundError(f"Missing baseline NPZ directory: {npz_dir}")

    images_npz_path = os.path.join(npz_dir, "images.npz")
    texts_npz_path = os.path.join(npz_dir, "texts.npz")
    if not os.path.exists(images_npz_path):
        raise FileNotFoundError(f"Missing images.npz: {images_npz_path}")
    if not os.path.exists(texts_npz_path):
        raise FileNotFoundError(f"Missing texts.npz: {texts_npz_path}")

    image_data = np.load(images_npz_path)
    text_data = np.load(texts_npz_path)
    return image_data["X"], image_data["y"], text_data["X"], text_data["y"]


def analyze_single_baseline(model_name, dataset):
    image_features, image_labels, text_features, text_labels = load_baseline_npz_features(model_name, dataset)

    output_dir = os.path.join(OUTPUT_ROOT, "kde_analysis", dataset, f"{model_name}_baseline", "run00")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "shot00_kde_alignment.pdf")

    image_umap, text_umap = plot_kde_text_alignment(
        image_features,
        image_labels,
        text_features,
        text_labels,
        output_path,
        f" ({model_name.upper()} BASELINE Run 0, 0-shot)",
    )
    metrics = calculate_alignment_metrics(image_umap, image_labels, text_umap, text_labels)

    summary_row = {
        "Shot": 0,
        "Overall_Alignment": metrics["overall_alignment"],
        "Silhouette_Score": metrics["silhouette_score"],
    }
    for index, class_name in enumerate(metrics["class_names"]):
        if index < len(metrics["class_distances"]):
            summary_row[f"{class_name}_Distance"] = metrics["class_distances"][index]

    summary_path = os.path.join(output_dir, "all_shots_summary.csv")
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    return summary_row


def main():
    datasets = resolve_datasets()
    models = resolve_models()
    if not models:
        return

    all_rows = []
    for dataset in datasets:
        for model_name in models:
            try:
                row = analyze_single_baseline(model_name, dataset)
            except Exception:
                continue
            row["Dataset"] = dataset
            row["Model"] = model_name
            all_rows.append(row)

    if all_rows:
        combined_path = os.path.join(OUTPUT_ROOT, "kde_analysis", "baseline_all_summary.csv")
        combined_df = pd.DataFrame(all_rows)
        cols = ["Dataset", "Model", "Shot", "Overall_Alignment", "Silhouette_Score"]
        other_cols = [col for col in combined_df.columns if col not in cols]
        combined_df = combined_df[cols + other_cols]
        combined_df.to_csv(combined_path, index=False)


if __name__ == "__main__":
    main()
