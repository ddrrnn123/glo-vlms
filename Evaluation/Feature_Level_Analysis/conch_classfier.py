#!/usr/bin/env python3
"""
Generate KDE visualizations for CONCH classifier hidden embeddings.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.metrics import silhouette_score

from kdeplot_utils import plot_kde2d


CLASS_NAMES = ["Atubular", "GSG", "Ischemic", "SSG", "Viable"]
KDE_CMAPS = ["Reds", "Blues", "Greens", "Oranges", "Purples"]

NPZ_ROOT = "/Data3/Daniel/fewshot/result_0207"
OUTPUT_ROOT = "/Data3/Daniel/fewshot/result_0207/inference_results/classifier_kde"
DATASETS = ["cornell", "vandy"]


def load_umap_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "embeddings" in data and "labels" in data:
        embeddings = data["embeddings"]
        labels = data["labels"]
    elif "X" in data and "y" in data:
        embeddings = data["X"]
        labels = data["y"]
    else:
        raise KeyError(f"Unsupported NPZ keys: {list(data.keys())}")
    return embeddings, labels


def umap_transform(embeddings):
    reducer = umap.UMAP(
        n_neighbors=100,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def plot_classifier_kde(embeddings_2d, labels, output_path):
    fig, ax = plt.subplots(figsize=(20, 15))
    plt.rcParams["font.size"] = 30
    plt.rcParams["axes.labelsize"] = 30
    plt.rcParams["xtick.labelsize"] = 30
    plt.rcParams["ytick.labelsize"] = 30

    df = pd.DataFrame(
        {
            "Dimension 1": embeddings_2d[:, 0],
            "Dimension 2": embeddings_2d[:, 1],
            "class": [CLASS_NAMES[index] for index in labels],
        }
    )

    for class_idx in range(len(CLASS_NAMES)):
        class_data = df[df["class"] == CLASS_NAMES[class_idx]]
        if len(class_data) >= 2:
            plot_kde2d(
                data=class_data,
                x="Dimension 1",
                y="Dimension 2",
                ax=ax,
                fill=True,
                levels=6,
                thresh=0.4,
                bw_adjust=0.6,
                cmap=KDE_CMAPS[class_idx],
                alpha=0.6,
                legend=False,
                common_norm=False,
                common_grid=True,
            )

    ax.set_xlabel("Dimension 1", fontsize=30)
    ax.set_ylabel("Dimension 2", fontsize=30)
    ax.tick_params(axis="both", which="major", labelsize=30)
    ax.xaxis.get_label().set_text("Dimension 1")
    ax.yaxis.get_label().set_text("Dimension 2")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_npz_path(npz_path):
    parent_dir = os.path.basename(os.path.dirname(npz_path))
    dataset = os.path.basename(os.path.dirname(os.path.dirname(npz_path)))
    parts = parent_dir.split("_")
    run_id = int(parts[0].replace("run", ""))
    shot = int(parts[1].replace("shot", ""))
    return dataset, run_id, shot


def resolve_datasets():
    if isinstance(DATASETS, str):
        if DATASETS.lower() == "all":
            return ["cornell", "vandy"]
        return [DATASETS]
    return [str(dataset) for dataset in DATASETS]


def find_all_conch_class_npz():
    npz_paths = []
    for dataset in resolve_datasets():
        pattern_new = os.path.join(NPZ_ROOT, "conch_classifier", dataset, "run*_shot*", "images.npz")
        pattern_old = os.path.join(NPZ_ROOT, "conch_classifier", dataset, "run*_shot*", "umap_embeddings.npz")
        npz_paths.extend(glob.glob(pattern_new))
        npz_paths.extend(glob.glob(pattern_old))
    return sorted(set(npz_paths))


def process_single_npz(npz_path):
    try:
        dataset, run_id, shot = parse_npz_path(npz_path)
        output_dir = os.path.join(OUTPUT_ROOT, "conch_classifier", dataset, f"run{run_id:02d}")
        output_path = os.path.join(output_dir, f"shot{shot:02d}_kde.png")

        embeddings, labels = load_umap_embeddings(npz_path)
        embeddings_2d = umap_transform(embeddings)
        silhouette = float(silhouette_score(embeddings_2d, labels, metric="euclidean"))

        if not os.path.exists(output_path):
            plot_classifier_kde(embeddings_2d, labels, output_path)

        return dataset, run_id, shot, silhouette
    except Exception:
        return None


def main():
    npz_paths = find_all_conch_class_npz()
    if not npz_paths:
        return

    silhouette_results = []
    for npz_path in npz_paths:
        result = process_single_npz(npz_path)
        if result is not None:
            silhouette_results.append(
                {
                    "Dataset": result[0],
                    "Run": result[1],
                    "Shot": result[2],
                    "Silhouette_Score": result[3],
                }
            )

    if silhouette_results:
        results_by_run = {}
        for result in silhouette_results:
            key = (result["Dataset"], result["Run"])
            if key not in results_by_run:
                results_by_run[key] = []
            results_by_run[key].append(result)

        for (dataset, run_id), run_results in sorted(results_by_run.items()):
            csv_path = os.path.join(
                OUTPUT_ROOT,
                "conch_classifier",
                dataset,
                f"run{run_id:02d}",
                "silhouette_summary.csv",
            )
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            run_df = pd.DataFrame(
                [{"Shot": row["Shot"], "Silhouette_Score": row["Silhouette_Score"]} for row in run_results]
            )
            run_df.sort_values("Shot").to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
