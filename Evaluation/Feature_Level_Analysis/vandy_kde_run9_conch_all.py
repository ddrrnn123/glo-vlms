#!/usr/bin/env python3
"""
Vandy Run9 KDE script (CONCH methods + baseline).

- text-anchor methods: conch_vanilla / conch_lora / conch_adapter
- classifier method: conch_classifier (no text anchors)
- baseline: conch_basemodel (with text anchors)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import silhouette_score

from kdeplot_utils import plot_kde2d


# ========= Fixed config =========
DATASET = "vandy"
RUN_ID = 9
SHOTS = [1, 2, 4, 8, 16, 32]

NPZ_ROOT = "/Data3/Daniel/fewshot/result_0207"
BASELINE_ROOT = "/Data3/Daniel/fewshot/result_0207/baseline_npz"
OUTPUT_ROOT = "/Data3/Daniel/fewshot/result_0207/inference_results/vandy_kde_run9"

TEXT_ANCHOR_METHODS = ["conch_vanilla", "conch_lora", "conch_adapter"]
CLASSIFIER_METHOD = "conch_classifier"

TARGET_CLASS_NAMES = [
    "Disappearing",
    "Non-glomerular",
    "Normal",
    "Obsolescent",
    "Solidified",
]
TARGET_CLASS_TO_IDX = {name: idx for idx, name in enumerate(TARGET_CLASS_NAMES)}

COLORS = ["red", "blue", "green", "orange", "purple"]
KDE_CMAPS = ["Reds", "Blues", "Greens", "Oranges", "Purples"]


def normalize_to_short_name(class_name):
    key = str(class_name).strip().lower()
    mapping = {
        "disappearing glomeruli": "Disappearing",
        "non-glomerular": "Non-glomerular",
        "normal glomeruli": "Normal",
        "obsolescent glomeruli": "Obsolescent",
        "solidified glomeruli": "Solidified",
        "disappearing": "Disappearing",
        "normal": "Normal",
        "obsolescent": "Obsolescent",
        "solidified": "Solidified",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported Vandy class name: {class_name}")
    return mapping[key]


def build_old_to_new_index(text_class_names):
    old_to_new = {}
    short_names = []
    for old_idx, class_name in enumerate(text_class_names):
        short_name = normalize_to_short_name(class_name)
        short_names.append(short_name)
        if short_name not in TARGET_CLASS_TO_IDX:
            raise ValueError(f"Missing target class: {short_name}")
        old_to_new[old_idx] = TARGET_CLASS_TO_IDX[short_name]

    seen = set(short_names)
    target = set(TARGET_CLASS_NAMES)
    if seen != target:
        missing = sorted(target - seen)
        extra = sorted(seen - target)
        raise ValueError(
            f"Unexpected Vandy class set. missing={missing}, extra={extra}, source={list(text_class_names)}"
        )

    if len(set(short_names)) != len(short_names):
        raise ValueError(f"Duplicate class names after mapping: {short_names}")

    return old_to_new


def remap_labels(labels, old_to_new, name):
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {labels.shape}")

    labels_int = labels.astype(np.int64)
    if not np.array_equal(labels_int, labels):
        raise ValueError(f"{name} contains non-integer labels.")

    unique_vals = np.unique(labels_int)
    for val in unique_vals:
        if int(val) not in old_to_new:
            raise ValueError(f"{name} contains unknown label {val}, old_to_new={old_to_new}")

    remapped = np.array([old_to_new[int(v)] for v in labels_int], dtype=np.int64)
    return remapped


def get_umap_reducer():
    return umap.UMAP(
        n_neighbors=100,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )


def load_npz_pair(npz_dir):
    images_npz = os.path.join(npz_dir, "images.npz")
    texts_npz = os.path.join(npz_dir, "texts.npz")

    if not os.path.exists(images_npz):
        raise FileNotFoundError(f"Missing images.npz: {images_npz}")
    if not os.path.exists(texts_npz):
        raise FileNotFoundError(f"Missing texts.npz: {texts_npz}")

    image_data = np.load(images_npz, allow_pickle=True)
    text_data = np.load(texts_npz, allow_pickle=True)

    for key in ("X", "y"):
        if key not in image_data:
            raise KeyError(f"images.npz missing key {key}: {images_npz}")
        if key not in text_data:
            raise KeyError(f"texts.npz missing key {key}: {texts_npz}")
    if "class_names" not in text_data:
        raise KeyError(f"texts.npz missing key class_names: {texts_npz}")

    return (
        image_data["X"],
        image_data["y"],
        text_data["X"],
        text_data["y"],
        text_data["class_names"],
    )


def _annotation_overflow_score(annotation, ax, renderer):
    text_bbox = annotation.get_window_extent(renderer=renderer)
    ax_bbox = ax.get_window_extent(renderer=renderer)

    overflow_left = max(0.0, ax_bbox.x0 - text_bbox.x0)
    overflow_right = max(0.0, text_bbox.x1 - ax_bbox.x1)
    overflow_bottom = max(0.0, ax_bbox.y0 - text_bbox.y0)
    overflow_top = max(0.0, text_bbox.y1 - ax_bbox.y1)

    return overflow_left + overflow_right + overflow_bottom + overflow_top


def _auto_annotate_inside_axes(ax, xy, label, color):
    # Try multiple directions and keep the first placement without overflow.
    candidates = [
        (10, 10),
        (-10, 10),
        (10, -10),
        (-10, -10),
        (16, 0),
        (-16, 0),
        (0, 16),
        (0, -16),
    ]

    best = None
    fig = ax.figure

    for dx, dy in candidates:
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"

        ann = ax.annotate(
            label,
            xy,
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=50,
            fontweight="bold",
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=color,
                alpha=0.8,
            ),
            zorder=11,
        )

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        score = _annotation_overflow_score(ann, ax, renderer)
        ann.remove()

        if best is None or score < best[0]:
            best = (score, dx, dy, ha, va)
        if score == 0:
            break

    _, best_dx, best_dy, best_ha, best_va = best
    ax.annotate(
        label,
        xy,
        xytext=(best_dx, best_dy),
        textcoords="offset points",
        ha=best_ha,
        va=best_va,
        fontsize=50,
        fontweight="bold",
        color=color,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=color,
            alpha=0.8,
        ),
        zorder=11,
    )


def plot_kde_with_text_anchors(image_umap, image_labels, text_umap, text_labels, output_path):
    fig, ax = plt.subplots(figsize=(20, 15))

    plt.rcParams["font.size"] = 30
    plt.rcParams["axes.labelsize"] = 30
    plt.rcParams["xtick.labelsize"] = 30
    plt.rcParams["ytick.labelsize"] = 30

    image_df = pd.DataFrame(
        {
            "x": image_umap[:, 0],
            "y": image_umap[:, 1],
            "class": [TARGET_CLASS_NAMES[i] for i in image_labels],
        }
    )

    for class_idx in range(len(TARGET_CLASS_NAMES)):
        class_data = image_df[image_df["class"] == TARGET_CLASS_NAMES[class_idx]]
        if len(class_data) < 2:
            continue

        plot_kde2d(
            data=class_data,
            x="x",
            y="y",
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

    for class_idx in range(len(TARGET_CLASS_NAMES)):
        mask = text_labels == class_idx
        if not np.any(mask):
            continue

        ax.scatter(
            text_umap[mask, 0],
            text_umap[mask, 1],
            c="white",
            marker="o",
            s=3000,
            alpha=0.9,
            edgecolors=COLORS[class_idx],
            linewidth=3,
            zorder=10,
        )

        anchor_xy = (float(text_umap[mask, 0][0]), float(text_umap[mask, 1][0]))
        _auto_annotate_inside_axes(
            ax=ax,
            xy=anchor_xy,
            label=TARGET_CLASS_NAMES[class_idx],
            color=COLORS[class_idx],
        )

    ax.set_xlabel("Dimension 1", fontsize=30)
    ax.set_ylabel("Dimension 2", fontsize=30)
    ax.tick_params(axis="both", which="major", labelsize=30)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_classifier_kde_no_text(image_umap, image_labels, output_path):
    fig, ax = plt.subplots(figsize=(20, 15))

    plt.rcParams["font.size"] = 30
    plt.rcParams["axes.labelsize"] = 30
    plt.rcParams["xtick.labelsize"] = 30
    plt.rcParams["ytick.labelsize"] = 30

    df = pd.DataFrame(
        {
            "Dimension 1": image_umap[:, 0],
            "Dimension 2": image_umap[:, 1],
            "class": [TARGET_CLASS_NAMES[i] for i in image_labels],
        }
    )

    for class_idx in range(len(TARGET_CLASS_NAMES)):
        class_data = df[df["class"] == TARGET_CLASS_NAMES[class_idx]]
        if len(class_data) < 2:
            continue

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


def calculate_alignment_metrics(image_umap, image_labels, text_umap, text_labels):
    alignment_scores = []
    for class_idx in range(len(TARGET_CLASS_NAMES)):
        image_mask = image_labels == class_idx
        text_mask = text_labels == class_idx
        if np.any(image_mask) and np.any(text_mask):
            image_centroid = np.mean(image_umap[image_mask], axis=0)
            distances = [np.linalg.norm(tp - image_centroid) for tp in text_umap[text_mask]]
            alignment_scores.append(np.mean(distances))

    overall_alignment = float(np.mean(alignment_scores)) if alignment_scores else float("nan")
    silhouette = float(silhouette_score(image_umap, image_labels, metric="euclidean"))

    return {
        "class_distances": alignment_scores,
        "overall_alignment": overall_alignment,
        "silhouette_score": silhouette,
        "class_names": TARGET_CLASS_NAMES,
    }


def load_and_remap(npz_dir):
    image_features, image_labels, text_features, text_labels, text_class_names = load_npz_pair(npz_dir)
    old_to_new = build_old_to_new_index(text_class_names)
    image_labels = remap_labels(image_labels, old_to_new, "image_labels")
    text_labels = remap_labels(text_labels, old_to_new, "text_labels")
    return image_features, image_labels, text_features, text_labels, text_class_names


def run_text_anchor_methods():
    for method in TEXT_ANCHOR_METHODS:
        rows = []
        for shot in SHOTS:
            npz_dir = os.path.join(NPZ_ROOT, method, DATASET, f"run{RUN_ID:02d}_shot{shot:02d}")

            image_features, image_labels, text_features, text_labels, text_class_names = load_and_remap(npz_dir)

            umap_reducer = get_umap_reducer()
            image_umap = umap_reducer.fit_transform(image_features)
            text_umap = umap_reducer.transform(text_features)

            output_path = os.path.join(
                OUTPUT_ROOT, method, f"run{RUN_ID:02d}", f"shot{shot:02d}_kde_alignment.png"
            )
            plot_kde_with_text_anchors(image_umap, image_labels, text_umap, text_labels, output_path)

            metrics = calculate_alignment_metrics(image_umap, image_labels, text_umap, text_labels)

            row = {
                "Shot": shot,
                "Overall_Alignment": metrics["overall_alignment"],
                "Silhouette_Score": metrics["silhouette_score"],
            }
            for i, class_name in enumerate(metrics["class_names"]):
                if i < len(metrics["class_distances"]):
                    row[f"{class_name}_Distance"] = metrics["class_distances"][i]
            rows.append(row)

        summary_path = os.path.join(OUTPUT_ROOT, method, f"run{RUN_ID:02d}", "all_shots_summary.csv")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        pd.DataFrame(rows).sort_values("Shot").to_csv(summary_path, index=False)


def run_classifier_method():
    rows = []
    for shot in SHOTS:
        npz_dir = os.path.join(NPZ_ROOT, CLASSIFIER_METHOD, DATASET, f"run{RUN_ID:02d}_shot{shot:02d}")

        image_features, image_labels, _, _, text_class_names = load_and_remap(npz_dir)

        umap_reducer = get_umap_reducer()
        image_umap = umap_reducer.fit_transform(image_features)
        silhouette = float(silhouette_score(image_umap, image_labels, metric="euclidean"))

        output_path = os.path.join(
            OUTPUT_ROOT, CLASSIFIER_METHOD, f"run{RUN_ID:02d}", f"shot{shot:02d}_kde.png"
        )
        if os.path.exists(output_path):
            pass
        else:
            plot_classifier_kde_no_text(image_umap, image_labels, output_path)

        rows.append({"Shot": shot, "Silhouette_Score": silhouette})

    summary_path = os.path.join(OUTPUT_ROOT, CLASSIFIER_METHOD, f"run{RUN_ID:02d}", "silhouette_summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    pd.DataFrame(rows).sort_values("Shot").to_csv(summary_path, index=False)


def run_conch_baseline():
    npz_dir = os.path.join(BASELINE_ROOT, "conch_basemodel", DATASET, "run00_shot00")

    image_features, image_labels, text_features, text_labels, text_class_names = load_and_remap(npz_dir)

    umap_reducer = get_umap_reducer()
    image_umap = umap_reducer.fit_transform(image_features)
    text_umap = umap_reducer.transform(text_features)

    output_path = os.path.join(OUTPUT_ROOT, "conch_baseline", "run00", "shot00_kde_alignment.png")
    plot_kde_with_text_anchors(image_umap, image_labels, text_umap, text_labels, output_path)

    metrics = calculate_alignment_metrics(image_umap, image_labels, text_umap, text_labels)
    row = {
        "Shot": 0,
        "Overall_Alignment": metrics["overall_alignment"],
        "Silhouette_Score": metrics["silhouette_score"],
    }
    for i, class_name in enumerate(metrics["class_names"]):
        if i < len(metrics["class_distances"]):
            row[f"{class_name}_Distance"] = metrics["class_distances"][i]

    summary_path = os.path.join(OUTPUT_ROOT, "conch_baseline", "run00", "all_shots_summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    pd.DataFrame([row]).to_csv(summary_path, index=False)


def main():
    run_text_anchor_methods()
    run_classifier_method()
    run_conch_baseline()


if __name__ == "__main__":
    main()
