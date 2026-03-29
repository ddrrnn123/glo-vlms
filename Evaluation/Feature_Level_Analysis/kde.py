#!/usr/bin/env python3
"""
Run KDE-based image-text feature alignment analysis from NPZ feature files.
"""

import json
import os

import adapters
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import umap
from PIL import Image
from sklearn.metrics import silhouette_score
from transformers import CLIPModel, CLIPProcessor

from kdeplot_utils import plot_kde2d


MODEL_METHOD = "all"
DATASETS = ["cornell", "vandy"]
RUN_IDS = [9]
SHOT_CONFIGS = [1, 2, 4, 8, 16, 32]
NPZ_ROOT = "/Data3/Daniel/fewshot/result_0207"

CLASS_PROMPTS = {
    "Atubular Glomerulus": "A pathology image of atubular glomerulus",
    "Globally Sclerotic Glomerulus": "A pathology image of globally sclerotic glomerulus",
    "Ischemic Glomerulus": "A pathology image of ischemic glomerulus",
    "Segmentally Sclerotic Glomerulus": "A pathology image of segmentally sclerotic glomerulus",
    "Viable Glomerulus": "A pathology image of viable glomerulus",
}

CLASS_NAMES = ["Atubular", "GSG", "Ischemic", "SSG", "Viable"]
TRAIN_CLASS_TO_IDX = {
    "Atubular Glomerulus": 0,
    "Globally Sclerotic Glomerulus": 1,
    "Ischemic Glomerulus": 2,
    "Segmentally Sclerotic Glomerulus": 3,
    "Viable Glomerulus": 4,
}

COLORS = ["red", "blue", "green", "orange", "purple"]
KDE_CMAPS = ["Reds", "Blues", "Greens", "Oranges", "Purples"]


def resolve_datasets():
    if isinstance(DATASETS, str):
        if DATASETS.lower() == "all":
            return ["cornell", "vandy"]
        return [DATASETS]
    return [str(dataset) for dataset in DATASETS]


def resolve_run_ids():
    if isinstance(RUN_IDS, str):
        if RUN_IDS.lower() == "all":
            return list(range(1, 11))
        return [int(RUN_IDS)]
    if isinstance(RUN_IDS, int):
        return [RUN_IDS]
    if isinstance(RUN_IDS, (list, tuple, set)):
        return sorted({int(value) for value in RUN_IDS})
    raise ValueError(f"Unsupported RUN_IDS configuration: {RUN_IDS}")


def discover_model_methods(npz_root, datasets):
    discovered = []
    skip_dirs = {"baseline_npz", "inference_results", "ssl"}

    if not os.path.exists(npz_root):
        return discovered

    for name in sorted(os.listdir(npz_root)):
        full = os.path.join(npz_root, name)
        if not os.path.isdir(full):
            continue
        if name in skip_dirs or "_" not in name:
            continue

        valid = False
        for dataset in datasets:
            dataset_dir = os.path.join(full, dataset)
            if not os.path.isdir(dataset_dir):
                continue
            for entry in os.listdir(dataset_dir):
                entry_path = os.path.join(dataset_dir, entry)
                if entry.startswith("run") and "_shot" in entry and os.path.isdir(entry_path):
                    valid = True
                    break
            if valid:
                break

        if valid:
            discovered.append(name)

    return discovered


def load_npz_features(model_method, dataset, run_id, shot, npz_root):
    npz_dir = os.path.join(npz_root, model_method, dataset, f"run{run_id:02d}_shot{shot:02d}")
    if not os.path.exists(npz_dir):
        raise FileNotFoundError(f"Missing NPZ directory: {npz_dir}")

    images_npz_path = os.path.join(npz_dir, "images.npz")
    texts_npz_path = os.path.join(npz_dir, "texts.npz")
    if not os.path.exists(images_npz_path):
        raise FileNotFoundError(f"Missing images.npz: {images_npz_path}")
    if not os.path.exists(texts_npz_path):
        raise FileNotFoundError(f"Missing texts.npz: {texts_npz_path}")

    image_data = np.load(images_npz_path)
    text_data = np.load(texts_npz_path)
    image_features = image_data["X"]
    image_labels = image_data["y"]
    text_features = text_data["X"]
    text_labels = text_data["y"]
    return image_features, image_labels, text_features, text_labels


class AdapterCLIPExtractor:
    """Feature extractor for adapter-tuned CLIP models."""

    def __init__(self, model_name, adapter_path, device="cuda:1"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model = CLIPModel.from_pretrained(model_name)
        adapters.init(self.model)

        with open(os.path.join(adapter_path, "adapter_config.json"), "r") as handle:
            adapter_name = json.load(handle).get("name", "bottleneck_adapter")

        self.model.load_adapter(adapter_path, load_as=adapter_name)
        self.model = self.model.to(device).eval()
        self.model.active_adapters = adapter_name

        def preprocess_image(image):
            return self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        self.preprocess = preprocess_image

    def encode_image(self, images):
        with torch.no_grad():
            return F.normalize(self.model.get_image_features(pixel_values=images), dim=-1)

    def encode_text(self, text_prompts):
        with torch.no_grad():
            text_inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True)
            text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
            return F.normalize(self.model.get_text_features(**text_inputs), dim=-1)


def get_base_model_name(model_type):
    return {"clip": "openai/clip-vit-base-patch16", "plip": "vinid/plip"}[model_type]


def load_csv_data(csv_path):
    df = pd.read_csv(csv_path)
    data_samples = []
    for _, row in df.iterrows():
        if row["class_name"] in TRAIN_CLASS_TO_IDX:
            data_samples.append((row["path"], TRAIN_CLASS_TO_IDX[row["class_name"]]))
    return data_samples


def extract_image_features_from_csv(model, csv_path, device):
    data_samples = load_csv_data(csv_path)
    all_features, all_labels = [], []

    for start in range(0, len(data_samples), 256):
        batch = data_samples[start : start + 256]
        batch_images, batch_labels = [], []

        for img_path, label_idx in batch:
            try:
                image = Image.open(img_path).convert("RGB")
                batch_images.append(model.preprocess(image).to(device))
                batch_labels.append(label_idx)
            except Exception:
                continue

        if batch_images:
            batch_features = model.encode_image(torch.stack(batch_images))
            all_features.append(batch_features.cpu().numpy())
            all_labels.extend(batch_labels)

    if all_features:
        return np.vstack(all_features), np.array(all_labels)
    return np.array([]), np.array([])


def extract_text_features(model):
    text_prompts = list(CLASS_PROMPTS.values())
    text_labels = [TRAIN_CLASS_TO_IDX[class_name] for class_name in CLASS_PROMPTS.keys()]
    text_features = model.encode_text(text_prompts)
    return text_features.cpu().numpy(), np.array(text_labels)


def plot_kde_text_alignment(image_features, image_labels, text_features, text_labels, output_path, title_suffix=""):
    umap_reducer = umap.UMAP(
        n_neighbors=100,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    image_umap = umap_reducer.fit_transform(image_features)
    text_umap = umap_reducer.transform(text_features)

    fig, ax = plt.subplots(figsize=(20, 15))
    plt.rcParams["font.size"] = 30
    plt.rcParams["axes.labelsize"] = 30
    plt.rcParams["xtick.labelsize"] = 30
    plt.rcParams["ytick.labelsize"] = 30

    image_df = pd.DataFrame(
        {
            "x": image_umap[:, 0],
            "y": image_umap[:, 1],
            "class": [CLASS_NAMES[index] for index in image_labels],
        }
    )

    for class_idx in range(len(CLASS_NAMES)):
        class_image_data = image_df[image_df["class"] == CLASS_NAMES[class_idx]]
        if len(class_image_data) >= 2:
            plot_kde2d(
                data=class_image_data,
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

    for class_idx in range(len(CLASS_NAMES)):
        mask = text_labels == class_idx
        if np.any(mask):
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
            ax.annotate(
                CLASS_NAMES[class_idx],
                (text_umap[mask, 0][0], text_umap[mask, 1][0]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=50,
                fontweight="bold",
                color=COLORS[class_idx],
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=COLORS[class_idx],
                    alpha=0.8,
                ),
                zorder=11,
            )

    ax.set_xlabel("Dimension 1", fontsize=30)
    ax.set_ylabel("Dimension 2", fontsize=30)
    ax.tick_params(axis="both", which="major", labelsize=30)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return image_umap, text_umap


def calculate_alignment_metrics(image_umap, image_labels, text_umap, text_labels):
    alignment_scores = []
    for class_idx in range(len(CLASS_NAMES)):
        image_mask = image_labels == class_idx
        text_mask = text_labels == class_idx
        if np.any(image_mask) and np.any(text_mask):
            image_centroid = np.mean(image_umap[image_mask], axis=0)
            distances = [np.linalg.norm(point - image_centroid) for point in text_umap[text_mask]]
            alignment_scores.append(np.mean(distances))

    overall_alignment = np.mean(alignment_scores)
    silhouette = silhouette_score(image_umap, image_labels, metric="euclidean")
    return {
        "class_distances": alignment_scores,
        "overall_alignment": overall_alignment,
        "silhouette_score": silhouette,
        "class_names": CLASS_NAMES,
    }


def analyze_single_shot_run(run_id, shot_configs, test_csv_path, output_dir, device="cuda:1"):
    all_results = {}

    for shot in shot_configs:
        try:
            adapter_path = f"/Data3/Daniel/fewshot/model_all_results/weights/clip_adapter_run{run_id}_shot{shot}_weights"
            if not os.path.exists(adapter_path):
                continue

            model = AdapterCLIPExtractor(get_base_model_name("clip"), adapter_path, device)
            image_features, image_labels = extract_image_features_from_csv(model, test_csv_path, device)
            if len(image_features) == 0:
                continue

            text_features, text_labels = extract_text_features(model)
            output_path = f"{output_dir}/kde_analysis/clip_adapter_run{run_id}_shot{shot}_kde_alignment.pdf"
            image_umap, text_umap = plot_kde_text_alignment(
                image_features,
                image_labels,
                text_features,
                text_labels,
                output_path,
                f" (Run {run_id}, {shot}-shot)",
            )
            all_results[shot] = calculate_alignment_metrics(image_umap, image_labels, text_umap, text_labels)
        except Exception:
            continue

    return all_results


def generate_alignment_summary(run_id, all_results, output_dir):
    if not all_results:
        return

    summary_data = []
    for shot in sorted(all_results.keys()):
        metrics = all_results[shot]
        row = {
            "Shot": shot,
            "Overall_Alignment": metrics["overall_alignment"],
            "Silhouette_Score": metrics["silhouette_score"],
        }
        for index, class_name in enumerate(metrics["class_names"]):
            if index < len(metrics["class_distances"]):
                row[f"{class_name}_Distance"] = metrics["class_distances"][index]
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "kde_analysis", f"run{run_id}_kde_alignment_summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    return summary_df


def analyze_single_shot_npz(model_method, dataset, run_id, shot, npz_root, output_dir):
    try:
        image_features, image_labels, text_features, text_labels = load_npz_features(
            model_method, dataset, run_id, shot, npz_root
        )

        output_path = os.path.join(
            output_dir,
            "kde_analysis",
            dataset,
            model_method,
            f"run{run_id:02d}",
            f"shot{shot:02d}_kde_alignment.pdf",
        )

        image_umap, text_umap = plot_kde_text_alignment(
            image_features,
            image_labels,
            text_features,
            text_labels,
            output_path,
            f" ({model_method.upper()} Run {run_id}, {shot}-shot)",
        )
        alignment_metrics = calculate_alignment_metrics(image_umap, image_labels, text_umap, text_labels)
        return shot, alignment_metrics
    except Exception:
        return shot, None


def analyze_all_shots():
    run_ids = resolve_run_ids()
    shot_configs = SHOT_CONFIGS
    npz_root = NPZ_ROOT
    datasets = resolve_datasets()

    if str(MODEL_METHOD).lower() == "all":
        model_methods = discover_model_methods(npz_root, datasets)
    else:
        model_methods = [MODEL_METHOD]

    if not model_methods:
        return

    output_dir = "/Data3/Daniel/fewshot/result_0207/inference_results"
    os.makedirs(os.path.join(output_dir, "kde_analysis"), exist_ok=True)

    for dataset in datasets:
        for model_method in model_methods:
            for run_id in run_ids:
                results = {}
                for shot in shot_configs:
                    shot_result = analyze_single_shot_npz(model_method, dataset, run_id, shot, npz_root, output_dir)
                    if shot_result[1] is not None:
                        results[shot_result[0]] = shot_result[1]

                if results:
                    summary_data = []
                    for shot in sorted(results.keys()):
                        metrics = results[shot]
                        row = {
                            "Shot": shot,
                            "Overall_Alignment": metrics["overall_alignment"],
                            "Silhouette_Score": metrics["silhouette_score"],
                        }
                        for index, class_name in enumerate(metrics["class_names"]):
                            if index < len(metrics["class_distances"]):
                                row[f"{class_name}_Distance"] = metrics["class_distances"][index]
                        summary_data.append(row)

                    summary_df = pd.DataFrame(summary_data)
                    summary_path = os.path.join(
                        output_dir,
                        "kde_analysis",
                        dataset,
                        model_method,
                        f"run{run_id:02d}",
                        "all_shots_summary.csv",
                    )
                    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
                    summary_df.to_csv(summary_path, index=False)


def main():
    analyze_all_shots()


if __name__ == "__main__":
    main()
