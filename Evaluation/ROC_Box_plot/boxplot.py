#!/usr/bin/env python3
"""
Generate class-wise AUC boxplots from per-sample prediction CSV files.
"""

import os
import re
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score


PER_SAMPLE_ROOT = "/Data3/Daniel/fewshot/result_0207/inference_results/per_sample"
OUTPUT_DIR = "/Data3/Daniel/fewshot/result_0207/inference_results/boxplot"

DATASETS = ["cornell", "vandy"]
ALLOWED_MODELS = {"clip", "conch", "plip"}

DATASET_CLASS_NAMES = {
    "cornell": [
        "Atubular Glomeruli",
        "Global Glomerulosclerosis",
        "Ischemic Glomeruli",
        "Segmental Glomerulosclerosis",
        "Viable Glomeruli",
    ],
    "vandy": [
        "Normal glomeruli",
        "Obsolescent glomeruli",
        "Solidified glomeruli",
        "Disappearing glomeruli",
        "Non-glomerular",
    ],
}

METHOD_STYLE_MAPPING = {
    "Adapter": {"color": "#d62728", "linestyle": "-"},
    "Vanilla": {"color": "#2ca02c", "linestyle": "-"},
    "LoRA": {"color": "#ff7f0e", "linestyle": "-"},
    "Classifier": {"color": "#8209f4", "linestyle": "-"},
    "Zero-shot": {"color": "#7A6E6E", "linestyle": "-"},
}

METHOD_DISPLAY_MAP = {
    "adapter": "Adapter",
    "vanilla": "Vanilla",
    "lora": "LoRA",
    "classifier": "Classifier",
    "class": "Classifier",
    "class_model": "Classifier",
    "itc": "Vanilla",
    "basemodel": "Zero-shot",
    "zeroshot": "Zero-shot",
    "zero-shot": "Zero-shot",
}


def save_horizontal_legend(methods_to_plot: list, method_colors: dict, output_dir: str) -> None:
    if not methods_to_plot:
        return

    os.makedirs(output_dir, exist_ok=True)
    fig_w = max(6.0, 1.8 * len(methods_to_plot))
    fig, ax = plt.subplots(figsize=(fig_w, 1.4))
    ax.axis("off")

    handles = [
        Patch(
            facecolor=method_colors[method],
            edgecolor=method_colors[method],
            alpha=0.5,
            label=method,
        )
        for method in methods_to_plot
    ]

    ax.legend(
        handles=handles,
        loc="center",
        ncol=len(methods_to_plot),
        frameon=True,
        fontsize=20,
        handlelength=1.2,
        columnspacing=1.2,
        borderpad=0.4,
    )

    fig.savefig(os.path.join(output_dir, "legend_horizontal.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def calculate_class_auc(df: pd.DataFrame, target_class: str) -> float:
    if df.empty:
        return np.nan

    try:
        y_true_binary = (df["true_label"] == target_class).astype(int)
        prob_col = f"prob_{target_class}"
        if prob_col not in df.columns:
            return np.nan
        y_prob = df[prob_col].values
        if len(np.unique(y_true_binary)) < 2:
            return np.nan
        return float(roc_auc_score(y_true_binary, y_prob))
    except Exception:
        return np.nan


def calculate_class_accuracy(df: pd.DataFrame, target_class: str) -> float:
    if df.empty:
        return np.nan

    try:
        class_df = df[df["true_label"] == target_class]
        if class_df.empty:
            return np.nan
        correct = (class_df["pred_label"] == target_class).sum()
        total = len(class_df)
        return correct / total if total > 0 else np.nan
    except Exception:
        return np.nan


def parse_run_shot_filename(filename: str) -> Optional[dict]:
    match = re.match(r"^run(\d+)_shot(\d+)\.csv$", filename)
    if not match:
        return None
    return {"run": int(match.group(1)), "shot": int(match.group(2))}


def _read_csv_safe(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    return df


def _resolve_datasets():
    if isinstance(DATASETS, str):
        if DATASETS.lower() == "all":
            return ["cornell", "vandy"]
        return [DATASETS]
    return [str(dataset) for dataset in DATASETS]


def process_finetuned_data(per_sample_root: str, dataset: str, class_names: list) -> pd.DataFrame:
    dataset_root = os.path.join(per_sample_root, dataset)
    results = []

    if not os.path.isdir(dataset_root):
        return pd.DataFrame()

    for method_raw in sorted(os.listdir(dataset_root)):
        method_dir = os.path.join(dataset_root, method_raw)
        if not os.path.isdir(method_dir):
            continue
        if method_raw.lower() == "baseline":
            continue

        method_disp = METHOD_DISPLAY_MAP.get(method_raw.lower(), method_raw.title())

        for model in sorted(os.listdir(method_dir)):
            model_dir = os.path.join(method_dir, model)
            if not os.path.isdir(model_dir):
                continue
            if model.lower() not in ALLOWED_MODELS:
                continue

            for fname in sorted(os.listdir(model_dir)):
                if not fname.endswith(".csv"):
                    continue

                parsed = parse_run_shot_filename(fname)
                if parsed is None:
                    continue

                df = _read_csv_safe(os.path.join(model_dir, fname))
                if df.empty:
                    continue

                for class_name in class_names:
                    class_auc = calculate_class_auc(df, class_name)
                    if np.isnan(class_auc):
                        continue
                    results.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "method": method_disp,
                            "method_raw": method_raw,
                            "run": parsed["run"],
                            "shots": parsed["shot"],
                            "class_name": class_name,
                            "auc": class_auc,
                        }
                    )

    return pd.DataFrame(results)


def process_baseline_data(per_sample_root: str, dataset: str, class_names: list) -> pd.DataFrame:
    baseline_root = os.path.join(per_sample_root, dataset, "baseline")
    results = []

    if not os.path.isdir(baseline_root):
        return pd.DataFrame()

    for model in sorted(os.listdir(baseline_root)):
        model_dir = os.path.join(baseline_root, model)
        if not os.path.isdir(model_dir):
            continue
        if model.lower() not in ALLOWED_MODELS:
            continue

        for fname in sorted(os.listdir(model_dir)):
            if not fname.endswith(".csv"):
                continue

            parsed = parse_run_shot_filename(fname)
            if parsed is None:
                continue

            df = _read_csv_safe(os.path.join(model_dir, fname))
            if df.empty:
                continue

            method_disp = METHOD_DISPLAY_MAP.get("basemodel", "Zero-shot")
            for class_name in class_names:
                class_auc = calculate_class_auc(df, class_name)
                if np.isnan(class_auc):
                    continue
                results.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "method": method_disp,
                        "method_raw": "basemodel",
                        "run": parsed["run"],
                        "shots": parsed["shot"],
                        "class_name": class_name,
                        "auc": class_auc,
                    }
                )

    return pd.DataFrame(results)


def plot_boxplots_by_model(df: pd.DataFrame, class_names: list, output_dir: str) -> None:
    if df.empty:
        return

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    unique_models = sorted(df["model"].unique())
    unique_shots = sorted(df["shots"].unique())
    methods_to_plot = [method for method in df["method"].unique() if method in METHOD_STYLE_MAPPING]
    method_colors = {method: METHOD_STYLE_MAPPING[method]["color"] for method in methods_to_plot}
    n_classes = len(class_names)

    for model_name in unique_models:
        model_df = df[df["model"] == model_name].copy()
        if model_df.empty:
            continue

        fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 6), sharey=True)
        if n_classes == 1:
            axes = [axes]

        for i, class_name in enumerate(class_names):
            ax = axes[i]
            class_df = model_df[model_df["class_name"] == class_name].copy()
            if class_df.empty:
                ax.set_axis_off()
                continue

            sns.boxplot(
                x="shots",
                y="auc",
                hue="method",
                data=class_df,
                ax=ax,
                order=unique_shots,
                hue_order=methods_to_plot,
                palette=method_colors,
                showfliers=False,
                linewidth=0.8,
            )

            for patch in ax.artists:
                fc = patch.get_facecolor()
                patch.set_facecolor((fc[0], fc[1], fc[2], 0.4))
            for patch in ax.patches:
                fc = patch.get_facecolor()
                patch.set_facecolor((fc[0], fc[1], fc[2], 0.5))

            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Shots", fontsize=26)
            ax.tick_params(axis="x", labelsize=26)
            ax.tick_params(axis="y", labelsize=26)

            if i == 0:
                ax.set_ylabel("AUC", fontsize=26)
            else:
                ax.set_ylabel("")

            if ax.get_legend():
                ax.get_legend().remove()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f"boxplot_{model_name}_by_class.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    save_horizontal_legend(methods_to_plot, method_colors, output_dir)


def main() -> None:
    datasets = _resolve_datasets()

    for dataset in datasets:
        class_names = DATASET_CLASS_NAMES.get(dataset)
        if class_names is None:
            continue

        finetuned_df = process_finetuned_data(PER_SAMPLE_ROOT, dataset, class_names)
        baseline_df = process_baseline_data(PER_SAMPLE_ROOT, dataset, class_names)

        if not finetuned_df.empty and not baseline_df.empty:
            combined_df = pd.concat([finetuned_df, baseline_df], ignore_index=True)
        elif not finetuned_df.empty:
            combined_df = finetuned_df
        elif not baseline_df.empty:
            combined_df = baseline_df
        else:
            continue

        plot_boxplots_by_model(combined_df, class_names, os.path.join(OUTPUT_DIR, dataset))


if __name__ == "__main__":
    main()
