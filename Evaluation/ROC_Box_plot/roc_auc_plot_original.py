#!/usr/bin/env python3
"""
Generate per-model ROC PDF files for each shot using multi-run averages.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from roc_plot_shared import (
    RocPlotStyle,
    add_chance_line,
    create_roc_figure,
    plot_roc_band,
    plot_roc_curve,
    save_legend_png,
    save_roc_pdf,
)


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

DATASETS = ["cornell", "vandy"]
MODEL_NAMES = ["clip", "conch", "plip"]
SHOTS = [1, 2, 4, 8, 16, 32]
METHODS = ["adapter", "lora", "classifier", "vanilla"]

RESULT_ROOT = "/Data3/Daniel/fewshot/result_0207/inference_results"
PER_SAMPLE_ROOT = f"{RESULT_ROOT}/per_sample"
STATISTICS_DIR = f"{RESULT_ROOT}/run_comparison"
OUTPUT_ROOT = f"{RESULT_ROOT}/roc_auc/original"

RAW2DISPLAY = {
    "adapter": "Adapter",
    "lora": "LoRA",
    "classifier": "Classifier",
    "class": "Classifier",
    "itc": "Vanilla",
    "vanilla": "Vanilla",
    "basemodel": "Zero-shot",
    "zeroshot": "Zero-shot",
}

ORIGINAL_ROC_STYLE = RocPlotStyle(
    figure_size=(8, 6),
    curve_linewidth=2.0,
    band_alpha=0.2,
    legend_fontsize=10,
    legend_framealpha=0.9,
)


def parse_run_shot_filename(filename: str) -> Optional[Tuple[int, int]]:
    stem = filename.replace(".csv", "")
    match = re.match(r"^run(\d+)_shot(\d+)$", stem)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _canon_method(method_name: str) -> str:
    text = str(method_name).strip().lower()
    alias = {
        "class_model": "classifier",
        "class": "classifier",
        "classifier_model": "classifier",
        "itc": "vanilla",
        "base": "baseline",
        "basemodel": "baseline",
        "zero": "baseline",
        "zero_shot": "baseline",
        "zero-shot": "baseline",
        "zero shot": "baseline",
        "zeroshot": "baseline",
    }
    return alias.get(text, text)


def _resolve_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    col_map = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        key = str(name).strip().lower()
        if key in col_map:
            return col_map[key]
    return None


def load_statistics_summary(dataset: str) -> Dict[Tuple[str, str, int], float]:
    stats_file = Path(STATISTICS_DIR) / f"{dataset}_statistics_summary.csv"
    if not stats_file.exists():
        return {}

    try:
        df = pd.read_csv(stats_file)
    except Exception:
        return {}

    auc_lookup: Dict[Tuple[str, str, int], float] = {}
    selected_methods = {_canon_method(method) for method in METHODS}

    col_model = _resolve_col(df, ["model"])
    col_method = _resolve_col(df, ["method"])
    col_shot = _resolve_col(df, ["shot", "shots"])
    col_auc = _resolve_col(df, ["AUC_Mean", "AUC_mean", "auc_mean", "auc"])

    required = [col_model, col_method, col_shot, col_auc]
    if any(col is None for col in required):
        return {}

    for _, row in df.iterrows():
        model = str(row[col_model]).strip().lower()
        method = _canon_method(str(row[col_method]).strip())
        shot = int(float(row[col_shot]))
        auc_mean = float(row[col_auc])

        if model in MODEL_NAMES and method in selected_methods.union({"baseline"}):
            auc_lookup[(model, method, shot)] = auc_mean

    return auc_lookup


def validate_csv_for_roc(df: pd.DataFrame, class_names: List[str]) -> bool:
    if df.empty:
        return False
    required_cols = ["true_label"] + [f"prob_{c}" for c in class_names]
    missing = [col for col in required_cols if col not in df.columns]
    return len(missing) == 0


def collect_run_csvs(dataset: str, model: str, method: str, shot: int, class_names: List[str]) -> List[pd.DataFrame]:
    method_dir = Path(PER_SAMPLE_ROOT) / dataset / method / model
    if not method_dir.exists():
        return []

    candidates = list(method_dir.glob(f"run*_shot{shot:02d}.csv"))
    if not candidates:
        candidates = list(method_dir.glob(f"run*_shot{shot}.csv"))

    run_dfs: List[pd.DataFrame] = []
    for csv_path in sorted(candidates):
        parsed = parse_run_shot_filename(csv_path.name)
        if not parsed:
            continue
        _, file_shot = parsed
        if file_shot != shot:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if validate_csv_for_roc(df, class_names):
            run_dfs.append(df)

    return run_dfs


def load_baseline_data(dataset: str, model: str, class_names: List[str]) -> Optional[pd.DataFrame]:
    baseline_dir = Path(PER_SAMPLE_ROOT) / dataset / "baseline" / model
    if not baseline_dir.exists():
        return None

    baseline_dfs = []
    for csv_path in sorted(baseline_dir.glob("run*_shot*.csv")):
        parsed = parse_run_shot_filename(csv_path.name)
        if not parsed:
            continue
        _, file_shot = parsed
        if file_shot != 0:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if validate_csv_for_roc(df, class_names):
            baseline_dfs.append(df)

    if not baseline_dfs:
        return None
    if len(baseline_dfs) == 1:
        return baseline_dfs[0]
    return pd.concat(baseline_dfs, ignore_index=True)


def compute_single_roc(df: pd.DataFrame, class_names: List[str]) -> Tuple[np.ndarray, np.ndarray, float]:
    try:
        true_labels_raw = df["true_label"]
        is_numeric = pd.api.types.is_numeric_dtype(true_labels_raw)
        is_str_numeric = all(str(x).strip().isdigit() for x in true_labels_raw.dropna())

        if is_numeric or is_str_numeric:
            y_true = np.zeros((len(df), len(class_names)))
            for i, (_, row) in enumerate(df.iterrows()):
                idx = int(str(row["true_label"]).strip())
                if 0 <= idx < len(class_names):
                    y_true[i, idx] = 1
            y_score = df[[f"prob_{c}" for c in class_names]].values
        else:
            true_labels = df["true_label"].astype(str).str.strip()
            valid = true_labels.isin(class_names)
            df = df[valid].copy()
            true_labels = true_labels[valid]
            if len(df) == 0:
                return np.linspace(0, 1, 100), np.zeros(100), 0.0
            y_true = label_binarize(true_labels, classes=class_names)
            if y_true.ndim == 1:
                y_true = y_true.reshape(-1, 1)
            y_score = df[[f"prob_{c}" for c in class_names]].values

        all_fpr, all_tpr, aucs = [], [], []
        for i in range(len(class_names)):
            if len(np.unique(y_true[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
                all_fpr.append(fpr)
                all_tpr.append(tpr)
                aucs.append(roc_auc_score(y_true[:, i], y_score[:, i]))

        if all_fpr:
            mean_fpr = np.linspace(0, 1, 100)
            mean_tpr = np.zeros_like(mean_fpr)
            for fpr, tpr in zip(all_fpr, all_tpr):
                mean_tpr += np.interp(mean_fpr, fpr, tpr)
            mean_tpr /= len(all_tpr)
            macro_auc = float(np.mean(aucs))
        else:
            mean_fpr = np.linspace(0, 1, 100)
            mean_tpr = np.zeros(100)
            macro_auc = 0.0

        return mean_fpr, mean_tpr, macro_auc
    except Exception:
        return np.linspace(0, 1, 100), np.zeros(100), 0.0


def compute_multi_run_roc(run_dfs: List[pd.DataFrame], class_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not run_dfs:
        fpr_common = np.linspace(0, 1, 100)
        return fpr_common, np.zeros(100), np.zeros(100)

    all_tpr = []
    fpr_common = np.linspace(0, 1, 100)
    for df in run_dfs:
        fpr, tpr, _ = compute_single_roc(df, class_names)
        all_tpr.append(np.interp(fpr_common, fpr, tpr))

    all_tpr = np.array(all_tpr)
    return fpr_common, np.mean(all_tpr, axis=0), np.std(all_tpr, axis=0)


def get_method_style(method_name: str) -> dict:
    method_lower = str(method_name).lower()
    if "adapter" in method_lower:
        return {"color": "#d62728", "linestyle": "-"}
    if "vanilla" in method_lower:
        return {"color": "#2ca02c", "linestyle": "-"}
    if "lora" in method_lower:
        return {"color": "#ff7f0e", "linestyle": "-"}
    if "classifier" in method_lower:
        return {"color": "#9467bd", "linestyle": "-"}
    if "zeroshot" in method_lower:
        return {"color": "#7A6E6E", "linestyle": "-"}
    return {"color": "#133333", "linestyle": "-"}


def plot_single_shot_pdf(
    dataset: str,
    model: str,
    shot: int,
    class_names: List[str],
    auc_lookup: Dict[Tuple[str, str, int], float],
    output_dir: Path,
) -> None:
    fig, ax = create_roc_figure(ORIGINAL_ROC_STYLE)

    for method in METHODS:
        run_dfs = collect_run_csvs(dataset, model, method, shot, class_names)
        if not run_dfs:
            continue

        fpr, tpr_mean, tpr_std = compute_multi_run_roc(run_dfs, class_names)
        auc_display = auc_lookup.get((model, method, shot), 0.0)
        style = get_method_style(method)
        display_name = RAW2DISPLAY.get(method, method)

        plot_roc_curve(
            ax,
            fpr,
            tpr_mean,
            label=f"{display_name}, AUC={auc_display:.4f}",
            color=style["color"],
            linestyle=style["linestyle"],
            style=ORIGINAL_ROC_STYLE,
        )
        plot_roc_band(ax, fpr, tpr_mean, tpr_std, color=style["color"], style=ORIGINAL_ROC_STYLE)

    baseline_df = load_baseline_data(dataset, model, class_names)
    if baseline_df is not None:
        fpr, tpr, _ = compute_single_roc(baseline_df, class_names)
        auc_display = auc_lookup.get((model, "baseline", 0), 0.0)
        style = get_method_style("zeroshot")
        plot_roc_curve(
            ax,
            fpr,
            tpr,
            label=f"Zero-shot, AUC={auc_display:.4f}",
            color=style["color"],
            linestyle=style["linestyle"],
            style=ORIGINAL_ROC_STYLE,
        )

    add_chance_line(ax, ORIGINAL_ROC_STYLE)
    ax.set_title(f"Shot {shot}", fontsize=26, fontweight="bold")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("False Positive Rate", fontsize=22)
    ax.set_ylabel("True Positive Rate", fontsize=22)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc="lower right", fontsize=ORIGINAL_ROC_STYLE.legend_fontsize, framealpha=ORIGINAL_ROC_STYLE.legend_framealpha)

    output_file = output_dir / f"roc_{model.upper()}_shot{shot}.pdf"
    save_roc_pdf(fig, output_file, ORIGINAL_ROC_STYLE)


def plot_legends(output_dir: Path) -> None:
    method_order = ["zeroshot", "adapter", "vanilla", "lora", "classifier"]
    legend_elements = []
    for method in method_order:
        style = get_method_style(method)
        legend_elements.append(
            Line2D(
                [0],
                [0],
                label=RAW2DISPLAY.get(method, "Zero-shot" if method == "zeroshot" else method),
                color=style["color"],
                linestyle=style["linestyle"],
            )
        )

    save_legend_png(legend_elements, output_dir / "legend_horizontal.png", figsize=(6, 0.8), ncol=len(legend_elements), fontsize=8)
    save_legend_png(legend_elements, output_dir / "legend_vertical.png", figsize=(1.8, 3), ncol=1, fontsize=8)


def main() -> None:
    per_sample_dir = Path(PER_SAMPLE_ROOT)
    statistics_dir = Path(STATISTICS_DIR)
    if not per_sample_dir.exists() or not statistics_dir.exists():
        raise SystemExit(1)

    for dataset in DATASETS:
        dataset_key = dataset.strip().lower()
        class_names = DATASET_CLASS_NAMES.get(dataset_key)
        if not class_names:
            continue

        dataset_per_sample = per_sample_dir / dataset_key
        if not dataset_per_sample.exists():
            continue

        auc_lookup = load_statistics_summary(dataset_key)
        output_dir = Path(OUTPUT_ROOT) / dataset_key
        output_dir.mkdir(parents=True, exist_ok=True)

        for model in MODEL_NAMES:
            for shot in SHOTS:
                try:
                    plot_single_shot_pdf(
                        dataset=dataset_key,
                        model=model,
                        shot=shot,
                        class_names=class_names,
                        auc_lookup=auc_lookup,
                        output_dir=output_dir,
                    )
                except Exception:
                    continue

        try:
            plot_legends(output_dir)
        except Exception:
            continue


if __name__ == "__main__":
    main()
