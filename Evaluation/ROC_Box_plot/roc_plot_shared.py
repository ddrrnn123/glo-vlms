from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


PathLike = Union[str, Path]


@dataclass(frozen=True)
class RocPlotStyle:
    """Shared plotting style used by ROC scripts."""

    figure_size: Tuple[float, float] = (12, 10)
    curve_linewidth: float = 4.0
    band_alpha: float = 0.15
    chance_line_alpha: float = 0.3
    chance_line_width: float = 1.0
    title_fontsize: int = 26
    title_fontweight: str = "bold"
    axis_label_fontsize: int = 22
    legend_fontsize: int = 24
    legend_framealpha: float = 0.95
    grid_alpha: float = 0.3
    grid_linewidth: float = 0.5
    save_dpi: int = 300
    save_bbox_inches: str = "tight"


CROSS_ROC_STYLE = RocPlotStyle()


def create_roc_figure(style: RocPlotStyle = CROSS_ROC_STYLE):
    """Create a figure/axes pair with the shared ROC figure size."""
    return plt.subplots(figsize=style.figure_size)


def plot_roc_curve(
    ax,
    fpr: np.ndarray,
    tpr: np.ndarray,
    color: str,
    linestyle: str,
    style: RocPlotStyle = CROSS_ROC_STYLE,
    label: str = None,
    alpha: float = 1.0,
    linewidth: float = None,
):
    """Plot one ROC line with shared defaults."""
    ax.plot(
        fpr,
        tpr,
        label=label,
        linewidth=style.curve_linewidth if linewidth is None else linewidth,
        color=color,
        linestyle=linestyle,
        alpha=alpha,
    )


def plot_roc_band(
    ax,
    fpr: np.ndarray,
    tpr_mean: np.ndarray,
    tpr_std: np.ndarray,
    color: str,
    style: RocPlotStyle = CROSS_ROC_STYLE,
    alpha: float = None,
):
    """Plot ROC uncertainty band with shared defaults."""
    band_alpha = style.band_alpha if alpha is None else alpha
    ax.fill_between(
        fpr,
        tpr_mean - tpr_std,
        tpr_mean + tpr_std,
        alpha=band_alpha,
        color=color,
    )


def add_chance_line(ax, style: RocPlotStyle = CROSS_ROC_STYLE):
    """Add y=x chance baseline."""
    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        lw=style.chance_line_width,
        alpha=style.chance_line_alpha,
    )


def apply_cross_axes_format(
    ax,
    shot: int,
    style: RocPlotStyle = CROSS_ROC_STYLE,
):
    """Apply the cross-model axis/title styling."""
    ax.set_title(f"Shot {shot}", fontsize=style.title_fontsize, fontweight=style.title_fontweight)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("False Positive Rate", fontsize=style.axis_label_fontsize)
    ax.set_ylabel("True Positive Rate", fontsize=style.axis_label_fontsize)
    ax.grid(True, alpha=style.grid_alpha, linewidth=style.grid_linewidth)
    ax.legend(loc="lower right", fontsize=style.legend_fontsize, framealpha=style.legend_framealpha)


def save_roc_pdf(
    fig,
    output_file: PathLike,
    style: RocPlotStyle = CROSS_ROC_STYLE,
):
    """Save ROC figure as PDF with shared export options."""
    output_path = Path(output_file)
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches=style.save_bbox_inches, dpi=style.save_dpi)
    plt.close(fig)


def save_roc_png(
    fig,
    output_file: PathLike,
    style: RocPlotStyle = CROSS_ROC_STYLE,
):
    """Save ROC figure as PNG with shared export options."""
    output_path = Path(output_file)
    fig.savefig(output_path, dpi=style.save_dpi, bbox_inches=style.save_bbox_inches)
    plt.close(fig)


def save_legend_png(
    legend_handles: Sequence[Line2D],
    output_file: PathLike,
    figsize: Tuple[float, float],
    ncol: int,
    fontsize: int = 9,
):
    """Save a standalone legend PNG using shared export defaults."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.legend(handles=legend_handles, loc="center", ncol=ncol, fontsize=fontsize)
    fig.savefig(str(output_file), dpi=300, bbox_inches="tight")
    plt.close(fig)
