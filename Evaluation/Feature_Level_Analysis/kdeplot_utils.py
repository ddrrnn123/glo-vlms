import matplotlib.pyplot as plt
import seaborn as sns


def plot_kde2d(
    data,
    x,
    y,
    hue=None,
    ax=None,
    fill=True,
    levels=10,
    thresh=0.05,
    bw_adjust=1.0,
    cmap=None,
    palette=None,
    common_norm=False,
    common_grid=True,
    linewidths=1.0,
    alpha=0.8,
    legend=True,
    title=None,
    savepath=None,
    **kde_kwargs,
):
    """Plot a 2D KDE contour map with optional filled regions."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    sns.kdeplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        fill=fill,
        levels=levels,
        thresh=thresh,
        bw_adjust=bw_adjust,
        cmap=cmap,
        palette=palette,
        common_norm=common_norm,
        common_grid=common_grid,
        linewidths=linewidths,
        alpha=alpha,
        ax=ax,
        **kde_kwargs,
    )

    if title:
        ax.set_title(title)

    if not legend:
        legend_obj = ax.get_legend()
        if legend_obj is not None:
            legend_obj.remove()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    return fig, ax
