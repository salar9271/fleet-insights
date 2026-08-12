"""Matplotlib figure builders for the clustering + supervised comparison.

Pure plotting: every function takes already-computed arrays/DataFrames (from
src.clustering / scripts.diagnostics) and saves one PNG at dpi=150. No data
loading or model fitting happens here -- see scripts/make_figures.py for
that orchestration.

Colors follow the project's dataviz palette: three fixed categorical hues
(blue/orange/aqua) for cluster/class identity, a blue<->red diverging scale
with a neutral gray midpoint for signed z-scores, and a single-hue blue
sequential scale for non-negative counts.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

DPI = 150

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"]
)
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "blue_sequential", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def _clean_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_LINE)
    ax.spines["bottom"].set_color(AXIS_LINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_PRIMARY)
    ax.yaxis.label.set_color(INK_PRIMARY)


def _grid_heatmap_axes(ax, n_rows, n_cols):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)


def plot_pca_scatter(X_scaled, cluster_labels, true_labels, out_path):
    """Two-panel PCA scatter of the same train windows: left colored by
    KMeans cluster, right colored by true class."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var_pct = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 5), facecolor=SURFACE, sharex=True, sharey=True
    )

    ax = axes[0]
    for i, cluster in enumerate(sorted(np.unique(cluster_labels))):
        mask = cluster_labels == cluster
        ax.scatter(
            coords[mask, 0], coords[mask, 1], s=22, alpha=0.85,
            color=CATEGORICAL[i % len(CATEGORICAL)], edgecolors="none",
            label=f"cluster {cluster}",
        )
    ax.set_title("Colored by KMeans cluster", color=INK_PRIMARY, fontsize=11)
    ax.set_xlabel(f"PC1 ({var_pct[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_pct[1]:.1f}% var)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    _clean_axes(ax)

    ax = axes[1]
    for i, cls in enumerate(sorted(np.unique(true_labels))):
        mask = true_labels == cls
        ax.scatter(
            coords[mask, 0], coords[mask, 1], s=22, alpha=0.85,
            color=CATEGORICAL[i % len(CATEGORICAL)], edgecolors="none",
            label=str(cls),
        )
    ax.set_title("Colored by true class", color=INK_PRIMARY, fontsize=11)
    ax.set_xlabel(f"PC1 ({var_pct[0]:.1f}% var)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    _clean_axes(ax)

    fig.suptitle(
        "PCA of train windows: unsupervised clusters vs. true labels",
        color=INK_PRIMARY, fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def plot_cluster_heatmap(cluster_profile, out_path):
    """Heatmap of mean z-score per feature (rows) by cluster (columns)."""
    data = cluster_profile.T
    n_rows, n_cols = data.shape

    fig, ax = plt.subplots(
        figsize=(max(6.5, 2.0 + 1.4 * n_cols), 0.35 * n_rows + 1.6), facecolor=SURFACE
    )
    vmax = float(np.abs(data.to_numpy()).max())
    im = ax.imshow(data.to_numpy(), cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"cluster {c}" for c in data.columns], color=INK_PRIMARY, fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(data.index, color=INK_PRIMARY, fontsize=9)
    ax.set_title("Cluster profile: mean z-score per feature", color=INK_PRIMARY, fontsize=12)
    _grid_heatmap_axes(ax, n_rows, n_cols)

    for i in range(n_rows):
        for j in range(n_cols):
            val = data.iat[i, j]
            text_color = "#ffffff" if abs(val) > vmax * 0.55 else INK_PRIMARY
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("mean z-score", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def plot_silhouette_vs_k(kmeans_results, out_path):
    """Line chart of silhouette score vs. k, with the selected k marked."""
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor=SURFACE)
    ax.plot(
        kmeans_results["k"], kmeans_results["silhouette"],
        color=CATEGORICAL[0], linewidth=2, marker="o", markersize=7,
        markerfacecolor=CATEGORICAL[0], markeredgecolor=SURFACE, markeredgewidth=1,
        zorder=3,
    )

    best_idx = kmeans_results["silhouette"].idxmax()
    best_k = kmeans_results.loc[best_idx, "k"]
    best_sil = kmeans_results.loc[best_idx, "silhouette"]
    ax.scatter(
        [best_k], [best_sil], s=110, facecolor="none",
        edgecolor=CATEGORICAL[1], linewidth=2, zorder=4,
    )
    ax.annotate(
        f"selected k={int(best_k)}", (best_k, best_sil),
        textcoords="offset points", xytext=(10, 8), fontsize=9, color=INK_SECONDARY,
    )

    ax.set_xlabel("k")
    ax.set_ylabel("silhouette score")
    ax.set_title("KMeans silhouette score vs. k (train)", color=INK_PRIMARY, fontsize=12)
    ax.set_xticks(kmeans_results["k"])
    ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    _clean_axes(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def plot_feature_correlation_heatmap(corr_df, out_path):
    """Heatmap of a feature x feature Pearson correlation matrix (train
    windows). Fixed -1..1 color range (not data-driven) since correlation is
    already on a fixed scale, unlike the z-score profile heatmap above."""
    data = corr_df.to_numpy()
    n = len(corr_df)

    fig, ax = plt.subplots(figsize=(1.1 + 0.5 * n, 1.1 + 0.5 * n), facecolor=SURFACE)
    im = ax.imshow(data, cmap=DIVERGING_CMAP, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(corr_df.columns, color=INK_PRIMARY, fontsize=8, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr_df.index, color=INK_PRIMARY, fontsize=8)
    ax.set_title("Feature correlation matrix (train windows, Pearson r)", color=INK_PRIMARY, fontsize=12)
    _grid_heatmap_axes(ax, n, n)

    for i in range(n):
        for j in range(n):
            val = data[i, j]
            text_color = "#ffffff" if abs(val) > 0.6 else INK_PRIMARY
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Pearson r", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion_matrix(cm_df, out_path):
    """Heatmap of a labeled confusion matrix DataFrame
    (index 'true_<label>', columns 'pred_<label>')."""
    data = cm_df.to_numpy()
    n_rows, n_cols = data.shape
    labels_true = [s.replace("true_", "") for s in cm_df.index]
    labels_pred = [s.replace("pred_", "") for s in cm_df.columns]

    fig, ax = plt.subplots(figsize=(5.5, 4.8), facecolor=SURFACE)
    im = ax.imshow(data, cmap=SEQUENTIAL_CMAP, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(labels_pred, color=INK_PRIMARY, fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels_true, color=INK_PRIMARY, fontsize=9)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title("RandomForest confusion matrix (held-out test)", color=INK_PRIMARY, fontsize=12)
    _grid_heatmap_axes(ax, n_rows, n_cols)

    vmax = data.max()
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            text_color = "#ffffff" if val > vmax * 0.55 else INK_PRIMARY
            ax.text(j, i, str(int(val)), ha="center", va="center", fontsize=10, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("count", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
