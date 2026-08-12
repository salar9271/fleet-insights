"""Generate the project's report figures (PNG, dpi=150) into
reports/figures/. Reuses the existing clustering and diagnostic pipelines
read-only -- does not change clustering or reporting logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.diagnostics import diagnostic_a
from src.clustering import TEST_WINDOWS_PATH, TRAIN_WINDOWS_PATH, load_windows, run_clustering
from src.visualise import (
    plot_cluster_heatmap,
    plot_confusion_matrix,
    plot_pca_scatter,
    plot_silhouette_vs_k,
)

FIGURES_DIR = Path("reports/figures")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = run_clustering()
    train_df = load_windows(TRAIN_WINDOWS_PATH)
    test_df = load_windows(TEST_WINDOWS_PATH)
    feature_cols = results["feature_cols"]
    X_train_scaled = results["scaler"].transform(train_df[feature_cols])

    out = FIGURES_DIR / "pca_clusters_vs_class.png"
    plot_pca_scatter(X_train_scaled, results["train_labels"], train_df["class"].to_numpy(), out)
    print(f"Saved {out}")

    out = FIGURES_DIR / "cluster_profile_heatmap.png"
    plot_cluster_heatmap(results["profile"].drop(columns=["size"]), out)
    print(f"Saved {out}")

    out = FIGURES_DIR / "silhouette_vs_k.png"
    plot_silhouette_vs_k(results["kmeans_results"], out)
    print(f"Saved {out}")

    diag_a = diagnostic_a(train_df, test_df, feature_cols)
    out = FIGURES_DIR / "rf_confusion_matrix.png"
    plot_confusion_matrix(diag_a["confusion_matrix"], out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
