"""C1 diagnostic: feature redundancy in the original 16-feature set, and a
re-clustering comparison on a de-duplicated feature set.

Three things, in order:

A) Full pairwise correlation matrix on the original 16 train features ->
   reports/feature_correlation.csv, reports/figures/feature_correlation_heatmap.png.

B) A reduced feature set that drops the redundancy the matrix in (A) shows:
   jerk_rms (r=1.00 with jerk_std), three of the four acc_mag_* summary
   stats (keep the single highest-permutation-importance one -- see
   REDUCED_FEATURE_COLS below for the exact numbers), and accX_max_abs
   (r=0.86 with accX_std). The full clustering procedure from
   src.clustering.run_clustering() -- train-only StandardScaler, KMeans
   swept over k=2..6, k selected by train silhouette, external validation
   by test ARI -- is re-run on this reduced set and saved separately to
   models/scaler_reduced.joblib / models/kmeans_reduced.joblib. The original
   16-feature artifacts (models/scaler.joblib, models/kmeans.joblib, the
   ones the API actually loads) are untouched.

C) The same reduced set plus gyro_mag_std (L2 norm of the three gyro axes --
   added to src/features.py alongside this script, mirroring how acc_mag is
   a mount-orientation-invariant summary of the accelerometer axes), saved
   to models/scaler_reduced_gyro.joblib / models/kmeans_reduced_gyro.joblib.

Original, reduced, and reduced+gyro results are all printed and saved side
by side. None replaces another -- reports/feature_reduction_findings.md
turns this script's output into the write-up; the comparison itself is the
finding.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib

from src.clustering import (
    K_RANGE,
    ORIGINAL_FEATURE_COLS,
    RANDOM_STATE,
    TEST_WINDOWS_PATH,
    TRAIN_WINDOWS_PATH,
    cluster_profile,
    external_validation,
    fit_scaler,
    kmeans_sweep,
    load_windows,
    select_best_k,
)
from src.visualise import plot_feature_correlation_heatmap

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

# acc_mag_p95 is the sole acc_mag-family representative kept below: of
# acc_mag_mean/std/max/p95 (pairwise r=0.79-0.94 on the train windows, see
# reports/feature_correlation.csv), acc_mag_p95 has the highest RandomForest
# permutation importance (0.0213, vs 0.0101 for acc_mag_std and negative
# for acc_mag_mean/acc_mag_max -- reports/rf_permutation_importance.csv),
# so it is the one member of that family actually pulling weight in the
# supervised diagnostic, not an arbitrary pick.
#
# jerk_rms dropped: r=1.00 with jerk_std (reports/feature_correlation.csv)
# -- with per-window mean jerk close to zero, RMS and STD of the same
# signal are numerically the same feature twice.
#
# accX_max_abs dropped: r=0.86 with accX_std (reports/feature_correlation.csv).
REDUCED_FEATURE_COLS = [
    "acc_mag_p95",
    "accY_mean", "accY_std", "accY_rate_below_neg3", "accY_rate_above_pos3",
    "accX_std",
    "jerk_std",
    "gyroX_std", "gyroY_std", "gyroZ_std",
    "spectral_energy_ratio_0.2_0.8hz",
]
REDUCED_PLUS_GYRO_FEATURE_COLS = REDUCED_FEATURE_COLS + ["gyro_mag_std"]


def correlation_matrix(train_df, feature_cols=ORIGINAL_FEATURE_COLS):
    return train_df[feature_cols].corr()


def top_correlated_pairs(corr, threshold=0.75):
    cols = list(corr.columns)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) > threshold:
                pairs.append((a, b, float(r)))
    pairs.sort(key=lambda x: -abs(x[2]))
    return pairs


def run_variant(train_df, test_df, feature_cols, k_range=K_RANGE, random_state=RANDOM_STATE):
    """Same procedure as src.clustering.run_clustering(), parameterized by
    feature_cols so it can be re-run on a reduced feature set without
    touching the original 16-feature pipeline or its saved artifacts."""
    scaler = fit_scaler(train_df, feature_cols)
    X_train = scaler.transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    kmeans_results, kmeans_models = kmeans_sweep(X_train, k_range, random_state)
    best_k = select_best_k(kmeans_results)
    best_model = kmeans_models[best_k]

    train_labels = best_model.labels_
    test_labels = best_model.predict(X_test)

    profile = cluster_profile(X_train, train_labels, feature_cols)
    test_ari, test_contingency = external_validation(test_df["class"].to_numpy(), test_labels)

    return {
        "feature_cols": feature_cols,
        "scaler": scaler,
        "kmeans_results": kmeans_results,
        "best_k": best_k,
        "best_model": best_model,
        "profile": profile,
        "test_ari": test_ari,
        "test_contingency": test_contingency,
    }


def print_variant(name, result):
    print(f"\n=== {name}: {len(result['feature_cols'])} features ===")
    print(f"Features: {result['feature_cols']}")
    print(result["kmeans_results"].to_string(index=False))
    print(f"Chosen k (max train silhouette): {result['best_k']}")
    print(f"ARI (test, held-out): {result['test_ari']:.4f}")
    print("Cluster profile (train, mean z-score per feature):")
    print(result["profile"].to_string())
    print("Contingency table (test):")
    print(result["test_contingency"].to_string())


def main():
    train_df = load_windows(TRAIN_WINDOWS_PATH)
    test_df = load_windows(TEST_WINDOWS_PATH)

    print("=== Feature correlation matrix (original 16 features, train) ===")
    corr = correlation_matrix(train_df)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    corr.to_csv(REPORTS_DIR / "feature_correlation.csv")
    print(f"Saved {REPORTS_DIR / 'feature_correlation.csv'}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_feature_correlation_heatmap(corr, FIGURES_DIR / "feature_correlation_heatmap.png")
    print(f"Saved {FIGURES_DIR / 'feature_correlation_heatmap.png'}")

    pairs = top_correlated_pairs(corr)
    print("\nPairs at |r| > 0.75:")
    for a, b, r in pairs:
        print(f"  {a:30s} {b:30s} r={r:.3f}")

    original = run_variant(train_df, test_df, ORIGINAL_FEATURE_COLS)
    reduced = run_variant(train_df, test_df, REDUCED_FEATURE_COLS)
    reduced_gyro = run_variant(train_df, test_df, REDUCED_PLUS_GYRO_FEATURE_COLS)

    print_variant("Original (deployed)", original)
    print_variant("Reduced", reduced)
    print_variant("Reduced + gyro_mag_std", reduced_gyro)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(reduced["scaler"], MODELS_DIR / "scaler_reduced.joblib")
    joblib.dump(reduced["best_model"], MODELS_DIR / "kmeans_reduced.joblib")
    joblib.dump(reduced_gyro["scaler"], MODELS_DIR / "scaler_reduced_gyro.joblib")
    joblib.dump(reduced_gyro["best_model"], MODELS_DIR / "kmeans_reduced_gyro.joblib")

    reduced["profile"].to_csv(REPORTS_DIR / "cluster_profile_reduced.csv")
    reduced_gyro["profile"].to_csv(REPORTS_DIR / "cluster_profile_reduced_gyro.csv")

    print("\nSaved models/scaler_reduced.joblib, models/kmeans_reduced.joblib")
    print("Saved models/scaler_reduced_gyro.joblib, models/kmeans_reduced_gyro.joblib")
    print("Saved reports/cluster_profile_reduced.csv, reports/cluster_profile_reduced_gyro.csv")
    print("(models/scaler.joblib and models/kmeans.joblib, the deployed original-16-feature artifacts, are untouched)")


if __name__ == "__main__":
    main()
