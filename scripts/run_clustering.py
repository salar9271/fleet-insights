"""Run unsupervised clustering on data/processed/windows_{train,test}.parquet
and print the full evaluation. Saves the fitted scaler + KMeans to models/
and the cluster profile to reports/cluster_profile.csv.

True labels are used only for the external-validation section below -- they
never influence scaling, clustering, k selection, eps selection, or naming.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib

from src.clustering import run_clustering

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def main():
    results = run_clustering()

    print("=== KMeans sweep (k=2..6), fit on TRAIN only ===")
    print(results["kmeans_results"].to_string(index=False))
    print(f"\nChosen k (max train silhouette, unsupervised): {results['best_k']}")

    print("\n=== DBSCAN eps sweep, for comparison (eps from TRAIN k-distance percentiles) ===")
    print(results["dbscan_results"].to_string(index=False))

    print("\n=== Cluster profile (TRAIN, mean z-score per feature) ===")
    print(results["profile"].to_string())

    print("\n=== Cluster names (derived from feature z-scores only, not true labels) ===")
    for cluster, name in results["cluster_names"].items():
        print(f"  cluster {cluster}: {name}")

    print("\n=== External validation -- TRAIN ===")
    print(f"Adjusted Rand Index (train): {results['train_ari']:.4f}")
    print("Cluster x class contingency table (train):")
    print(results["train_contingency"].to_string())

    print("\n=== External validation -- TEST (held-out) ===")
    print(f">>> Adjusted Rand Index (TEST, held-out): {results['test_ari']:.4f} <<<")
    print("Cluster x class contingency table (test):")
    print(results["test_contingency"].to_string())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(results["scaler"], MODELS_DIR / "scaler.joblib")
    joblib.dump(results["best_model"], MODELS_DIR / "kmeans.joblib")

    profile_out = results["profile"].copy()
    profile_out.insert(0, "cluster_name", [results["cluster_names"][c] for c in profile_out.index])
    profile_out.to_csv(REPORTS_DIR / "cluster_profile.csv")

    print("\nSaved scaler -> models/scaler.joblib")
    print(f"Saved KMeans (k={results['best_k']}) -> models/kmeans.joblib")
    print("Saved cluster profile -> reports/cluster_profile.csv")


if __name__ == "__main__":
    main()
