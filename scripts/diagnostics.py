"""Two bounded diagnostics on the ARI ~0.12 clustering result:

A) Is the signal there at all? Supervised RandomForest, train windows ->
   held-out test windows: macro F1, confusion matrix, permutation importance.
B) Does k=3 tell a different story? k=2 remains the silhouette-selected
   choice from scripts/run_clustering.py; k=3 is reported here purely as a
   labelled-taxonomy comparison, saved alongside (not overwriting) the k=2
   artefacts.

Also reports per-feature separation (mean z-score by TRUE class, train) to
see which features distinguish AGGRESSIVE from NORMAL, if any.

Scope is intentionally limited to what's listed above.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, f1_score

from src.clustering import (
    K_RANGE,
    RANDOM_STATE,
    TEST_WINDOWS_PATH,
    TRAIN_WINDOWS_PATH,
    cluster_profile,
    external_validation,
    fit_scaler,
    get_feature_columns,
    kmeans_sweep,
    load_windows,
)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def diagnostic_a(train_df, test_df, feature_cols):
    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["class"].to_numpy()
    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df["class"].to_numpy()

    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    labels_sorted = sorted(set(y_train) | set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{l}" for l in labels_sorted],
        columns=[f"pred_{l}" for l in labels_sorted],
    )

    # Computed on the held-out test set: reflects each feature's
    # contribution to generalization, not to fitting the training data.
    perm = permutation_importance(
        clf, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE, scoring="f1_macro"
    )
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return {
        "model": clf,
        "macro_f1": macro_f1,
        "confusion_matrix": cm_df,
        "importance": importance_df,
    }


def diagnostic_b(train_df, test_df, feature_cols, scaler):
    X_train = scaler.transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    kmeans_results, kmeans_models = kmeans_sweep(X_train, K_RANGE, RANDOM_STATE)
    model_k3 = kmeans_models[3]

    train_labels = model_k3.labels_
    test_labels = model_k3.predict(X_test)

    train_ari, train_contingency = external_validation(train_df["class"].to_numpy(), train_labels)
    test_ari, test_contingency = external_validation(test_df["class"].to_numpy(), test_labels)

    profile = cluster_profile(X_train, train_labels, feature_cols)

    sil_k2 = float(kmeans_results.loc[kmeans_results["k"] == 2, "silhouette"].iloc[0])
    sil_k3 = float(kmeans_results.loc[kmeans_results["k"] == 3, "silhouette"].iloc[0])

    return {
        "model": model_k3,
        "train_ari": train_ari,
        "train_contingency": train_contingency,
        "test_ari": test_ari,
        "test_contingency": test_contingency,
        "profile": profile,
        "silhouette_k2": sil_k2,
        "silhouette_k3": sil_k3,
    }


def per_feature_separation_by_class(train_df, feature_cols, scaler):
    X_train = scaler.transform(train_df[feature_cols])
    df = pd.DataFrame(X_train, columns=feature_cols)
    df["class"] = train_df["class"].to_numpy()
    return df.groupby("class")[feature_cols].mean()


def main():
    train_df = load_windows(TRAIN_WINDOWS_PATH)
    test_df = load_windows(TEST_WINDOWS_PATH)
    feature_cols = get_feature_columns(train_df)
    scaler = fit_scaler(train_df, feature_cols)

    print("=== Diagnostic A: supervised RandomForest, TRAIN windows -> held-out TEST windows ===")
    a = diagnostic_a(train_df, test_df, feature_cols)
    print(f"Macro F1 (test): {a['macro_f1']:.4f}")
    print("\nConfusion matrix (test):")
    print(a["confusion_matrix"].to_string())
    print("\nPermutation importance (test set, f1_macro, 30 repeats), sorted:")
    print(a["importance"].to_string(index=False))

    print("\n=== Diagnostic B: k=3 comparison ===")
    print("(k=2 remains the silhouette-selected model from run_clustering.py; k=3 below is a labelled-taxonomy comparison only)")
    b = diagnostic_b(train_df, test_df, feature_cols, scaler)
    print(f"silhouette k=2: {b['silhouette_k2']:.4f}  |  silhouette k=3: {b['silhouette_k3']:.4f}")
    print(f"ARI (train, k=3): {b['train_ari']:.4f}")
    print(f"ARI (test, k=3): {b['test_ari']:.4f}")
    print("\nContingency table (test, k=3):")
    print(b["test_contingency"].to_string())
    print("\nCluster profile (train, mean z-score per feature, k=3):")
    print(b["profile"].to_string())

    print("\n=== Per-feature separation by TRUE class (train, mean z-score) ===")
    sep = per_feature_separation_by_class(train_df, feature_cols, scaler)
    print(sep.to_string())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(a["model"], MODELS_DIR / "random_forest.joblib")
    joblib.dump(b["model"], MODELS_DIR / "kmeans_k3.joblib")

    b["profile"].to_csv(REPORTS_DIR / "cluster_profile_k3.csv")
    sep.to_csv(REPORTS_DIR / "feature_separation_by_class.csv")
    a["importance"].to_csv(REPORTS_DIR / "rf_permutation_importance.csv", index=False)

    print("\nSaved models/random_forest.joblib, models/kmeans_k3.joblib")
    print("Saved reports/cluster_profile_k3.csv, reports/feature_separation_by_class.csv, reports/rf_permutation_importance.csv")
    print("(models/kmeans.joblib and reports/cluster_profile.csv from the k=2 run are untouched)")


if __name__ == "__main__":
    main()
