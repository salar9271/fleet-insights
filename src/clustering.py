"""Unsupervised clustering over the windowed features in data/processed/.

The scaler and every clustering model are fit on TRAIN windows only. TEST
windows are only ever transformed/predicted, never fit on. True class labels
are used exclusively for post-hoc external validation (ARI, contingency
tables) -- they play no role in scaling, clustering, k selection, eps
selection, or cluster naming.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

TRAIN_WINDOWS_PATH = "data/processed/windows_train.parquet"
TEST_WINDOWS_PATH = "data/processed/windows_test.parquet"

NON_FEATURE_COLS = {"session_id", "class", "session_fs_hz"}
K_RANGE = range(2, 7)
DBSCAN_MIN_SAMPLES = 5
DBSCAN_ESP_PERCENTILES = (25, 50, 75, 90)
RANDOM_STATE = 42
CLUSTER_NAME_TOP_N = 2

# The feature set the deployed pipeline (models/scaler.joblib,
# models/kmeans.joblib) is fit on and the API expects every upload to
# produce. Named explicitly -- rather than "every non-metadata column in the
# windows dataframe" -- so that adding a new feature to src/features.py (e.g.
# gyro_mag_std, added for the reports/feature_reduction_findings.md
# comparison) can never silently change what the deployed model is scored
# against. See reports/feature_reduction_findings.md for the redundancy
# analysis and the reduced-feature-set comparison, which use their own
# explicit column lists rather than this one.
ORIGINAL_FEATURE_COLS = [
    "acc_mag_mean", "acc_mag_std", "acc_mag_max", "acc_mag_p95",
    "accY_mean", "accY_std", "accY_rate_below_neg3", "accY_rate_above_pos3",
    "accX_std", "accX_max_abs",
    "jerk_std", "jerk_rms",
    "gyroX_std", "gyroY_std", "gyroZ_std",
    "spectral_energy_ratio_0.2_0.8hz",
]


def load_windows(path):
    return pd.read_parquet(path)


def get_feature_columns(df):
    """The deployed pipeline's feature contract (ORIGINAL_FEATURE_COLS),
    filtered to columns actually present in df. Used by the API and by the
    original-16-feature runs, so it must stay tied to that fixed list rather
    than "all non-metadata columns" -- see the ORIGINAL_FEATURE_COLS comment."""
    return [c for c in ORIGINAL_FEATURE_COLS if c in df.columns]


class FeatureContractError(RuntimeError):
    """Raised when a saved scaler/KMeans pair does not match the current
    feature contract (ORIGINAL_FEATURE_COLS) -- e.g. src/features.py changed
    (a feature added, removed, or redefined) without re-running
    scripts/run_clustering.py to refit models/scaler.joblib and
    models/kmeans.joblib against the new feature set."""


def validate_feature_contract(scaler, kmeans, expected_cols=ORIGINAL_FEATURE_COLS):
    """Fails loudly (rather than silently mis-scoring requests) if a loaded
    scaler/KMeans pair was not fit on exactly expected_cols, in order.
    scaler.feature_names_in_ is set by sklearn because fit_scaler() always
    fits on a DataFrame (see fit_scaler() above); KMeans is fit on the
    scaler's numpy-array output, so it never has feature names of its own,
    but its n_features_in_ must still agree with the scaler's."""
    fitted_cols = list(getattr(scaler, "feature_names_in_", []))
    if fitted_cols != list(expected_cols):
        raise FeatureContractError(
            f"Saved scaler's feature names {fitted_cols} do not match the "
            f"current feature contract {list(expected_cols)}. The artifacts "
            f"in models/ are stale relative to src/features.py / "
            f"src/clustering.py -- re-run scripts/run_clustering.py."
        )
    if kmeans.n_features_in_ != scaler.n_features_in_:
        raise FeatureContractError(
            f"Saved KMeans expects {kmeans.n_features_in_} features but the "
            f"saved scaler produces {scaler.n_features_in_} -- models/scaler.joblib "
            f"and models/kmeans.joblib were not fit together. Re-run "
            f"scripts/run_clustering.py to regenerate both from the same run."
        )


def fit_scaler(train_df, feature_cols):
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def kmeans_sweep(X_train, k_range=K_RANGE, random_state=RANDOM_STATE):
    rows = []
    models = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_train)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X_train, labels),
        })
        models[k] = km
    return pd.DataFrame(rows), models


def select_best_k(kmeans_results):
    """Unsupervised model selection: highest train silhouette. Never uses
    true labels."""
    return int(kmeans_results.loc[kmeans_results["silhouette"].idxmax(), "k"])


def dbscan_eps_sweep(X_train, min_samples=DBSCAN_MIN_SAMPLES,
                      percentiles=DBSCAN_ESP_PERCENTILES, true_labels=None):
    """eps values are derived from the train k-distance graph (unsupervised).
    true_labels, if given, is used only to report an ARI column for
    comparison -- it plays no part in choosing eps."""
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X_train)
    kth_dist = np.sort(nn.kneighbors(X_train)[0][:, -1])
    eps_values = sorted(set(np.percentile(kth_dist, p).round(4) for p in percentiles))

    rows = []
    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_train)
        non_noise = labels != -1
        n_clusters = len(set(labels[non_noise]))

        sil = float("nan")
        if n_clusters >= 2 and non_noise.sum() > n_clusters:
            sil = silhouette_score(X_train[non_noise], labels[non_noise])

        ari = adjusted_rand_score(true_labels, labels) if true_labels is not None else float("nan")

        rows.append({
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "n_noise": int((~non_noise).sum()),
            "silhouette": sil,
            "ari_train_comparison_only": ari,
        })
    return pd.DataFrame(rows)


def cluster_profile(X_scaled, labels, feature_cols):
    """Mean z-score of each feature per cluster (X_scaled is already
    StandardScaler output, so its column means ARE z-scores), plus size."""
    df = pd.DataFrame(X_scaled, columns=feature_cols)
    df["cluster"] = labels
    profile = df.groupby("cluster")[feature_cols].mean()
    profile["size"] = df.groupby("cluster").size()
    return profile


def name_cluster(z_row, feature_cols, top_n=CLUSTER_NAME_TOP_N):
    """Human-readable name built only from the cluster's own feature
    z-scores -- the top-|z| features and their direction. Never looks at
    true labels."""
    z = z_row[feature_cols]
    top_features = z.abs().sort_values(ascending=False).index[:top_n]
    parts = [f"{'High' if z[f] > 0 else 'Low'} {f}" for f in top_features]
    return " / ".join(parts)


def name_clusters(profile, feature_cols, top_n=CLUSTER_NAME_TOP_N):
    return {cluster: name_cluster(row, feature_cols, top_n) for cluster, row in profile.iterrows()}


def external_validation(true_labels, cluster_labels):
    ari = adjusted_rand_score(true_labels, cluster_labels)
    contingency = pd.crosstab(
        pd.Series(cluster_labels, name="cluster"),
        pd.Series(true_labels, name="class"),
    )
    return ari, contingency


def run_clustering(train_path=TRAIN_WINDOWS_PATH, test_path=TEST_WINDOWS_PATH,
                    k_range=K_RANGE, random_state=RANDOM_STATE):
    train_df = load_windows(train_path)
    test_df = load_windows(test_path)
    feature_cols = get_feature_columns(train_df)

    scaler = fit_scaler(train_df, feature_cols)
    X_train = scaler.transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    kmeans_results, kmeans_models = kmeans_sweep(X_train, k_range, random_state)
    best_k = select_best_k(kmeans_results)
    best_model = kmeans_models[best_k]

    train_labels = best_model.labels_
    test_labels = best_model.predict(X_test)

    profile = cluster_profile(X_train, train_labels, feature_cols)
    cluster_names = name_clusters(profile, feature_cols)

    dbscan_results = dbscan_eps_sweep(X_train, true_labels=train_df["class"].to_numpy())

    train_ari, train_contingency = external_validation(train_df["class"].to_numpy(), train_labels)
    test_ari, test_contingency = external_validation(test_df["class"].to_numpy(), test_labels)

    return {
        "feature_cols": feature_cols,
        "scaler": scaler,
        "kmeans_results": kmeans_results,
        "best_k": best_k,
        "best_model": best_model,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "profile": profile,
        "cluster_names": cluster_names,
        "dbscan_results": dbscan_results,
        "train_ari": train_ari,
        "train_contingency": train_contingency,
        "test_ari": test_ari,
        "test_contingency": test_contingency,
    }
