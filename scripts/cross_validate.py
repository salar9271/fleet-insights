"""C2 diagnostic: leave-one-session-out cross-validation across the 4 train
sessions, to give the headline point estimates (silhouette-selected k, test
ARI, RandomForest macro F1) a spread rather than reporting them as if a
single fixed train/test split were a stable, generalizable number.

Each of the 4 train sessions (session_id train-0..3) is held out in turn;
the other 3 sessions are used to fit a fresh StandardScaler, sweep KMeans
over k=2..6 (k selected by train silhouette, exactly as
scripts/run_clustering.py does), and train a RandomForestClassifier
(n_estimators=300, same random_state) -- both evaluated against the held-out
session. The real held-out test_motion_data.csv sessions are never touched
here; this is purely about how stable the train-side numbers are across
which session plays "held out."

IMPORTANT caveat, confirmed by this script's own output: every train
session is single-class (train-0=NORMAL, train-1=AGGRESSIVE, train-2=SLOW,
train-3=SLOW -- see docs/data_selection.md). That makes the per-fold ARI
degenerate: adjusted_rand_score against a held-out session whose true
labels are all one class is 0.0 by definition for any non-trivial cluster
assignment, regardless of clustering quality -- a property of the ARI
formula on a single-class reference partition, not a statement about this
project's clustering. The per-fold ARI is still reported, because that is
the literal, honest answer to what was asked; a pooled ARI (concatenating
out-of-fold cluster predictions across all 4 folds, which together span all
3 classes) is reported alongside it as a non-degenerate supplementary read
on the same question.

The same single-class-per-session structure means 2 of the 4 RandomForest
folds (holding out train-0 or train-1) train on only 2 of the 3 classes and
evaluate on a class the model never saw at all -- see the per-fold
"train fold classes present" column.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score, f1_score

from src.clustering import (
    K_RANGE,
    ORIGINAL_FEATURE_COLS,
    RANDOM_STATE,
    TRAIN_WINDOWS_PATH,
    fit_scaler,
    kmeans_sweep,
    load_windows,
    select_best_k,
)

REPORTS_DIR = Path("reports")


def run_fold(train_df, held_out_session, feature_cols=ORIGINAL_FEATURE_COLS,
             k_range=K_RANGE, random_state=RANDOM_STATE):
    fold_train = train_df[train_df["session_id"] != held_out_session]
    fold_holdout = train_df[train_df["session_id"] == held_out_session]

    scaler = fit_scaler(fold_train, feature_cols)
    X_train = scaler.transform(fold_train[feature_cols])
    X_holdout = scaler.transform(fold_holdout[feature_cols])
    y_train = fold_train["class"].to_numpy()
    y_holdout = fold_holdout["class"].to_numpy()

    kmeans_results, kmeans_models = kmeans_sweep(X_train, k_range, random_state)
    best_k = select_best_k(kmeans_results)
    best_model = kmeans_models[best_k]
    holdout_cluster_labels = best_model.predict(X_holdout)
    ari = adjusted_rand_score(y_holdout, holdout_cluster_labels)

    clf = RandomForestClassifier(n_estimators=300, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_holdout)
    macro_f1 = f1_score(y_holdout, y_pred, average="macro", zero_division=0)

    return {
        "held_out_session": held_out_session,
        "held_out_class": y_holdout[0],
        "n_holdout": len(fold_holdout),
        "train_classes": sorted(set(y_train)),
        "best_k": best_k,
        "ari": ari,
        "macro_f1": macro_f1,
        "holdout_cluster_labels": holdout_cluster_labels,
        "y_holdout": y_holdout,
    }


def main():
    train_df = load_windows(TRAIN_WINDOWS_PATH)
    sessions = sorted(train_df["session_id"].unique())

    print(f"=== Leave-one-session-out CV across {len(sessions)} train sessions ===")
    folds = [run_fold(train_df, s) for s in sessions]

    rows = []
    for f in folds:
        print(f"\nHeld out: {f['held_out_session']} (class={f['held_out_class']}, n={f['n_holdout']})")
        print(f"  Train fold classes present: {f['train_classes']}")
        print(f"  Chosen k: {f['best_k']}  ARI: {f['ari']:.4f}  Macro F1: {f['macro_f1']:.4f}")
        rows.append({
            "held_out_session": f["held_out_session"],
            "held_out_class": f["held_out_class"],
            "n_holdout": f["n_holdout"],
            "train_classes_present": "+".join(f["train_classes"]),
            "chosen_k": f["best_k"],
            "ari": f["ari"],
            "macro_f1": f["macro_f1"],
        })

    results_df = pd.DataFrame(rows)

    pooled_true = np.concatenate([f["y_holdout"] for f in folds])
    pooled_pred = np.concatenate([f["holdout_cluster_labels"] for f in folds])
    pooled_ari = adjusted_rand_score(pooled_true, pooled_pred)

    print("\n=== Summary across folds ===")
    print(results_df.to_string(index=False))
    print(f"\nChosen k:  mean={results_df['chosen_k'].mean():.2f}  std={results_df['chosen_k'].std():.2f}  "
          f"values={list(results_df['chosen_k'])}")
    print(f"Test ARI:  mean={results_df['ari'].mean():.2f}  std={results_df['ari'].std():.2f}")
    print(f"Macro F1:  mean={results_df['macro_f1'].mean():.2f}  std={results_df['macro_f1'].std():.2f}")
    print(f"\nPooled ARI (out-of-fold cluster labels vs. true class, all 4 folds concatenated): {pooled_ari:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORTS_DIR / "cross_validation_folds.csv", index=False)
    print(f"\nSaved {REPORTS_DIR / 'cross_validation_folds.csv'}")


if __name__ == "__main__":
    main()
