# Leave-one-session-out cross-validation

Context: an audit of this project (finding C2) pointed out that every
headline number reported elsewhere in this repo — silhouette 0.37 at k=2,
test ARI 0.12, RandomForest macro F1 0.49 — is a single point estimate from
one fixed train/test split, computed over 180/152 *windows* that reduce to
only **4 independent train sessions** (`docs/data_selection.md`). No
resampling had been run to show whether those numbers are stable or an
artifact of this one particular split. This report runs leave-one-session-out
(LOSO) cross-validation across the 4 train sessions —`scripts/cross_validate.py`
— to put a spread on them. Full per-fold output:
[`reports/cross_validation_folds.csv`](cross_validation_folds.csv).

**Read this whole document before quoting the macro F1 numbers below** — the
headline mean is dramatically lower than the 0.49 reported elsewhere, for a
specific, mechanical reason explained in "Why macro F1 collapses" below, not
because the model is secretly much worse than previously reported.

## Setup

Each of the 4 train sessions is held out in turn; the other 3 fit a fresh
`StandardScaler`, a KMeans sweep over k=2..6 (k selected by train silhouette,
same procedure as `scripts/run_clustering.py`), and a
`RandomForestClassifier` (same hyperparameters as `scripts/diagnostics.py`).
Both are evaluated against the held-out session. The real held-out
`test_motion_data.csv` sessions are never touched by this — this is
entirely about how stable the train-side numbers are across which session
plays "held out."

A structural fact about this dataset drives everything below: **every train
session is single-class** (`train-0`=NORMAL, `train-1`=AGGRESSIVE,
`train-2`=SLOW, `train-3`=SLOW — SLOW is the class split across two sessions
by a mid-recording pause, per `docs/data_selection.md`). Holding out a
session therefore always means testing against a held-out set with exactly
one true class in it, and — for `train-0`/`train-1` specifically — training
on only 2 of the 3 classes, since the third class's only session is the one
being held out.

## Results

| held out | class | n | train fold has | chosen k | ARI (fold) | macro F1 (fold) |
|---|---|---|---|---|---|---|
| train-0 | NORMAL | 60 | AGGRESSIVE + SLOW only | 2 | 0.00 | 0.00 |
| train-1 | AGGRESSIVE | 55 | NORMAL + SLOW only | 2 | 0.00 | 0.00 |
| train-2 | SLOW | 26 | all 3 classes | 2 | 0.00 | 0.13 |
| train-3 | SLOW | 39 | all 3 classes | 2 | 0.00 | 0.02 |

**Silhouette-selected k: mean = 2.00, std = 0.00.** Every fold picks k=2,
including the two folds where the training data only contains 2 of the 3
classes. This is the one number in this report that's a clean, stable
result: unsupervised k-selection did not depend on which session was held
out.

**Test ARI (per fold): mean = 0.00, std = 0.00 — and this is not a finding
about clustering quality.** `adjusted_rand_score` against a reference
partition that is a single class is 0.0 **by definition**, for any
non-trivial cluster assignment, regardless of how good the clustering is —
verified directly against `sklearn`: `adjusted_rand_score(['A']*10, [any
non-constant assignment])` returns exactly `0.0` every time. Because every
held-out session here is single-class, per-fold ARI is mathematically
guaranteed to be 0.00 before any model is even fit. It answers the literal
question asked ("report mean and standard deviation for test ARI"), but the
answer is a property of this dataset's session/class structure, not a
measurement.

A non-degenerate substitute: pooling the out-of-fold cluster predictions
from all 4 folds (which together do span all 3 classes) and computing one
ARI against the pooled true labels gives **pooled ARI = 0.16**. That is in
the same range as the original fixed-split test ARI (0.12,
`reports/clustering_findings.md`) — reassuring, in that the "clustering
weakly agrees with the 3-class taxonomy, ARI in the 0.1–0.2 range" finding
holds up under a different held-out arrangement and isn't an artifact of the
specific `test_motion_data.csv` split.

**RandomForest macro F1 (per fold): mean = 0.04, std = 0.06.** This is far
below the 0.49 macro F1 reported in `reports/clustering_findings.md`, and
that gap needs an explanation, not just a number.

## Why macro F1 collapses

Two different things are happening in the four folds, and both push macro
F1 down for reasons that have little to do with whether the underlying
features are informative:

1. **`train-0` and `train-1`: the held-out class is entirely absent from
   training.** With only one session per class (except SLOW), holding out
   `train-0` (NORMAL) means the RandomForest never sees a single NORMAL
   example during training — it is structurally unable to predict NORMAL
   for any window, ever. Every one of the 60 held-out NORMAL windows is
   necessarily misclassified, and macro F1 for that fold is exactly 0.00.
   Same story for `train-1` (AGGRESSIVE). This isn't a measurement of model
   generalization; it's a direct consequence of a session-level holdout on
   a dataset with only one non-split class per session — there is no way
   for any classifier to do better in this exact setup.

2. **`train-2` and `train-3`: all 3 classes are present in training, and
   macro F1 is still low (0.13, 0.02) — because the held-out set is 100%
   one class.** `f1_score(..., average="macro")` averages per-class F1 over
   every class that appears in either the true or predicted labels. With
   `y_true` entirely SLOW, any AGGRESSIVE or NORMAL prediction the model
   makes on these windows contributes a class with 0 true positives and
   nonzero false positives — F1 = 0 for that class — and that 0 gets
   averaged in alongside SLOW's own F1. A handful of wrong predictions on a
   single-class holdout can dominate the macro average in a way a balanced,
   3-class test set (the original 40/49/63-window split) does not.

The original macro F1 of 0.49 is a real, honestly-computed number on a real
held-out set (`test_motion_data.csv`, session-disjoint from train, all 3
classes represented in a roughly balanced 40/49/63 split) — it is not
invalidated by this report. What this CV shows is that **that specific
number benefits from evaluation on a class-balanced, multi-class test set**,
a property this dataset happens to have for the train/test file split but
does not have for any single-session holdout. With only 3–4 independent
sessions total, there is no way to get a session-disjoint, class-balanced
holdout other than the one the dataset ships with — so this project cannot
produce a "proper" multi-fold macro-F1 CV estimate at all, only this
degenerate single-class-per-fold version. That limitation is itself the
finding: **the 0.49 macro F1 headline number describes performance on the
one test split this dataset provides, and this repo does not have enough
independent sessions to say anything about how it would vary on a
different one.**

## Bottom line for the README and elsewhere

- **Silhouette-selected k is stable across held-out sessions (k=2 in 4/4
  folds).** This is the one LOSO result that can be reported as straightforward
  supporting evidence, and it is: k=2 does not depend on which session is
  excluded.
- **ARI is directionally consistent (pooled LOSO 0.16 vs. fixed-split
  0.12)** but per-fold ARI cannot be usefully summarized as a mean ± spread
  in this dataset, for the structural reason above — that should be stated
  plainly rather than reported as "0.00 ± 0.00," which would read as "the
  clustering doesn't work at all on held-out sessions" and is not what the
  number means.
- **Macro F1 mean ± std across folds (0.04 ± 0.06) should not be quoted
  next to the 0.49 fixed-split number without the explanation above** — they
  are not measuring the same thing, and juxtaposing them without context
  would understate the fixed-split result at least as much as reporting
  0.49 alone overstates the model's demonstrated robustness.
