# Feature redundancy and reduced-feature-set comparison

Context: an audit of this project (see the C1 finding in the review that
prompted this document) flagged two redundancy issues in the original
16-feature set used by `models/kmeans.joblib`: `jerk_std`/`jerk_rms`
correlated at r=1.00, and the four `acc_mag_*` summary statistics correlated
at r=0.79–0.94 with each other. Because KMeans on `StandardScaler` output
treats every feature as an equally-weighted Euclidean axis, several
near-duplicate features pointing the same direction can mechanically bias
the discovered structure toward that direction — the audit's concern was
that the project's headline "single intensity axis" finding might be, at
least in part, an artifact of that redundancy rather than solely a fact
about the driving data.

This report re-runs the exact same procedure `scripts/run_clustering.py`
uses — train-only `StandardScaler`, `KMeans` swept over k=2..6, k selected
by train silhouette, external validation by held-out test ARI — on three
feature sets side by side. **All three are kept; none replaces another.**
`models/kmeans.joblib`, the deployed model, continues to use the original
16-feature set; `scripts/feature_reduction.py` produced the other two
without touching it.

## Full feature correlation matrix

Full matrix: [`reports/feature_correlation.csv`](feature_correlation.csv).
Heatmap: [`reports/figures/feature_correlation_heatmap.png`](figures/feature_correlation_heatmap.png).

Pairs at |r| > 0.75 on the original 16 train features:

| feature A | feature B | r |
|---|---|---|
| jerk_std | jerk_rms | 1.000 |
| acc_mag_std | acc_mag_max | 0.942 |
| acc_mag_std | acc_mag_p95 | 0.933 |
| acc_mag_mean | acc_mag_p95 | 0.880 |
| acc_mag_std | jerk_rms | 0.877 |
| acc_mag_std | jerk_std | 0.875 |
| acc_mag_max | acc_mag_p95 | 0.863 |
| accX_std | accX_max_abs | 0.857 |
| acc_mag_max | jerk_rms | 0.854 |
| acc_mag_max | jerk_std | 0.853 |
| acc_mag_mean | accX_std | 0.825 |
| acc_mag_mean | acc_mag_std | 0.789 |
| acc_mag_p95 | jerk_rms | 0.787 |
| acc_mag_p95 | jerk_std | 0.786 |
| acc_mag_p95 | accX_max_abs | 0.777 |
| acc_mag_mean | acc_mag_max | 0.774 |

`jerk_std`/`jerk_rms` at r=1.000 is not approximate redundancy — with
per-window mean jerk close to zero, RMS and STD of the same signal are
numerically the same feature computed twice. The four `acc_mag_*` stats and
`jerk_std`/`jerk_rms` together form one tightly-correlated block (13 of the
16 pairs above involve only these six features), consistent with all of them
being different summaries of the same underlying acceleration-magnitude
signal.

## Reduced feature set

Dropped, with justification (see code comments in
`scripts/feature_reduction.py` for the exact numbers):

- **`jerk_rms`** — r=1.00 with `jerk_std`.
- **`acc_mag_mean`, `acc_mag_std`, `acc_mag_max`** — kept only `acc_mag_p95`
  from this four-feature family, chosen on RandomForest permutation
  importance (not arbitrarily): `acc_mag_p95` is the highest-importance
  member (0.0213), ahead of `acc_mag_std` (0.0101), with `acc_mag_mean`
  (-0.0139) and `acc_mag_max` (-0.0049) at or below noise level — see
  [`reports/rf_permutation_importance.csv`](rf_permutation_importance.csv).
- **`accX_max_abs`** — r=0.86 with `accX_std`.

11 features remain: `acc_mag_p95`, `accY_mean`, `accY_std`,
`accY_rate_below_neg3`, `accY_rate_above_pos3`, `accX_std`, `jerk_std`,
`gyroX_std`, `gyroY_std`, `gyroZ_std`, `spectral_energy_ratio_0.2_0.8hz`.

A third variant adds `gyro_mag_std` (L2 norm of the three gyro axes, added
to `src/features.py` alongside this analysis — a mount-orientation-invariant
summary of total rotational rate, mirroring how `acc_mag` already summarizes
the three accelerometer axes) to the 11 reduced features, for 12 total.

## Side-by-side comparison

| | **Original (deployed)** | **Reduced** | **Reduced + gyro_mag_std** |
|---|---|---|---|
| Features | 16 | 11 | 12 |
| Silhouette, k=2 | 0.373 | 0.344 | 0.333 |
| Silhouette, k=3 | 0.191 | 0.303 | 0.192 |
| Silhouette, k=4 | 0.181 | 0.163 | 0.296 |
| Silhouette, k=5 | 0.196 | 0.189 | 0.212 |
| Silhouette, k=6 | 0.138 | 0.133 | 0.169 |
| **Chosen k** (max silhouette) | **2** | **2** | **2** |
| k=2 vs. next-best silhouette margin | 0.373 − 0.196 = 0.177 | 0.344 − 0.303 = 0.041 | 0.333 − 0.296 = 0.037 |
| ARI (test, held-out) | 0.1197 | 0.1172 | 0.1089 |
| Cluster sizes (train) | 51 / 129 | 46 / 134 | 48 / 132 |

Full silhouette sweeps, cluster profiles, and contingency tables:
[`reports/cluster_profile_reduced.csv`](cluster_profile_reduced.csv),
[`reports/cluster_profile_reduced_gyro.csv`](cluster_profile_reduced_gyro.csv)
(original: [`reports/cluster_profile.csv`](cluster_profile.csv)). Saved
models: `models/scaler_reduced.joblib` / `models/kmeans_reduced.joblib` and
`models/scaler_reduced_gyro.joblib` / `models/kmeans_reduced_gyro.joblib`
(`models/scaler.joblib` / `models/kmeans.joblib`, the deployed original-set
artifacts, are unchanged).

## Does de-duplication change anything?

**Not the headline conclusion — k=2 still wins, and ARI stays in the same
0.11–0.12 band on all three feature sets.** The audit's specific concern —
that removing the redundant features would flip the winning k, or that ARI
would jump once the intensity axis stopped being double- and triple-counted
— did not happen. That is itself informative: it argues the "single
intensity axis, weakly aligned with the 3-class taxonomy" finding is not
purely a multicollinearity artifact, since it survives removing the
multicollinearity.

**But it does change how decisive that conclusion looks.** On the original
16 features, k=2 beats the next-best k by 0.177 silhouette points — a clear,
unambiguous winner. On the reduced 11 features, that margin drops to 0.041
(k=2 at 0.344 vs. k=3 at 0.303); with `gyro_mag_std` added, it drops further
to 0.037 (k=2 at 0.333 vs. k=4 at 0.296). Removing the six-feature
intensity block's redundant votes turns "k=2 clearly wins" into "k=2 wins by
a margin easily within the kind of run-to-run noise a 180-window, 3-4-session
dataset produces" (see `reports/cross_validation_findings.md` for direct
evidence of that noise). The original 16-feature silhouette sweep
overstated how decisive the k=2 vs. k=3 choice actually is; the reduced-set
sweep is a more honest picture of a close call, not a landslide.

`gyro_mag_std` does not change the qualitative picture either: ARI moves
from 0.1172 (reduced) to 0.1089 (reduced+gyro), inside the same noise band
as every other number in this table, and the chosen k stays 2. It is not a
free win, but it is also not harmful, and it fixes the accelerometer/gyro
asymmetry the audit flagged (accelerometer got a mount-invariant magnitude
feature; gyroscope did not). It has not been added to the deployed
16-feature model.

## What this means for the deployed model

`models/kmeans.joblib` keeps the original 16-feature set. This report is a
diagnostic finding, not a retraining decision: swapping the deployed model
to the reduced set would trade a small, well-quantified redundancy problem
for a change with no measured benefit (ARI does not improve) and a real
cost (breaking the feature contract every existing upload and report
depends on, for a project already flagged for weak generalization evidence —
see `reports/cross_validation_findings.md`). The redundancy is worth having
found and documented; it does not, on its own, justify a model change.
