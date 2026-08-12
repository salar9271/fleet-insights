# Clustering findings: diagnostics on ARI ~0.12

Context: `scripts/run_clustering.py` fit KMeans (k chosen by silhouette on
train, k=2) on the windowed features in `data/processed/`, scaler and model
fit on train only. External validation against the true 3-class labels gave
Adjusted Rand Index (ARI) 0.1077 on train and **0.1197 on test**. This
document reports two bounded follow-up diagnostics, run by
`scripts/diagnostics.py`, to determine whether that low ARI reflects
uninformative features or a mismatch between the unsupervised geometry and
the label taxonomy.

## Diagnostic A: is the signal there at all?

A RandomForest was trained on the same 16 features from the train windows
and evaluated on the held-out test windows (never seen during training).

- **Macro F1 (test): 0.4900**
- Majority-class baseline (always predict `SLOW`, the majority test class):
  macro F1 = 0.1953

The model scores well above the majority-class baseline, so **the features
are informative** — a supervised model can recover a substantial amount of
class structure from them.

Confusion matrix (test, rows = true class, columns = predicted class):

| true \\ pred | AGGRESSIVE | NORMAL | SLOW |
|---|---|---|---|
| AGGRESSIVE | 28 | 3 | 9 |
| NORMAL | 15 | 12 | 22 |
| SLOW | 9 | 17 | 37 |

`AGGRESSIVE` (28/40 correct) and `SLOW` (37/63 correct) are recovered
reasonably well. `NORMAL` is not (12/49 correct): it is confused with both
neighbors, most often with `SLOW` (22 cases). This is the same pattern that
shows up in the per-feature separation below.

Permutation importance (test set, macro-F1 scoring, 30 repeats), most
important features first:

| feature | importance (mean) | importance (std) |
|---|---|---|
| accY_std | 0.0682 | 0.0269 |
| acc_mag_p95 | 0.0224 | 0.0181 |
| spectral_energy_ratio_0.2_0.8hz | 0.0162 | 0.0120 |
| acc_mag_std | 0.0122 | 0.0125 |
| accY_mean | 0.0056 | 0.0110 |

The remaining 11 features have importances at or below noise level (roughly
±0.01 or smaller, with standard deviations of similar or larger magnitude),
consistent with those features being redundant with the top ones (several
are other summaries of the same acceleration-magnitude signal) rather than
carrying independent information.

**Verdict for Diagnostic A: the features are informative, but the
unsupervised geometry does not align with the label taxonomy** — not "the
features are uninformative." A supervised model can partially recover the
classes from this same feature set; the low ARI is therefore about how
KMeans partitions the space relative to the label boundaries, not about a
lack of signal in the data.

## Diagnostic B: does k=3 tell a different story?

k=2 is the silhouette-selected model (silhouette 0.3731) saved by
`scripts/run_clustering.py` to `models/kmeans.joblib` /
`reports/cluster_profile.csv`, and remains the selected model — nothing
here changes that. k=3 (silhouette 0.1908, the same value already reported
in the original k-sweep) is reported below purely as a labelled-taxonomy
comparison and is saved separately to `models/kmeans_k3.joblib` /
`reports/cluster_profile_k3.csv`, without touching the k=2 artefacts.

- ARI (train, k=3): 0.1002
- **ARI (test, k=3): 0.0951** — lower than k=2's 0.1197, not higher.

Contingency table (test, k=3):

| cluster | AGGRESSIVE | NORMAL | SLOW |
|---|---|---|---|
| 0 (n=70) | 7 | 23 | 40 |
| 1 (n=60) | 18 | 19 | 23 |
| 2 (n=22) | 15 | 7 | 0 |

Cluster profile (train, mean z-score per feature):

| feature | cluster 0 (n=79) | cluster 1 (n=70) | cluster 2 (n=31) |
|---|---|---|---|
| acc_mag_mean | -0.784 | 0.239 | 1.460 |
| acc_mag_std | -0.792 | 0.163 | 1.652 |
| acc_mag_max | -0.791 | 0.218 | 1.524 |
| acc_mag_p95 | -0.787 | 0.144 | 1.680 |
| accY_mean | 0.235 | 0.007 | -0.615 |
| accY_std | -0.577 | 0.179 | 1.067 |
| accY_rate_below_neg3 | -0.256 | -0.125 | 0.935 |
| accY_rate_above_pos3 | -0.192 | -0.007 | 0.505 |
| accX_std | -0.684 | 0.168 | 1.366 |
| accX_max_abs | -0.694 | 0.122 | 1.494 |
| jerk_std | -0.736 | 0.170 | 1.490 |
| jerk_rms | -0.736 | 0.171 | 1.491 |
| gyroX_std | -0.458 | 0.183 | 0.754 |
| gyroY_std | -0.443 | 0.154 | 0.782 |
| gyroZ_std | -0.432 | 0.163 | 0.732 |
| spectral_energy_ratio_0.2_0.8hz | 0.260 | -0.126 | -0.378 |

k=3 splits the same single intensity axis that k=2 found into three tiers
(low / medium / high) instead of two — every feature moves monotonically
across clusters 0 → 1 → 2. Cluster 0 (low) is mostly `SLOW`, cluster 2
(high) is mostly `AGGRESSIVE` with zero `SLOW` windows, but cluster 1
(medium, the largest addition over k=2) mixes all three classes almost
evenly (18/19/23). Splitting the intensity axis into a third tier does not
resolve the ambiguity between classes — it relocates it into a mixed middle
cluster, and the resulting agreement with the true labels is slightly worse
than at k=2, not better.

## Per-feature separation by true class

Mean z-score of each feature by true class (train, z-scores from the same
StandardScaler used for clustering):

| feature | AGGRESSIVE | NORMAL | SLOW |
|---|---|---|---|
| acc_mag_mean | 0.841 | -0.245 | -0.485 |
| acc_mag_std | 0.726 | -0.205 | -0.425 |
| acc_mag_max | 0.673 | -0.189 | -0.395 |
| acc_mag_p95 | 0.798 | -0.220 | -0.472 |
| accY_mean | -0.354 | 0.156 | 0.156 |
| accY_std | 0.837 | -0.235 | -0.491 |
| accY_rate_below_neg3 | 0.499 | -0.180 | -0.256 |
| accY_rate_above_pos3 | 0.201 | -0.048 | -0.126 |
| accX_std | 0.707 | -0.313 | -0.309 |
| accX_max_abs | 0.587 | -0.236 | -0.279 |
| jerk_std | 0.570 | -0.159 | -0.336 |
| jerk_rms | 0.570 | -0.159 | -0.335 |
| gyroX_std | 0.104 | 0.043 | -0.128 |
| gyroY_std | 0.306 | -0.070 | -0.195 |
| gyroZ_std | 0.276 | 0.013 | -0.245 |
| spectral_energy_ratio_0.2_0.8hz | -0.264 | 0.079 | 0.150 |

`AGGRESSIVE` sits 0.5–1.2 z away from the other two classes on nearly every
acceleration-magnitude and volatility feature. `NORMAL` and `SLOW`, by
contrast, sit within roughly 0.02–0.25 z of each other on the same
features — an order of magnitude closer together. Gyro features show weak
separation for all three classes. No feature in this set separates `NORMAL`
from `SLOW` with the kind of margin that separates `AGGRESSIVE` from either
of them.

## Conclusion

The signal is real: a supervised model trained on these features clearly
outperforms a majority-class baseline on held-out data, and `AGGRESSIVE`
driving is genuinely distinguishable from the other two classes on nearly
every feature in this set. What the unsupervised clustering finds is a
single acceleration-intensity axis, and that axis cleanly separates
`AGGRESSIVE` from the rest but does not separate `NORMAL` from `SLOW` —
neither at k=2 (silhouette-selected) nor at k=3 (labelled-taxonomy
comparison, which performs slightly worse on ARI despite the extra
cluster). Intensity-based structure does not recover the 3-class taxonomy:
it recovers a 2-way "aggressive vs. not" split at best, and the `NORMAL` /
`SLOW` boundary is not present in this feature set's geometry.
