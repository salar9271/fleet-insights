# Data selection

## Decision

The project proceeds with `data/driving_behavior_v2/` (`train_motion_data.csv`,
`test_motion_data.csv` — Kaggle `outofskills/driving-behavior`) as the primary
dataset. `data/raw/sensor_raw.csv` is dropped.

Reasons: `driving_behavior_v2` has real per-row timestamps (`sensor_raw.csv`
had none), which allows correct time-ordering and genuine sampling-gap
detection for windowing. It also ships with a pre-existing train/test file
split that appears to be session-level (disjoint timestamp ranges, zero exact
duplicate rows between files) rather than a shuffled row split.

`data/raw/features_14.csv`, `sero_features_10.csv`, and `sero_features_20.csv`
remain out of scope — they were pre-computed baselines for `sensor_raw.csv`
and are not used going forward.

## Limitations

- **No trip, driver, or vehicle identifier column** in either file. There is
  no way to attribute rows to a specific trip, driver, or car — the same gap
  that existed in `sensor_raw.csv`.
- **One continuous recording session per class, per file.** Row order is
  sorted by class, and the large timestamp gaps (>5x the median gap) line up
  exactly with the class transitions, confirming each class block is a single
  continuous drive rather than multiple concatenated trips (train has one
  additional internal gap inside its third class block, i.e. a recording
  pause mid-session). This means train and test each contribute only 3
  independent sessions total — one per class — not a pool of many trips.
- **Timestamps are relative, not wall-clock.** They look like device-local
  tick counters (train: ~3,581,629–3,583,791; test: ~818,922–820,709), not
  Unix time. Disjoint ranges and zero shared rows argue against direct
  leakage between the provided train/test files, but this is circumstantial —
  there is no shared clock or ID to prove train and test were recorded at
  different times, on different drives, or by different drivers/vehicles.
- **3-class label taxonomy** (`SLOW` / `NORMAL` / `AGGRESSIVE`), not the
  4-class scheme (`1`–`4`) used by `sensor_raw.csv`. Any comparison to prior
  class-based analysis on the old dataset does not carry over.

## Spectral feature band

`Timestamp` has no confirmed real-world unit (see above). Treating it as
seconds gives an empirically estimated sampling rate of ~1.7 Hz, consistent
across both files independently and close to this dataset family's
documented ~2 Hz rate — this is the working assumption used for the
per-session `session_fs_hz` value attached to every feature window.

At ~1.7 Hz the Nyquist frequency is ~0.85 Hz. The spectral energy ratio
feature was initially specified as the 0.5–2 Hz band, but that band mostly
sits above Nyquist at the estimated rate, so it would have measured almost
nothing but the data's fixed frequency ceiling rather than any real signal
content. The band was changed to **0.2–0.8 Hz**, which stays safely under
the ~0.85 Hz Nyquist limit, so the ratio reflects actual low-frequency
energy in the acceleration-magnitude signal instead of being clipped by the
sampling-rate assumption.

## Scope of generalization claims

**The project makes no multi-vehicle or cross-driver generalization claims.**
With no vehicle or driver identifier and only one session per class per file,
results describe behavior within the sessions actually recorded, not
generalization across a fleet of cars or a population of drivers. Any
train/test evaluation is, at best, a single held-out session per class rather
than an aggregate over independent trips.
