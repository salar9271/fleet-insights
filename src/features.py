"""Window-based feature extraction for data/driving_behavior_v2/.

Train and test CSVs are processed independently and never combined. Each file
is first split into sessions using large timestamp gaps (and class changes,
as a belt-and-suspenders check) so that no window ever crosses a session or
class boundary -- see docs/data_selection.md for why sessions, not trips, are
the largest independent unit available in this dataset.

Timestamp unit assumption: the Timestamp column is a relative device counter,
not wall-clock time (see docs/data_selection.md). Treating it as seconds
yields ~1.7 samples/unit in both files independently, which lines up with
this dataset family's documented ~2 Hz sampling rate -- so it is treated as
seconds for the sampling-rate estimate used by the spectral feature below.
The estimated fs is per-session (not a single global assumption) and is
attached to every window as `session_fs_hz` so the assumption stays visible
downstream instead of being buried in a constant.

At ~1.7 Hz the Nyquist frequency is ~0.85 Hz, so the spectral band is set to
0.2-0.8 Hz (not the initially-requested 0.5-2 Hz, which mostly exceeded
Nyquist and would have measured little more than the fixed ceiling imposed
by the data rate rather than anything about the signal). 0.2-0.8 Hz stays
safely under Nyquist at the estimated rate -- see docs/data_selection.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

TRAIN_PATH = "data/driving_behavior_v2/train_motion_data.csv"
TEST_PATH = "data/driving_behavior_v2/test_motion_data.csv"
OUT_DIR = "data/processed"

DEFAULT_WINDOW_LENGTH = 20
GAP_MULTIPLIER = 5.0
ACC_Y_THRESHOLD = 3.0
SPECTRAL_BAND = (0.2, 0.8)
FALLBACK_FS_HZ = 2.0


def load_raw(path):
    df = pd.read_csv(path)
    return df.reset_index(drop=True)


def segment_sessions(df, gap_multiplier=GAP_MULTIPLIER):
    """Assign a local integer session_id, cutting at large timestamp gaps
    and at any class change (windows must never cross either)."""
    ts = df["Timestamp"].to_numpy()
    cls = df["Class"].to_numpy()
    diffs = np.diff(ts)
    median_gap = np.median(diffs)
    threshold = median_gap * gap_multiplier if median_gap > 0 else 0

    session_id = np.zeros(len(df), dtype=int)
    current = 0
    for i in range(1, len(df)):
        large_gap = diffs[i - 1] > threshold
        class_change = cls[i] != cls[i - 1]
        if large_gap or class_change:
            current += 1
        session_id[i] = current

    df = df.copy()
    df["session_id"] = session_id
    return df


def estimate_session_fs(timestamps):
    """Samples/second for a session, assuming Timestamp is in seconds.
    Falls back to FALLBACK_FS_HZ if the session has a degenerate span."""
    if len(timestamps) < 2:
        return FALLBACK_FS_HZ
    span = timestamps[-1] - timestamps[0]
    if span <= 0:
        return FALLBACK_FS_HZ
    return (len(timestamps) - 1) / span


def spectral_energy_ratio(signal, fs, band=SPECTRAL_BAND):
    freqs, psd = welch(signal, fs=fs, nperseg=len(signal))
    total_energy = psd.sum()
    if total_energy <= 0:
        return 0.0
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    return psd[band_mask].sum() / total_energy


def compute_window_features(chunk, session_id, fs):
    acc = chunk[["AccX", "AccY", "AccZ"]].to_numpy()
    acc_mag = np.linalg.norm(acc, axis=1)
    accX = chunk["AccX"].to_numpy()
    accY = chunk["AccY"].to_numpy()
    gyroX = chunk["GyroX"].to_numpy()
    gyroY = chunk["GyroY"].to_numpy()
    gyroZ = chunk["GyroZ"].to_numpy()

    jerk = np.diff(acc_mag)
    majority_class = chunk["Class"].mode().iloc[0]

    return {
        "session_id": session_id,
        "class": majority_class,
        "session_fs_hz": fs,
        "acc_mag_mean": acc_mag.mean(),
        "acc_mag_std": acc_mag.std(),
        "acc_mag_max": acc_mag.max(),
        "acc_mag_p95": np.percentile(acc_mag, 95),
        "accY_mean": accY.mean(),
        "accY_std": accY.std(),
        "accY_rate_below_neg3": float(np.mean(accY < -ACC_Y_THRESHOLD)),
        "accY_rate_above_pos3": float(np.mean(accY > ACC_Y_THRESHOLD)),
        "accX_std": accX.std(),
        "accX_max_abs": np.abs(accX).max(),
        "jerk_std": jerk.std(),
        "jerk_rms": np.sqrt(np.mean(jerk**2)),
        "gyroX_std": gyroX.std(),
        "gyroY_std": gyroY.std(),
        "gyroZ_std": gyroZ.std(),
        "spectral_energy_ratio_0.2_0.8hz": spectral_energy_ratio(acc_mag, fs),
    }


def make_windows(df, split_name, window_length=DEFAULT_WINDOW_LENGTH,
                  gap_multiplier=GAP_MULTIPLIER):
    df = segment_sessions(df, gap_multiplier=gap_multiplier)

    rows = []
    for local_session_id, session_df in df.groupby("session_id", sort=True):
        session_df = session_df.reset_index(drop=True)
        session_id = f"{split_name}-{local_session_id}"
        fs = estimate_session_fs(session_df["Timestamp"].to_numpy())

        n_windows = len(session_df) // window_length
        for w in range(n_windows):
            start = w * window_length
            end = start + window_length
            chunk = session_df.iloc[start:end]
            rows.append(compute_window_features(chunk, session_id, fs))

    return pd.DataFrame(rows)


def build_features(train_path=TRAIN_PATH, test_path=TEST_PATH, out_dir=OUT_DIR,
                    window_length=DEFAULT_WINDOW_LENGTH, gap_multiplier=GAP_MULTIPLIER):
    train_raw = load_raw(train_path)
    test_raw = load_raw(test_path)

    train_windows = make_windows(train_raw, "train", window_length, gap_multiplier)
    test_windows = make_windows(test_raw, "test", window_length, gap_multiplier)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_windows.to_parquet(out_dir / "windows_train.parquet", index=False)
    test_windows.to_parquet(out_dir / "windows_test.parquet", index=False)

    return train_windows, test_windows
