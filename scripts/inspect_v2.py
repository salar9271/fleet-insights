"""Step 1 (v2 dataset): read-only inspection of data/driving_behavior_v2/.

Note: the folder actually lives at data/driving_behavior_v2/, not
data/raw/driving_behavior_v2/ as mentioned in the task -- this script points
at the real location. Prints diagnostics only; does not write anything.
"""

import glob
import os

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

DATA_DIR = "data/driving_behavior_v2"

TIMESTAMP_KEYWORDS = ("time", "timestamp", "date")
TRIP_KEYWORDS = ("trip", "session")
DRIVER_KEYWORDS = ("driver",)
CAR_KEYWORDS = ("car", "vehicle")
CLASS_KEYWORDS = ("class", "target", "label")


def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def find_columns(columns, keywords):
    return [c for c in columns if any(k in c.lower() for k in keywords)]


def find_class_col(columns):
    for c in columns:
        if c.lower() in CLASS_KEYWORDS or any(k in c.lower() for k in CLASS_KEYWORDS):
            return c
    return None


def inspect_file(path):
    section(f"FILE: {path}")
    df = pd.read_csv(path)

    print(f"\n--- shape ---\n{df.shape}")
    print("\n--- dtypes ---")
    print(df.dtypes)
    print("\n--- exact column names ---")
    print(list(df.columns))
    print("\n--- head(5) ---")
    print(df.head(5))

    class_col = find_class_col(df.columns)
    section("Class distribution")
    if class_col is None:
        print("No class/target/label-like column found.")
    else:
        print(f"class column used: {class_col!r}")
        counts = df[class_col].value_counts(dropna=False)
        print(counts)
        print("\nproportions:")
        print((counts / len(df)).round(4))

    section("ID columns present?")
    trip_cols = find_columns(df.columns, TRIP_KEYWORDS)
    driver_cols = find_columns(df.columns, DRIVER_KEYWORDS)
    car_cols = find_columns(df.columns, CAR_KEYWORDS)
    print(f"trip/session-id-like columns: {trip_cols if trip_cols else 'NONE FOUND'}")
    print(f"driver-id-like columns:       {driver_cols if driver_cols else 'NONE FOUND'}")
    print(f"car-id-like columns:          {car_cols if car_cols else 'NONE FOUND'}")

    section("Timestamp column?")
    ts_cols = find_columns(df.columns, TIMESTAMP_KEYWORDS)
    ts_col = None
    if not ts_cols:
        print("NONE FOUND")
    else:
        ts_col = ts_cols[0]
        print(f"timestamp column: {ts_col!r} (raw dtype: {df[ts_col].dtype})")
        ts_numeric = pd.to_numeric(df[ts_col], errors="coerce")
        n_unparsed = ts_numeric.isna().sum() - df[ts_col].isna().sum()
        if n_unparsed:
            print(f"  WARNING: {n_unparsed} values failed to parse as numeric")

        is_monotonic = ts_numeric.is_monotonic_increasing
        is_monotonic_nondecreasing = (ts_numeric.diff().dropna() >= 0).all()
        print(f"  strictly monotonically increasing: {is_monotonic}")
        print(f"  monotonically non-decreasing (ties allowed): {is_monotonic_nondecreasing}")

        gaps = ts_numeric.diff().dropna()
        if not gaps.empty:
            median_gap = gaps.median()
            min_gap = gaps.min()
            max_gap = gaps.max()
            print(f"  median gap: {median_gap}")
            print(f"  min gap:    {min_gap}")
            print(f"  max gap:    {max_gap}")
            if pd.notna(median_gap) and median_gap > 0:
                threshold = median_gap * 5
                n_large = int((gaps > threshold).sum())
                print(f"  threshold (5x median): {threshold}")
                print(f"  count of gaps > 5x median (candidate trip boundaries): {n_large}")
            else:
                print("  median gap is zero/invalid; skipping 5x-median threshold check")
            n_zero_or_neg = int((gaps <= 0).sum())
            print(f"  count of zero/negative gaps (duplicate or out-of-order timestamps): {n_zero_or_neg}")

    section("Sorted by class, or interleaved in time?")
    if class_col is None:
        print("No class column found -> cannot check.")
    else:
        labels = df[class_col].to_numpy()
        change_points = int(np.sum(labels[1:] != labels[:-1]))
        n_unique = df[class_col].nunique(dropna=False)
        print(f"label change-points along row order: {change_points}")
        print(f"unique classes: {n_unique}")
        contiguous = change_points == max(n_unique - 1, 0)
        print(f"labels form contiguous blocks (sorted by class): {contiguous}")
        if ts_col is not None and not contiguous:
            print("-> since not sorted by class and a timestamp exists, rows look interleaved in time.")

    section("NaN counts per column")
    print(df.isna().sum())

    return df, class_col, ts_col


def leakage_check(files_info):
    train_info = next((v for k, v in files_info.items() if "train" in k.lower()), None)
    test_info = next((v for k, v in files_info.items() if "test" in k.lower()), None)
    if train_info is None or test_info is None:
        return

    section("Train/test leakage check")
    train_df, _, train_ts = train_info
    test_df, _, test_ts = test_info

    if train_ts is not None and test_ts is not None:
        train_range = (pd.to_numeric(train_df[train_ts], errors="coerce").min(),
                        pd.to_numeric(train_df[train_ts], errors="coerce").max())
        test_range = (pd.to_numeric(test_df[test_ts], errors="coerce").min(),
                       pd.to_numeric(test_df[test_ts], errors="coerce").max())
        print(f"train timestamp range: {train_range}")
        print(f"test timestamp range:  {test_range}")
        overlap = not (train_range[1] < test_range[0] or test_range[1] < train_range[0])
        print(f"train/test timestamp ranges overlap: {overlap}")

        train_ts_vals = set(pd.to_numeric(train_df[train_ts], errors="coerce").dropna().unique())
        test_ts_vals = set(pd.to_numeric(test_df[test_ts], errors="coerce").dropna().unique())
        shared_ts = train_ts_vals & test_ts_vals
        print(f"exact timestamp values shared between train and test: {len(shared_ts)}")
    else:
        print("No comparable timestamp column found in both files -> cannot check time-range overlap.")

    trip_cols_train = find_columns(train_df.columns, TRIP_KEYWORDS)
    trip_cols_test = find_columns(test_df.columns, TRIP_KEYWORDS)
    if trip_cols_train and trip_cols_test:
        common = set(trip_cols_train) & set(trip_cols_test)
        for col in common:
            shared_trips = set(train_df[col].unique()) & set(test_df[col].unique())
            print(f"trip/session id column {col!r}: {len(shared_trips)} ids shared between train and test")
    else:
        print("No trip/session-id column present in both files -> cannot check trip-level leakage directly.")

    exact_dupe_rows = pd.merge(train_df, test_df, how="inner")
    print(f"exact duplicate rows (all columns identical) between train and test: {len(exact_dupe_rows)}")


def main():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not paths:
        print(f"No CSV files found under {DATA_DIR}/")
        return

    files_info = {}
    for path in paths:
        df, class_col, ts_col = inspect_file(path)
        files_info[os.path.basename(path)] = (df, class_col, ts_col)

    leakage_check(files_info)


if __name__ == "__main__":
    main()
