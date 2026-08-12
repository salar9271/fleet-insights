"""Step 1: read-only inspection of the raw sensor data and the pre-computed
feature baselines. Prints diagnostics only; does not write anything."""

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

RAW_PATH = "data/raw/sensor_raw.csv"
FEATURE_PATHS = [
    "data/raw/features_14.csv",
    "data/raw/sero_features_10.csv",
    "data/raw/sero_features_20.csv",
]

TIMESTAMP_KEYWORDS = ("time", "timestamp", "date")
TRIP_KEYWORDS = ("trip",)
DRIVER_KEYWORDS = ("driver",)
CAR_KEYWORDS = ("car", "vehicle")


def find_columns(columns, keywords):
    return [c for c in columns if any(k in c.lower() for k in keywords)]


def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def inspect_raw():
    section(f"RAW: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)

    print(f"\n--- shape ---\n{df.shape}")

    print("\n--- dtypes ---")
    print(df.dtypes)

    print("\n--- exact column names ---")
    print(list(df.columns))

    print("\n--- head(10) ---")
    print(df.head(10))

    class_col = None
    for candidate in df.columns:
        if candidate.lower() in ("target(class)", "target", "class", "label"):
            class_col = candidate
            break
    if class_col is None:
        class_col = df.columns[0]

    section("Class distribution")
    print(f"class column used: {class_col!r}")
    counts = df[class_col].value_counts(dropna=False).sort_index()
    print(counts)
    print("\nproportions:")
    print((counts / len(df)).round(4))

    section("Are labels sorted into contiguous blocks?")
    labels = df[class_col].to_numpy()
    change_points = int(np.sum(labels[1:] != labels[:-1]))
    n_unique = df[class_col].nunique(dropna=False)
    print(f"number of label change-points along row order: {change_points}")
    print(f"number of unique classes: {n_unique}")
    print(
        "contiguous blocks (labels only ever change at most once per class): "
        f"{change_points == n_unique - 1 if n_unique > 0 else 'n/a'}"
    )
    print(
        "(if change_points == n_unique - 1, each class occupies exactly one "
        "contiguous run; a larger count means classes are interleaved)"
    )

    section("ID / timestamp columns present?")
    ts_cols = find_columns(df.columns, TIMESTAMP_KEYWORDS)
    trip_cols = find_columns(df.columns, TRIP_KEYWORDS)
    driver_cols = find_columns(df.columns, DRIVER_KEYWORDS)
    car_cols = find_columns(df.columns, CAR_KEYWORDS)
    print(f"timestamp-like columns: {ts_cols if ts_cols else 'NONE FOUND'}")
    print(f"trip-id-like columns:   {trip_cols if trip_cols else 'NONE FOUND'}")
    print(f"driver-id-like columns: {driver_cols if driver_cols else 'NONE FOUND'}")
    print(f"car-id-like columns:    {car_cols if car_cols else 'NONE FOUND'}")

    if ts_cols:
        section("Timestamp gap analysis")
        for ts_col in ts_cols:
            print(f"\n-- column: {ts_col!r} --")
            ts = pd.to_datetime(df[ts_col], errors="coerce")
            n_unparsed = ts.isna().sum() - df[ts_col].isna().sum()
            if n_unparsed:
                print(f"  WARNING: {n_unparsed} values failed to parse as datetime")
            gaps = ts.diff().dropna()
            if gaps.empty:
                print("  not enough valid timestamps to compute gaps")
                continue
            median_gap = gaps.median()
            min_gap = gaps.min()
            max_gap = gaps.max()
            print(f"  median gap: {median_gap}")
            print(f"  min gap:    {min_gap}")
            print(f"  max gap:    {max_gap}")
            if pd.notna(median_gap) and median_gap > pd.Timedelta(0):
                threshold = median_gap * 5
                n_large = int((gaps > threshold).sum())
                print(f"  threshold (5x median): {threshold}")
                print(f"  count of gaps > 5x median (candidate trip boundaries): {n_large}")
            else:
                print("  median gap is zero/invalid; skipping 5x-median threshold check")
    else:
        print("\nNo timestamp column found -> skipping sampling-gap / trip-boundary analysis.")

    section("NaN counts per column")
    print(df.isna().sum())

    section("describe() for numeric sensor columns")
    numeric_cols = [c for c in df.columns if c != class_col and pd.api.types.is_numeric_dtype(df[c])]
    print(f"numeric sensor columns: {numeric_cols}")
    print(df[numeric_cols].describe())


def inspect_feature_file(path):
    section(f"FEATURES: {path}")
    df = pd.read_csv(path)
    print(f"\n--- shape ---\n{df.shape}")

    class_col = None
    for candidate in df.columns:
        if candidate.lower() in ("target(class)", "target", "class", "label"):
            class_col = candidate
            break
    if class_col is None:
        class_col = df.columns[0]

    print(f"\nclass column used: {class_col!r}")
    counts = df[class_col].value_counts(dropna=False).sort_index()
    print("\n--- class distribution ---")
    print(counts)
    print("\nproportions:")
    print((counts / len(df)).round(4))


def main():
    inspect_raw()
    for path in FEATURE_PATHS:
        inspect_feature_file(path)


if __name__ == "__main__":
    main()
