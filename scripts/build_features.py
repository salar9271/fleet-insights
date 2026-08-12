"""Run window-based feature extraction on data/driving_behavior_v2/ and
report window counts and feature matrix shape. Read-only on data/raw and
data/driving_behavior_v2; writes only to data/processed/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import build_features


def report(name, df):
    print(f"\n=== {name} ===")
    print(f"feature matrix shape: {df.shape}")
    print(f"window count: {len(df)}")
    print("\nwindows per class:")
    print(df["class"].value_counts())
    print("\nwindows per session:")
    print(df["session_id"].value_counts().sort_index())
    print("\nestimated sampling rate per session (Hz, assumes Timestamp is in seconds -- see docs/data_selection.md):")
    print(df.groupby("session_id")["session_fs_hz"].first())


def main():
    train_windows, test_windows = build_features()
    report("train", train_windows)
    report("test", test_windows)


if __name__ == "__main__":
    main()
