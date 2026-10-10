from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SENSORS = [f"sensor_{i}" for i in range(1, 22)]
SETTINGS = ["setting_1", "setting_2", "setting_3"]


def _summarize_window(unit: int, tail: pd.DataFrame) -> dict[str, float | int]:
    """Summarize a single engine prefix without using future rows."""
    row: dict[str, float | int] = {"unit": int(unit), "cycle": int(tail["cycle"].max())}
    for col in SETTINGS:
        row[f"{col}_mean"] = float(tail[col].mean())
    for col in SENSORS:
        row[f"{col}_mean"] = float(tail[col].mean())
        row[f"{col}_std"] = float(tail[col].std(ddof=0))
        row[f"{col}_slope"] = float(np.polyfit(np.arange(len(tail)), tail[col], 1)[0]) if len(tail) > 1 else 0.0
    return row


def last_window_features(frame: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Create one final-prefix feature row per engine."""
    if window < 1:
        raise ValueError("window must be at least 1")
    rows = []
    for unit, group in frame.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
        tail = group.tail(window)
        rows.append(_summarize_window(unit, tail))
    return pd.DataFrame(rows)


def rolling_training_examples(frame: pd.DataFrame, window: int = 30, stride: int = 5, cap: int = 125) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate past-only trailing windows and their as-of RUL labels."""
    rows, targets = [], []
    for unit, group in frame.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
        group = group.reset_index(drop=True)
        valid = np.flatnonzero(group["rul"].to_numpy() <= cap)
        picks = valid[::stride]
        if len(valid) and (not len(picks) or picks[-1] != valid[-1]):
            picks = np.append(picks, valid[-1])
        for end in picks:
            start = max(0, end - window + 1)
            rows.append(_summarize_window(int(unit), group.iloc[start:end + 1]))
            targets.append(float(group.loc[end, "rul"]))
    return pd.DataFrame(rows), np.asarray(targets)


def calibration_examples(frame: pd.DataFrame, window: int = 30, cap: int = 125, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Choose one seeded as-of observation per held-out engine for calibration."""
    rng = np.random.default_rng(seed)
    rows, targets = [], []
    for unit, group in frame.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
        group = group.reset_index(drop=True)
        valid = np.flatnonzero(group["rul"].to_numpy() <= cap)
        if not len(valid):
            continue
        end = int(rng.choice(valid))
        start = max(0, end - window + 1)
        rows.append(_summarize_window(int(unit), group.iloc[start:end + 1]))
        targets.append(float(group.loc[end, "rul"]))
    return pd.DataFrame(rows), np.asarray(targets)


class OperatingRegimes:
    """Train-only scaling and clustering of the three operating settings."""

    def __init__(self, clusters: int = 6, seed: int = 42):
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=clusters, random_state=seed, n_init=20)

    def fit(self, train: pd.DataFrame) -> "OperatingRegimes":
        self.scaler.fit(train[SETTINGS])
        self.model.fit(self.scaler.transform(train[SETTINGS]))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.scaler.transform(frame[SETTINGS]))
