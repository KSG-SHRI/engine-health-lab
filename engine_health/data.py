from __future__ import annotations

from pathlib import Path

import pandas as pd

COLS = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [
    f"sensor_{i}" for i in range(1, 22)
]


def read_trajectory_file(path: str | Path) -> pd.DataFrame:
    """Read one whitespace-delimited NASA C-MAPSS trajectory file."""
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    frame = frame.dropna(axis=1, how="all")
    if frame.shape[1] != len(COLS):
        raise ValueError(f"Expected {len(COLS)} columns, found {frame.shape[1]} in {path}")
    frame.columns = COLS
    return frame.astype({"unit": "int64", "cycle": "int64"})


def add_capped_rul(frame: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """Add the conventional capped RUL target for run-to-failure training rows."""
    if cap <= 0:
        raise ValueError("cap must be positive")
    out = frame.copy()
    last = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = (last - out["cycle"]).clip(upper=cap)
    return out


def attach_test_rul(test: pd.DataFrame, truth_path: str | Path, cap: int = 125) -> pd.DataFrame:
    """Attach official end-of-test RUL targets to each test engine's last row."""
    truth = pd.read_csv(truth_path, sep=r"\s+", header=None).iloc[:, 0].astype(int)
    last_rows = test.sort_values(["unit", "cycle"]).groupby("unit", as_index=False).tail(1).copy()
    units = sorted(last_rows["unit"].unique())
    if len(units) != len(truth):
        raise ValueError(f"{len(units)} test units but {len(truth)} RUL targets")
    last_rows["rul"] = [min(int(v), cap) for v in truth]
    return last_rows.sort_values("unit").reset_index(drop=True)
