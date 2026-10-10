from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from .data import add_capped_rul, attach_test_rul, read_trajectory_file
from .features import OperatingRegimes, calibration_examples, last_window_features, rolling_training_examples
from .metrics import conformal_radius, regression_metrics


def run(subset: str, root: Path, seed: int = 42) -> dict:
    data_dir = root / "data" / "cmapss"
    train_path, test_path, rul_path = (data_dir / f"{name}_{subset}.txt" for name in ("train", "test", "RUL"))
    train = add_capped_rul(read_trajectory_file(train_path))
    official_test = attach_test_rul(read_trajectory_file(test_path), rul_path)

    # Split whole engines before feature/target-derived transforms are fitted.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    fit_idx, cal_idx = next(splitter.split(train, groups=train["unit"]))
    fit_rows, cal_rows = train.iloc[fit_idx], train.iloc[cal_idx]
    fit, y_fit = rolling_training_examples(fit_rows)
    cal, y_cal = calibration_examples(cal_rows, seed=seed)
    test = last_window_features(official_test)
    y_test = official_test.sort_values("unit")["rul"].to_numpy()

    feature_cols = [c for c in fit.columns if c not in {"unit", "cycle"}]
    results = {}
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.7, random_state=seed, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, l2_regularization=1.0, random_state=seed),
    }
    for name, model in models.items():
        model.fit(fit[feature_cols], y_fit)
        cal_pred = model.predict(cal[feature_cols])
        pred = model.predict(test[feature_cols])
        radius = conformal_radius(y_cal, cal_pred, coverage=0.9)
        stats = regression_metrics(y_test, pred)
        stats.update({
            "calibration_engines": int(len(y_cal)),
            "nominal_interval_coverage": 0.9,
            "calibrated_radius_cycles": radius,
            "test_interval_coverage": float(np.mean((y_test >= pred - radius) & (y_test <= pred + radius))),
            "test_interval_width_cycles": float(2 * radius),
        })
        results[name] = stats

    regimes = OperatingRegimes(clusters=6, seed=seed).fit(fit_rows)
    test_last = official_test.sort_values("unit").copy()
    test_last["regime"] = regimes.predict(test_last)
    best = min(results, key=lambda k: results[k]["rmse"])
    return {
        "dataset": subset,
        "seed": seed,
        "train_engines": int(train["unit"].nunique()),
        "calibration_engines": int(cal["unit"].nunique()),
        "test_engines": int(len(official_test)),
        "target_cap_cycles": 125,
        "models": results,
        "best_by_rmse": best,
        "best_by_rmse_by_regime": {
            str(int(regime)): regression_metrics(
                y_test[test_last["regime"].to_numpy() == regime],
                models[best].predict(test.loc[test_last["regime"].to_numpy() == regime, feature_cols]),
            )
            for regime in sorted(test_last["regime"].unique())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["FD001", "FD002", "FD003", "FD004"], default="FD002")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run(args.subset, root, args.seed)
    output = root / "results" / f"{args.subset.lower()}-seed-{args.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
