from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA PHM score; positive lateness (overestimated RUL) is penalized more."""
    error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    terms = np.where(error < 0, np.expm1(-error / 13.0), np.expm1(error / 10.0))
    return float(terms.sum())


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "nasa_score": nasa_score(y_true, y_pred),
    }


def conformal_radius(y_true: np.ndarray, y_pred: np.ndarray, coverage: float = 0.9) -> float:
    """Finite-sample absolute-residual quantile for a split-conformal interval."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    residuals = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    n = len(residuals)
    if n == 0:
        raise ValueError("calibration set is empty")
    rank = min(n, int(np.ceil((n + 1) * coverage)))
    return float(np.partition(residuals, rank - 1)[rank - 1])
