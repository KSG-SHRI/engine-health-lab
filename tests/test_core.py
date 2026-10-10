import numpy as np
import pandas as pd
import pytest

from engine_health.data import add_capped_rul
from engine_health.features import last_window_features, rolling_training_examples
from engine_health.metrics import conformal_radius, nasa_score


def test_capped_rul_resets_for_each_engine():
    frame = pd.DataFrame({"unit": [1, 1, 2, 2], "cycle": [1, 2, 1, 2]})
    out = add_capped_rul(frame, cap=1)
    assert out["rul"].tolist() == [1, 0, 1, 0]


def test_window_features_emit_one_row_per_engine():
    frame = pd.DataFrame({
        "unit": [1, 1, 2, 2], "cycle": [1, 2, 1, 2],
        "setting_1": [0., 0., 1., 1.], "setting_2": [0., 0., 0., 0.], "setting_3": [0., 0., 0., 0.],
        **{f"sensor_{i}": [0., 1., 2., 3.] for i in range(1, 22)},
    })
    out = last_window_features(frame, window=2)
    assert out["unit"].tolist() == [1, 2]
    assert out.loc[0, "sensor_1_mean"] == pytest.approx(0.5)


def test_rolling_examples_have_as_of_rul_targets():
    frame = pd.DataFrame({
        "unit": [1] * 6, "cycle": list(range(1, 7)), "rul": [5, 4, 3, 2, 1, 0],
        "setting_1": [0.] * 6, "setting_2": [0.] * 6, "setting_3": [0.] * 6,
        **{f"sensor_{i}": list(range(6)) for i in range(1, 22)},
    })
    features, target = rolling_training_examples(frame, window=3, stride=2, cap=5)
    assert features["cycle"].tolist() == [1, 3, 5, 6]
    assert target.tolist() == [5., 3., 1., 0.]


def test_nasa_score_penalizes_overestimation_more():
    under = nasa_score(np.array([10]), np.array([5]))
    over = nasa_score(np.array([10]), np.array([15]))
    assert over > under


def test_conformal_radius_uses_finite_sample_order_statistic():
    assert conformal_radius(np.array([0, 0, 0, 0]), np.array([1, 2, 3, 4]), 0.75) == 4


def test_bad_coverage_is_rejected():
    with pytest.raises(ValueError):
        conformal_radius(np.array([1]), np.array([1]), 1.0)
