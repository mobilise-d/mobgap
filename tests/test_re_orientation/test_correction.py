import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy.spatial.transform import Rotation
from tpcp import make_action_safe

from mobgap.consts import SF_SENSOR_COLS
from mobgap.re_orientation import ReorientationMethodDM, get_orientation_for_sequences
from mobgap.re_orientation.base import BaseReorientationCorrector
from mobgap.utils.conversions import to_body_frame


def test_dm_application_replays_gravity_then_pa_with_known_axis_values():
    data = pd.DataFrame([[1, 2, 3, 4, 5, 6]], columns=SF_SENSOR_COLS, index=pd.Index([42], name="sample"))
    original = data.copy(deep=True)
    correction = (Rotation.from_euler("z", -90, degrees=True), Rotation.from_euler("x", 180, degrees=True))
    expected = pd.DataFrame(
        [[2, 1, -3, 5, 4, -6]],
        columns=["acc_is", "acc_ml", "acc_pa", "gyr_is", "gyr_ml", "gyr_pa"],
        index=data.index,
    )
    assert_frame_equal(ReorientationMethodDM.apply_correction(data, correction), expected)
    assert_frame_equal(data, original)


class _SignedCorrector(BaseReorientationCorrector[dict[str, int]]):
    def __init__(self, ml_sign=1):
        self.ml_sign = ml_sign

    @staticmethod
    def apply_correction(data, correction):
        result = to_body_frame(data)
        result[["acc_ml", "gyr_ml"]] *= correction["ml_sign"]
        return result

    @make_action_safe
    def detect_correct(self, data, *, sampling_rate_hz, **kwargs):
        self.correction_ = {"ml_sign": self.ml_sign}
        self.correction_applied_ = self.ml_sign != 1
        self.corrected_data_ = self.apply_correction(data, self.correction_)
        return self


def test_alternate_correction_type_survives_selection_and_tpcp_clone():
    data = pd.DataFrame([[1, 2, 3, 4, 5, 6]], columns=SF_SENSOR_COLS)
    algo = _SignedCorrector(ml_sign=-1).clone().detect_correct(data, sampling_rate_hz=100)
    sections = pd.DataFrame(
        {"start": [0], "end": [10], "orientation": [algo.correction_], "correction_applied": [True]}
    )
    sequences = pd.DataFrame({"start": [1, 10], "end": [2, 12]})
    default = {"ml_sign": 1}
    corrections = get_orientation_for_sequences(sequences, sections, default=default)
    assert corrections.iloc[0] is algo.correction_
    assert corrections.iloc[1] is default
    expected = pd.DataFrame(
        [[1, -2, 3, 4, -5, 6]], columns=["acc_is", "acc_ml", "acc_pa", "gyr_is", "gyr_ml", "gyr_pa"]
    )
    assert_frame_equal(_SignedCorrector.apply_correction(data, corrections.iloc[0]), expected)
    assert_frame_equal(algo.corrected_data_, expected)
    assert_frame_equal(_SignedCorrector.apply_correction(data, corrections.iloc[1]), to_body_frame(data))


@pytest.mark.parametrize("gravity_angle", [0, 90, -90, 180])
@pytest.mark.parametrize("pa_angle", [0, 180])
@pytest.mark.parametrize("mode", ["full", "trust_gravity"])
def test_dm_replay_matches_detection_for_all_mounting_families(gravity_angle, pa_angle, mode):
    from mobgap.utils.rotations import flip_dataset

    time = np.arange(1000) / 100
    nominal = pd.DataFrame(
        {
            "acc_x": 9.81 + np.sin(2 * np.pi * 1.2 * time),
            "acc_y": np.zeros(1000),
            "acc_z": np.cos(2 * np.pi * 1.2 * time),
            "gyr_x": np.ones(1000),
            "gyr_y": np.ones(1000) * 2,
            "gyr_z": np.ones(1000) * 3,
        },
        index=pd.Index(np.arange(1000) + 100, name="sample"),
    )
    raw = flip_dataset(nominal, Rotation.from_euler("x", -pa_angle, degrees=True))
    raw = flip_dataset(raw, Rotation.from_euler("z", -gravity_angle, degrees=True))
    original = raw.copy(deep=True)
    algo = ReorientationMethodDM(correction_mode=mode).detect_correct(raw, sampling_rate_hz=100)
    replayed = ReorientationMethodDM.apply_correction(raw, algo.result_.correction_rotations)
    assert_frame_equal(replayed, algo.corrected_data_)
    assert_frame_equal(raw, original)
    if mode == "full":
        assert_frame_equal(replayed, to_body_frame(nominal))


@pytest.mark.parametrize("gravity", [0.0, -9.8])
def test_dm_replay_preserves_unresolved_fallback(gravity):
    data = pd.DataFrame([[gravity, 0, 0, 1, 2, 3]] * 10, columns=SF_SENSOR_COLS)
    algo = ReorientationMethodDM(
        correction_mode="full", gravity_detection_error_type="ignore", pa_direction_detection_error_type="ignore"
    ).detect_correct(data, sampling_rate_hz=100)
    assert algo.result_.orientation_resolved is False
    assert algo.result_.unresolved_reason == ("gravity" if gravity == 0 else "pa_direction")
    assert_frame_equal(
        ReorientationMethodDM.apply_correction(data, algo.result_.correction_rotations), algo.corrected_data_
    )


def test_identity_application_returns_independent_body_frame_data():
    data = pd.DataFrame([[1, 2, 3, 4, 5, 6]], columns=SF_SENSOR_COLS)
    result = ReorientationMethodDM.apply_correction(data, (Rotation.identity(), Rotation.identity()))
    assert_frame_equal(result, to_body_frame(data))
    result.iloc[0, 0] = 999
    assert data.iloc[0, 0] == 1
