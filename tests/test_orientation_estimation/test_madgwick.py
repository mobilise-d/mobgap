import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from mobgap.consts import GRAV_MS2, SF_ACC_COLS, SF_SENSOR_COLS
from mobgap.orientation_estimation import MadgwickAHRS


class TestMadgwickAHRS:
    def test_initial_orientation_none_aligns_first_acc_sample_to_global_z(self):
        data = _empty_sensor_frame_data()
        data["acc_y"] = GRAV_MS2

        output = MadgwickAHRS(beta=0, initial_orientation=None).estimate(data, sampling_rate_hz=100.0)

        rotated_first_acc_sample = output.orientation_object_[0].apply(data[SF_ACC_COLS].iloc[0].to_numpy(copy=True))
        assert_allclose(rotated_first_acc_sample, np.array([0.0, 0.0, GRAV_MS2]), atol=1e-12)
        assert len(output.orientation_object_) == len(data) + 1

    def test_initial_orientation_none_keeps_rotated_data_sample_aligned(self):
        data = _empty_sensor_frame_data()
        data["acc_x"] = GRAV_MS2

        output = MadgwickAHRS(beta=0, initial_orientation=None).estimate(data, sampling_rate_hz=100.0)

        assert len(output.rotated_data_) == len(data)
        assert_allclose(output.rotated_data_["acc_gz"], np.full(len(data), GRAV_MS2), atol=1e-12)

    def test_initial_orientation_none_falls_back_to_identity_for_zero_acc_sample(self):
        data = _empty_sensor_frame_data()

        output = MadgwickAHRS(beta=0, initial_orientation=None).estimate(data, sampling_rate_hz=100.0)

        assert_allclose(output.orientation_object_[0].as_quat(), np.array([0.0, 0.0, 0.0, 1.0]))


def _empty_sensor_frame_data(n_samples: int = 10) -> pd.DataFrame:
    return pd.DataFrame(np.zeros((n_samples, len(SF_SENSOR_COLS))), columns=SF_SENSOR_COLS)
