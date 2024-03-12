import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin
from sklearn import svm

from gaitlink.lrd import LrdUllrich


class TestMetaLrdUllrich(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = LrdUllrich

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().detect(
            data = pd.DataFrame(np.zeros((100, 6)), columns=["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]),
            ic_list=pd.DataFrame({"ic": [5, 10, 15]}),
            sampling_rate_hz=100.0,
        )

class TestLrdUllrich:
    def test_empty_data(self):
        test_params = LrdUllrich.PredefinedParameters.uniss_unige_all_all
        algo = LrdUllrich(**test_params)
        data = pd.DataFrame([], columns=["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"])
        ic_list = pd.DataFrame({"ic": []})
        with pytest.raises(ValueError):
            algo.detect(
                data = data,
                ic_list = ic_list,
                sampling_rate_hz=100.0,
            )

    def test_empty_ic(self):
        params = LrdUllrich.PredefinedParameters.uniss_unige_all_all
        algo = LrdUllrich(**params)
        data = pd.DataFrame(np.zeros((100, 6)), columns=["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"])
        ic_list = pd.DataFrame({"ic": []})
        with pytest.raises(ValueError):
            algo.detect(
                data = data,
                ic_list = ic_list,
                sampling_rate_hz=100.0,
            )

    def test_detect_with_model_not_fit(self):
        my_paras = {'model__C': 1.0, 'model__gamma': 1.0, 'model__kernel': 'linear'}
        algo = LrdUllrich(model = svm.SVC())
        algo.set_params(**my_paras)
        data = pd.DataFrame(np.zeros((100, 6)), columns=["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"])
        ic_list = pd.DataFrame({"ic": [5, 10, 15]})
        with pytest.raises(RuntimeError):
            algo.detect(data = data,
                        ic_list= ic_list,
                        sampling_rate_hz = 100.0)

    def test_load_invaliad_predetermined_model(self):
        with pytest.raises(AttributeError):
            LrdUllrich.PredefinedParameters.invalid_model

    def test_detect_custom_algo(self):
        my_paras = {'model__C': 1.0, 'model__gamma': 1.0, 'model__kernel': 'linear'}
        algo = LrdUllrich(model = svm.SVC())
        algo.set_params(**my_paras)
        x = np.linspace(0, 4 * np.pi, 100)[:, None]
        y = np.tile(np.sin(x), (1, 6))
        data = pd.DataFrame(y, columns=["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"])
        ic_list = pd.DataFrame({"ic": [5, 10, 15]})
        output = algo.detect(data, ic_list, sampling_rate_hz= 100.0).ic_lr_list_
        assert len(output) == 3
        assert list(output.columns) == ["ic", "lr_label"]

    def test_simple_sin_input(self):
        x = np.linspace(0, 4 * np.pi, 100)[:, None]
        y = np.tile(np.sin(x), (1, 6))
        data = pd.DataFrame(y, columns=["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"])
        ic_list = pd.DataFrame({"ic": [5, 10, 15]})
        params = LrdUllrich.PredefinedParameters.uniss_unige_all_all
        algo = LrdUllrich(**params)
        output = algo.detect(data, ic_list, sampling_rate_hz= 100.0).ic_lr_list_
        assert len(output) == 3
        assert list(output.columns) == ["ic", "lr_label"]

    def test_correct_ouput_format(self):
        params = LrdUllrich.PredefinedParameters.uniss_unige_all_all
        algo = LrdUllrich(**params)
        data = pd.DataFrame(np.zeros((100, 6)), columns=["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"])
        ic_list = pd.DataFrame({"ic": [10, 20 , 40]})
        output = algo.detect(data = data, ic_list = ic_list, sampling_rate_hz = 100.0)
        assert_frame_equal(output.ic_lr_list_[["ic"]], ic_list)
        assert output.feature_matrix_.shape == pd.DataFrame(np.repeat(ic_list.values, 6, axis=1)).shape
        assert isinstance(output.feature_matrix_, pd.DataFrame)


    def test_alternating_output(self):
        # Test a simple sine wave
        # We expect the labels to be alternating
        x = np.linspace(0, 4 * np.pi, 100)[:, None]
        y = np.tile(np.sin(x), (1, 6))
        y[:, [2, 5]] = y[:, [2, 5]] * -1 # make the z axis negative
        data = pd.DataFrame(y, columns=["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"])
        ic_list = pd.DataFrame({"ic": [15, 38, 65]})
        output = LrdUllrich.detect(data = data, ic_list = ic_list, sampling_rate_hz = 100.0).ic_lr_list_
        assert len(output) == 3
        assert list(output.columns) == ["ic", "lr_label"]
        assert (output["lr_label"] == ["right", "left", "right"]).all()




