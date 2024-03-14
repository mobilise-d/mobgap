import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from gaitlink.data import LabExampleDataset
from tpcp.testing import TestAlgorithmMixin
from gaitlink.lrd._utils import extract_ref_data
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
        test_params = LrdUllrich.PredefinedParameters.msproject_all
        algo = LrdUllrich(**test_params)
        data = pd.DataFrame([], columns=["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"])
        ic_list = pd.DataFrame({"ic": [5, 10 , 15]})
        output = algo.detect(data = data, ic_list = ic_list, sampling_rate_hz = 100.0)
        assert(len(output.ic_lr_list_) == 0)
        assert list(output.ic_lr_list_.columns) == ["ic", "lr_label"]
        assert(len(output.feature_matrix_) == 0)
        assert list(output.feature_matrix_.columns) == ["filtered_gyr_x", "gradient_gyr_x", "diff_2_gyr_x", "filtered_gyr_z", "gradient_gyr_z", "diff_2_gyr_z"]

    def test_empty_ic(self):
        params = LrdUllrich.PredefinedParameters.msproject_all
        algo = LrdUllrich(**params)
        data = pd.DataFrame(np.zeros((100, 6)), columns=["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"])
        ic_list = pd.DataFrame({"ic": []})
        output = algo.detect(data = data, ic_list = ic_list, sampling_rate_hz = 100.0)
        assert(len(output.ic_lr_list_) == 0)
        assert list(output.ic_lr_list_.columns) == ["ic", "lr_label"]
        assert(len(output.feature_matrix_) == 0)
        assert list(output.feature_matrix_.columns) == ["filtered_gyr_x", "gradient_gyr_x", "diff_2_gyr_x", "filtered_gyr_z", "gradient_gyr_z", "diff_2_gyr_z"]

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
        label_list = pd.DataFrame({"lr_label": ["left", "right", "left"]})
        algo.self_optimize(data_list = [data], ic_list = [ic_list], label_list=[label_list], sampling_rate_hz=100.0)
        output = algo.detect(data, ic_list, sampling_rate_hz= 100.0).ic_lr_list_
        assert len(output) == 3
        assert list(output.columns) == ["ic", "lr_label"]

    def test_simple_sin_input(self):
        x = np.linspace(0, 4 * np.pi, 100)[:, None]
        y = np.tile(np.sin(x), (1, 6))
        data = pd.DataFrame(y, columns=["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"])
        ic_list = pd.DataFrame({"ic": [5, 10, 15]})
        params = LrdUllrich.PredefinedParameters.msproject_all
        algo = LrdUllrich(**params)
        output = algo.detect(data, ic_list, sampling_rate_hz= 100.0).ic_lr_list_
        assert len(output) == 3
        assert list(output.columns) == ["ic", "lr_label"]

    def test_correct_ouput_format(self):
        params = LrdUllrich.PredefinedParameters.msproject_all
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

# TODO: this needs checking
class testRegression:
    @pytest.mark.parametrize("datapoint", LabExampleDataset(reference_system="INDIP", reference_para_level="wb"))

    def test_example_lab_data_msproject_all(self, datapoint, snapshot):

        params = LrdUllrich.PredefinedParameters.msproject_all
        algo = LrdUllrich(**params)
        try:
                data_list, ic_list, _ = extract_ref_data(datapoint)
        except:
                pytest.skip("No reference parameters available.")
        sampling_rate_hz = datapoint.sampling_rate_hz

        detected_lr_labels = [algo.detect(data=data, ic_list=ics, sampling_rate_hz=sampling_rate_hz).ic_lr_list_ for data, ics in zip(data_list, ic_list)]

        snapshot.assert_match(detected_lr_labels, str(datapoint.group_label))

    def test_example_lab_data_msproject_ms(self, datapoint, snapshot):

        params = LrdUllrich.PredefinedParameters.msproject_ms
        algo = LrdUllrich(**params)
        try:
                data_list, ic_list, _ = extract_ref_data(datapoint)
        except:
                pytest.skip("No reference parameters available.")
        sampling_rate_hz = datapoint.sampling_rate_hz

        detected_lr_labels = [algo.detect(data=data, ic_list=ics, sampling_rate_hz=sampling_rate_hz).ic_lr_list_ for data, ics in zip(data_list, ic_list)]

        snapshot.assert_match(detected_lr_labels, str(datapoint.group_label))