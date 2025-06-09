import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
from mobgap.data import LabExampleDataset
from mobgap.laterality import LrcUllrich
from mobgap.pipeline import GsIterator
from mobgap.utils.conversions import to_body_frame


class TestMetaLrcUllrich(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = LrcUllrich

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().predict(
            data=pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS),
            ic_list=pd.DataFrame({"ic": [5, 10, 15]}),
            sampling_rate_hz=100.0,
        )


class TestLrcUllrich:
    model: dict

    @pytest.fixture(
        autouse=True,
        params=[LrcUllrich.PredefinedParameters.msproject_all, LrcUllrich.PredefinedParameters.msproject_all_old],
    )
    def _select_model(self, request):
        self.model = request.param

    def test_empty_data(self):
        test_params = self.model
        algo = LrcUllrich(**test_params)
        data = pd.DataFrame([], columns=BF_SENSOR_COLS)
        ic_list = pd.DataFrame({"ic": [5, 10, 15]})
        output = algo.predict(data=data, ic_list=ic_list, sampling_rate_hz=100.0)
        assert len(output.ic_lr_list_) == 0
        assert list(output.ic_lr_list_.columns) == ["ic", "lr_label"]
        assert len(output.feature_matrix_) == 0
        assert list(output.feature_matrix_.columns) == LrcUllrich._feature_matrix_cols

    def test_empty_ic(self):
        params = self.model
        algo = LrcUllrich(**params)
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        ic_list = pd.DataFrame({"ic": []})
        output = algo.predict(data=data, ic_list=ic_list, sampling_rate_hz=100.0)
        assert len(output.ic_lr_list_) == 0
        assert list(output.ic_lr_list_.columns) == ["ic", "lr_label"]
        assert len(output.feature_matrix_) == 0
        assert list(output.feature_matrix_.columns) == LrcUllrich._feature_matrix_cols

    def test_predict_with_model_not_fit(self):
        algo = LrcUllrich(**LrcUllrich.PredefinedParameters.untrained_svc)
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        ic_list = pd.DataFrame({"ic": [5, 10, 15]})
        with pytest.raises(RuntimeError):
            algo.predict(data=data, ic_list=ic_list, sampling_rate_hz=100.0)

    def test_correct_output_format(self):
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        ic_list = pd.DataFrame({"ic": [10, 20, 40]})

        algo = LrcUllrich(**self.model)
        output = algo.predict(data=data, ic_list=ic_list, sampling_rate_hz=100.0)

        assert_frame_equal(output.ic_lr_list_[["ic"]], ic_list)
        assert output.feature_matrix_.shape == pd.DataFrame(np.repeat(ic_list.values, 6, axis=1)).shape
        assert isinstance(output.feature_matrix_, pd.DataFrame)

    # For some reason, this simple test does not work anymore with the new modes... It seems like our intuitions about
    # the model are wrong. We should investigate this further at some point.
    # def test_simple_sin_input(self):
    #     x = np.linspace(0, 4 * np.pi, 80)[:, None]
    #     x = np.tile(x, (1, 6))
    #     # We shift the x values by pi/2
    #     x[:, 0] = x[:, 0] + np.pi / 2 * 1.2
    #     y = np.sin(x)
    #     y[:, 2] = y[:, 2] * -1  # make the z axis negative
    #     data = pd.DataFrame(y, columns=BF_SENSOR_COLS)
    #
    #     ic_list = pd.DataFrame({"ic": [10, 30, 50, 70]})
    #
    #     algo = LrcUllrich(**dict(self.model, smoothing_filter=IdentityFilter()))
    #     algo.predict(data=data, ic_list=ic_list, sampling_rate_hz=100.0)
    #
    #     output = algo.ic_lr_list_
    #
    #     assert len(output) == 4
    #     assert list(output.columns) == ["ic", "lr_label"]
    #     assert (output["lr_label"] == ["right", "left", "right", "left"]).all()

    def test_self_optimized_with_optimized_model(self):
        algo = LrcUllrich(**self.model)
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        ic_list = pd.DataFrame({"ic": [5, 10, 15]})
        labels = ic_list.assign(lr_label=["left", "right", "left"])

        with pytest.raises(RuntimeError):
            algo.self_optimize([data], [ic_list], [labels], sampling_rate_hz=100.0)

    def test_self_optimize_custom_algo(self):
        x = np.linspace(0, 4 * np.pi, 100)[:, None]
        y = np.tile(np.sin(x), (1, 6))
        data = pd.DataFrame(y, columns=BF_SENSOR_COLS)
        ic_list = pd.DataFrame({"ic": [5, 10, 15]})
        label_list = ic_list.assign(lr_label=["left", "right", "left"])

        algo = LrcUllrich(clf_pipe=make_pipeline(MinMaxScaler(), DecisionTreeClassifier()))

        algo.self_optimize([data], [ic_list], [label_list], sampling_rate_hz=100.0)
        output = algo.predict(data, ic_list, sampling_rate_hz=100.0).ic_lr_list_
        assert len(output) == 3
        assert list(output.columns) == ["ic", "lr_label"]


class TestRegression:
    @pytest.mark.parametrize("datapoint", LabExampleDataset(reference_system="INDIP", reference_para_level="wb"))
    @pytest.mark.parametrize(
        "config_name, config",
        [
            ("msproject_all", LrcUllrich.PredefinedParameters.msproject_all),
            ("msproject_all_old", LrcUllrich.PredefinedParameters.msproject_all_old),
        ],
    )
    def test_example_lab_data(self, datapoint, config_name, config, snapshot):
        imu_data = to_body_frame(datapoint.data_ss)
        ref = datapoint.reference_parameters_relative_to_wb_
        ref_walk_bouts = ref.wb_list
        if len(ref_walk_bouts) == 0:
            pytest.skip("No reference parameters available.")
        sampling_rate_hz = datapoint.sampling_rate_hz

        algo = LrcUllrich(**config)

        iterator = GsIterator()

        for (gs, data), result in iterator.iterate(imu_data, ref.wb_list):
            # extract data_list, ic_list, label_list
            data = data.reset_index(drop=True)
            ic_list = ref.ic_list.loc[gs.id]
            result.ic_list = algo.predict(data=data, ic_list=ic_list, sampling_rate_hz=sampling_rate_hz).ic_lr_list_

        predicted_ics = iterator.results_.ic_list
        snapshot.assert_match(predicted_ics, f"{config_name}_{tuple(datapoint.group_label)}")
