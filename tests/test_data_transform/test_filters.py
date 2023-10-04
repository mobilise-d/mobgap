import numpy as np
import pandas as pd
import pytest
from scipy.signal import filtfilt, lfilter
from tpcp.testing import TestAlgorithmMixin

from gaitlink.data_transform import EpflDedriftFilter, EpflGaitFilter
from gaitlink.data_transform.base import FixedFilter


class TestMetaEpflGaitFilter(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = EpflGaitFilter

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().filter(pd.DataFrame(np.zeros((500, 3))), sampling_rate_hz=40.0)


class TestMetaEpflDedriftFilter(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = EpflDedriftFilter

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().filter(pd.DataFrame(np.zeros((500, 3))), sampling_rate_hz=40.0)


class TestFixedFilter:
    filter_subclass: type[FixedFilter]

    @pytest.fixture(autouse=True, params=(EpflDedriftFilter, EpflGaitFilter))
    def get_algorithm(self, request):
        self.filter_subclass = request.param

    @pytest.mark.parametrize("zero_phase", [True, False])
    @pytest.mark.parametrize("dtype", [np.array, pd.DataFrame, pd.Series, "array_1d"])
    def test_simple_filter(self, zero_phase, dtype):
        data_raw = np.random.rand(1000, 1)
        if dtype == pd.DataFrame:
            data = pd.DataFrame(data_raw)
        elif dtype == pd.Series:
            data = pd.Series(data_raw[:, 0])
        elif dtype == "array_1d":
            data = data_raw[:, 0]
        else:
            data = data_raw

        result = self.filter_subclass(zero_phase=zero_phase).filter(data, sampling_rate_hz=40.0)

        if zero_phase:
            reference = filtfilt(*self.filter_subclass().coefficients, data_raw, axis=0)
        else:
            reference = lfilter(*self.filter_subclass().coefficients, data_raw, axis=0)

        if dtype in (pd.Series, "array_1d"):
            reference = pd.Series(reference[:, 0])
        assert np.allclose(result.transformed_data_, reference)
        assert type(result.transformed_data_) == type(data)

        assert result.transformed_data_ is result.filtered_data_

    def test_error_on_wrong_sampling_rate(self):
        with pytest.raises(ValueError):
            self.filter_subclass().filter(
                pd.DataFrame(np.zeros((500, 3))), sampling_rate_hz=self.filter_subclass.EXPECTED_SAMPLING_RATE_HZ + 1
            )
