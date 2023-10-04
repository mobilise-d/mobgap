import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_series_equal

from gaitlink.utils.dtypes import dflike_as_2d_array


class TestDflikeAs2dArray:
    def test_invalid_dflike(self):
        with pytest.raises(TypeError):
            dflike_as_2d_array(1)

    @pytest.mark.parametrize("data", [np.arange(10).reshape(2, 5, 1), np.array(3)])
    def test_invalid_ndim(self, data):
        with pytest.raises(ValueError):
            dflike_as_2d_array(data)

    def test_1d_array(self):
        data = np.arange(10)
        result, convert = dflike_as_2d_array(data)

        assert np.all(result[:, 0] == data)
        assert np.all(convert(result) == data)

    def test_2d_array(self):
        data = np.arange(10).reshape(2, 5)
        result, convert = dflike_as_2d_array(data)

        assert np.all(result == data)
        assert np.all(convert(result) == data)

    def test_series(self):
        data = pd.Series(np.arange(10))
        data.index += 1
        result, convert = dflike_as_2d_array(data)

        assert np.all(result[:, 0] == data.to_numpy())
        assert_series_equal(convert(result), data)

    def test_dataframe(self):
        data = pd.DataFrame(np.arange(10).reshape(2, 5))
        data.index += 1
        result, convert = dflike_as_2d_array(data)

        assert np.all(result == data.to_numpy())
        assert np.all(result[:, 0] == data[0])
        pd.testing.assert_frame_equal(convert(result), data)
