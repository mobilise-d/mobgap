import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


def test_generic_data_transform():
    from examples.data_transform._01_generic_data_transforms import MyComplicatedAlgorithm, ScaleTransformer

    data = np.array([1, 2, 3], dtype="int64")

    my_algorithm = MyComplicatedAlgorithm()
    assert_frame_equal(my_algorithm.run(pd.DataFrame(data)).result_, pd.DataFrame([2, 3, 4]))

    my_algorithm = MyComplicatedAlgorithm(pre_processing=ScaleTransformer(scale_by=3))
    assert_frame_equal(my_algorithm.run(pd.DataFrame(data)).result_, pd.DataFrame([3, 6, 9]))


def test_filter(snapshot):
    from examples.data_transform._02_filter import butterworth_filter, epfl_filter

    filtered_data = epfl_filter.filtered_data_
    filtered_data.index = filtered_data.index.round("ms")

    snapshot.assert_match(filtered_data, "epfl")

    filtered_data = butterworth_filter.filtered_data_
    filtered_data.index = filtered_data.index.round("ms")

    snapshot.assert_match(filtered_data, "butterworth")


def test_resample(snapshot):
    from examples.data_transform._03_resample import resampler

    resampled_data = resampler.transformed_data_
    resampled_data.index = resampled_data.index.round("ms")

    snapshot.assert_match(resampled_data)


def test_cwt_filter(snapshot):
    from examples.data_transform._04_cwt_filter import cwt_filter

    filtered_data = cwt_filter.transformed_data_
    filtered_data.index = filtered_data.index.round("ms")

    snapshot.assert_match(filtered_data)


def test_gaussian_filter(snapshot):
    from examples.data_transform._05_gaussian_smoothing import gaussian_filter

    filtered_data = gaussian_filter.transformed_data_
    filtered_data.index = filtered_data.index.round("ms")

    snapshot.assert_match(filtered_data)


def test_savgol_filter(snapshot):
    from examples.data_transform._06_savgol_filter import savgol_filter

    filtered_data = savgol_filter.transformed_data_
    filtered_data.index = filtered_data.index.round("ms")

    snapshot.assert_match(filtered_data)
