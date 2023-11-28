import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


def test_generic_data_transform():
    from examples.data_transform._01_generic_data_transforms import MyComplicatedAlgorithm, ScaleTransformer

    data = np.array([1, 2, 3])

    my_algorithm = MyComplicatedAlgorithm()
    assert_frame_equal(my_algorithm.run(pd.DataFrame(data)).result_, pd.DataFrame([2, 3, 4]))

    my_algorithm = MyComplicatedAlgorithm(pre_processing=ScaleTransformer(scale_by=3))
    assert_frame_equal(my_algorithm.run(pd.DataFrame(data)).result_, pd.DataFrame([3, 6, 9]))


def test_filter(snapshot):
    from examples.data_transform._02_filter import epfl_filter

    snapshot.assert_match(epfl_filter.filtered_data_, "filtered_data")


def test_resample():
    # For now we just check that the code runs without errors, as the example might need to be reworked
    from examples.data_transform._03_resample import resampler  # noqa
