import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from gaitlink.utils.array_handling import (
    create_multi_groupby,
    sliding_window_view,
)


class TestSlidingWindowView:
    def test_no_overlap(self):
        view = sliding_window_view(np.arange(14), window_size_samples=3, overlap_samples=0)

        assert view.shape == (4, 3)
        assert np.all(view == np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))

    def test_overlap(self):
        view = sliding_window_view(np.arange(14), window_size_samples=3, overlap_samples=1)

        assert view.shape == (6, 3)
        assert np.all(view == np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 10], [10, 11, 12]]))

    def test_nd_array(self):
        data = np.arange(112).reshape(14, 2, 4)
        view = sliding_window_view(data, window_size_samples=3, overlap_samples=1)

        assert view.shape == (6, 3, 2, 4)
        assert np.all(view[:, :, 0] == sliding_window_view(data[:, 0], window_size_samples=3, overlap_samples=1))
        assert np.all(view[:, :, 1] == sliding_window_view(data[:, 1], window_size_samples=3, overlap_samples=1))

    def test_error_overlap_larger_than_window(self):
        with pytest.raises(ValueError):
            sliding_window_view(np.arange(14), window_size_samples=3, overlap_samples=4)


class TestMultiGroupby:
    def test_simple_iter(self):
        df = pd.DataFrame(
            {
                "group1": [1, 1, 2, 2, 3, 3],
                "group2": [1, 2, 1, 2, 1, 2],
                "value": [1, 2, 3, 4, 5, 6],
            }
        ).set_index(["group1", "group2"])
        df_2 = df + 10
        df_3 = df + 20

        multi_groupby = create_multi_groupby(df, [df_2, df_3], ["group1", "group2"])
        i = 0
        for group, (df1, df2, df3) in multi_groupby.groups.items():
            assert_frame_equal(df1, df.loc[[group]])
            assert_frame_equal(df1, multi_groupby.get_group(group)[0])
            assert_frame_equal(df2, df_2.loc[[group]])
            assert_frame_equal(df2, multi_groupby.get_group(group)[1])
            assert_frame_equal(df3, df_3.loc[[group]])
            assert_frame_equal(df3, multi_groupby.get_group(group)[2])
            i += 1

        assert i == multi_groupby.ngroups == len(multi_groupby) == len(df.groupby(level=["group1", "group2"]))

        # We test both plausible ways to access the groups
        for group, (df1, df2, df3) in multi_groupby:
            assert_frame_equal(df1, df.loc[[group]])
            assert_frame_equal(df2, df_2.loc[[group]])
            assert_frame_equal(df3, df_3.loc[[group]])

    def test_iter_with_one_level_groupby(self):
        df = pd.DataFrame(
            {
                "group1": [1, 1, 2, 2, 3, 3],
                "group2": [1, 2, 1, 2, 1, 2],
                "value": [1, 2, 3, 4, 5, 6],
            }
        ).set_index(["group1", "group2"])
        df_2 = pd.DataFrame(
            {
                "group1": [1, 1, 1, 2, 3, 3],
                "group2": [1, 2, 3, 1, 1, 2],
                "value": [11, 12, 13, 14, 15, 16],
            }
        ).set_index(["group1", "group2"])

        multi_groupby = create_multi_groupby(df, df_2, "group1")

        i = 0
        for group, (df1, df2) in multi_groupby.groups.items():
            assert_frame_equal(df1, df.loc[[group]])
            assert_frame_equal(df2, df_2.loc[[group]])
            i += 1

        assert i == multi_groupby.ngroups == len(multi_groupby) == len(df.groupby(level=["group1"]))

        for group, (df1, df2) in multi_groupby:
            assert_frame_equal(df1, df.loc[[group]])
            assert_frame_equal(df2, df_2.loc[[group]])

    def test_iter_with_missing_element_groupby(self):
        df = pd.DataFrame(
            {
                "group1": [1, 1, 3, 3],
                "group2": [1, 2, 1, 2],
                "value": [1, 2, 3, 4],
            }
        ).set_index(["group1", "group2"])
        df_2 = pd.DataFrame(
            {
                "group1": [1, 1, 2, 2],
                "group2": [1, 2, 1, 2],
                "value": [11, 12, 13, 14],
            }
        ).set_index(["group1", "group2"])

        multi_groupby = create_multi_groupby(df, df_2, "group1")
        for group, (df1, df2) in multi_groupby:
            assert_frame_equal(df1, df.loc[[group]])
            # For df_2 some values dont't exist in this case we expect an empty dataframe
            try:
                expected = df_2.loc[[group]]
            except KeyError:
                expected = pd.DataFrame(columns=df_2.columns, index=df_2.index[:0])
            assert_frame_equal(df2, expected)
