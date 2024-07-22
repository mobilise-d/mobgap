import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mobgap.utils.df_operations import create_multi_groupby

# NOTE: The tests for apply_aggregations and apply_transformations can be found in `test_pipeline/test_evaluation`.


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

    @pytest.mark.parametrize(
        "groupby",
        [
            "group1",
            ["group1", "group2"],
            ["group1", "group2", "group3"],
            ["group1", "group3"],
            ["group2", "group3"],
            ["group2"],
        ],
    )
    def test_groupby_group_subset(self, groupby):
        all_groups = ["group1", "group2", "group3"]
        df = pd.DataFrame(
            {
                "group1": [1, 1, 1],
                "group2": [1, 1, 2],
                "group3": [1, 2, 3],
                "value": [1, 2, 3],
            }
        ).set_index(all_groups)
        df_2 = df + 10

        if isinstance(groupby, str):
            groupby_as_list = [groupby]
        else:
            groupby_as_list = groupby

        remaining_groups = list(set(all_groups) - set(groupby_as_list))

        multi_groupby = create_multi_groupby(df, df_2, groupby)
        for group, (df1, df2) in multi_groupby.groups.items():
            for df_ in (df1, df2):
                assert set(df_.reset_index(remaining_groups).index.to_list()) == {group}
            assert_frame_equal(df1, df2 - 10)

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

    def test_apply(self):
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

        multi_groupby = create_multi_groupby(df, df_2, ["group1", "group2"])

        def func(df1, df2):
            return (df1 + df2).iloc[0]

        result = multi_groupby.apply(func)
        expected = df + df_2.reindex(df.index)
        assert_frame_equal(result, expected)

    def test_apply_args_kwargs(self):
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

        multi_groupby = create_multi_groupby(df, df_2, ["group1", "group2"])

        def func(df1, df2, a, b, c=1):
            return (df1 + df2).iloc[0] + a + b + c

        result = multi_groupby.apply(func, 1, b=2, c=3)
        expected = df + df_2.reindex(df.index) + 1 + 2 + 3
        assert_frame_equal(result, expected)
