from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mobgap.laterality import strides_list_from_ic_lr_list
from mobgap.laterality._utils import _unify_stride_list


class TestStridesListFromIcLrList:
    @pytest.mark.parametrize(
        ("multiindex", "column_name", "category_dtype"),
        [
            (False, None, "str"),
            (True, None, "str"),
            (False, "contact_fields", "str"),
            (False, None, "object"),
            (False, None, "string"),
        ],
    )
    def test_unsorted_categorical_contacts_with_equal_times(
        self, multiindex: bool, column_name: str | None, category_dtype: str
    ) -> None:
        """Pair each foot independently while retaining tie order and index metadata."""
        ic_lr_list = pd.DataFrame(
            {
                "ic": [4, 1, 1, 3, 2, 4],
                "lr_label": pd.Categorical(
                    ["left", "right", "left", "right", "left", "right"],
                    categories=pd.Index(["left", "right"], dtype=category_dtype),
                ),
            },
            index=pd.Index(range(6), name="step_id"),
        )
        expected = pd.DataFrame(
            {"start": [1, 1, 2, 3], "end": [2, 3, 4, 4], "lr_label": ["left", "right", "left", "right"]},
            index=pd.Index([2, 1, 4, 3], name="s_id"),
        ).pipe(_unify_stride_list)
        if multiindex:
            ic_lr_list.index = pd.MultiIndex.from_arrays([[7] * 6, ic_lr_list.index], names=["gs_id", "step_id"])
            expected.index = pd.MultiIndex.from_arrays([[7] * 4, expected.index], names=["gs_id", "s_id"])
        ic_lr_list.columns.name = expected.columns.name = column_name

        assert_frame_equal(strides_list_from_ic_lr_list(ic_lr_list), expected, check_exact=True)

    def test_empty(self):
        ic_lr_list = pd.DataFrame([], columns=["ic", "lr_label"])
        output = strides_list_from_ic_lr_list(ic_lr_list)
        assert output.columns.tolist() == ["start", "end", "lr_label"]
        assert output.empty

    def test_only_left(self):
        ic_lr_list = pd.DataFrame(
            {"step_id": [1, 3, 4], "ic": [1, 3, 5], "lr_label": ["left", "left", "left"]}
        ).set_index("step_id")

        output = strides_list_from_ic_lr_list(ic_lr_list)

        expected = (
            pd.DataFrame({"start": [1, 3], "end": [3, 5], "lr_label": ["left", "left"], "s_id": [1, 3]})
            .set_index("s_id")
            .pipe(_unify_stride_list)
        )

        assert_frame_equal(output, expected)

    def test_multiindex_input(self):
        ic_lr_list = pd.DataFrame({"ic": [1, 3, 5], "lr_label": ["left", "left", "left"]})
        ic_lr_list.index = pd.MultiIndex.from_tuples([(1, 1), (1, 2), (1, 3)], names=["wb_id", "step_id"])

        output = strides_list_from_ic_lr_list(ic_lr_list)

        expected = (
            pd.DataFrame(
                {"start": [1, 3], "end": [3, 5], "lr_label": ["left", "left"], "s_id": [1, 2], "wb_id": [1, 1]}
            )
            .set_index(["wb_id", "s_id"])
            .pipe(_unify_stride_list)
        )

        assert_frame_equal(output, expected)

    def test_left_right_input(self):
        ic_lr_list = pd.DataFrame(
            {"ic": [1, 3, 5, 7], "lr_label": ["left", "right", "left", "right"], "step_id": [1, 2, 3, 4]}
        ).set_index("step_id")
        output = strides_list_from_ic_lr_list(ic_lr_list)

        expected = (
            pd.DataFrame({"start": [1, 3], "end": [5, 7], "lr_label": ["left", "right"], "s_id": [1, 2]})
            .set_index("s_id")
            .pipe(_unify_stride_list)
        )

        assert_frame_equal(output, expected)

    def test_left_right_irregular(self):
        ic_lr_list = pd.DataFrame(
            {"ic": [1, 3, 5, 7, 10], "lr_label": ["left", "right", "left", "left", "right"], "step_id": [1, 2, 3, 4, 5]}
        ).set_index("step_id")
        output = strides_list_from_ic_lr_list(ic_lr_list)

        expected = (
            pd.DataFrame(
                {"start": [1, 3, 5], "end": [5, 10, 7], "lr_label": ["left", "right", "left"], "s_id": [1, 2, 3]}
            )
            .set_index("s_id")
            .pipe(_unify_stride_list)
        )

        assert_frame_equal(output, expected)
