import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.wba import IntervalParameterCriteria, StrideSelection


class TestStrideSelectionMeta(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = StrideSelection

    @pytest.fixture
    def after_action_instance(self, naive_stride_list) -> StrideSelection:
        return self.ALGORITHM_CLASS([("simple_thres", IntervalParameterCriteria("para_1", 1.5, 2.5))]).filter(
            naive_stride_list, sampling_rate_hz=1
        )


def test_stride_list_creation(naive_stride_list):
    assert len(naive_stride_list) == 100


@pytest.mark.parametrize(
    "rules",
    (
        None,
        [("para_1", IntervalParameterCriteria("para_1", 0, 1.5))],
    ),
)
def test_stride_filter_all_valid(naive_stride_list, rules):
    selector = StrideSelection(rules)
    filtered_stride_list = selector.filter(naive_stride_list, sampling_rate_hz=1).filtered_stride_list_
    pd.testing.assert_frame_equal(filtered_stride_list, naive_stride_list)


@pytest.mark.parametrize("rules", ([("para_1", IntervalParameterCriteria("para_1", 1.5, 2.5))],))
def test_stride_filter_all_invalid(naive_stride_list, rules):
    selector = StrideSelection(rules)
    filtered_stride_list = selector.filter(naive_stride_list, sampling_rate_hz=1).filtered_stride_list_
    assert len(filtered_stride_list) == 0
    assert len(selector.excluded_stride_list_) == len(naive_stride_list)
    assert selector.exclusion_reasons_.loc[naive_stride_list.iloc[0].name]["rule_name"] == rules[0][0]
    assert selector.exclusion_reasons_.loc[naive_stride_list.iloc[0].name]["rule_obj"] == rules[0][1]
    pd.testing.assert_series_equal(selector.excluded_stride_list_.iloc[0], naive_stride_list.iloc[0])


def test_stride_selection_wrong_rules():
    with pytest.raises(TypeError):
        StrideSelection([("test", "something_invalid")]).filter(pd.DataFrame([]), sampling_rate_hz=1)
