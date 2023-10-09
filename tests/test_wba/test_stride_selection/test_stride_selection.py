import pytest

from gaitlink.wba._stride_criteria import ThresholdCriteria
from gaitlink.wba._stride_selection import StrideSelection


def test_stride_list_creation(naive_stride_list):
    assert len(naive_stride_list) == 100


@pytest.mark.parametrize(
    "rules",
    (
        None,
        [("length", ThresholdCriteria("length", 0, 1.5))],
    ),
)
def test_stride_filter_all_valid(naive_stride_list, rules):
    selector = StrideSelection(rules)
    filtered_stride_list = selector.filter(naive_stride_list).filtered_stride_list_
    assert filtered_stride_list == naive_stride_list


@pytest.mark.parametrize("rules", ([("length", ThresholdCriteria("length", 1.5, 2.5))],))
def test_stride_filter_all_invalid(naive_stride_list, rules):
    selector = StrideSelection(rules)
    filtered_stride_list = selector.filter(naive_stride_list).filtered_stride_list_
    assert len(filtered_stride_list) == 0
    assert len(selector.excluded_stride_list_) == len(naive_stride_list)
    assert selector.exclusion_reasons_[naive_stride_list[0]["id"]] == rules[0]
    assert selector.excluded_stride_list_[0] == naive_stride_list[0]


def test_stride_selection_wrong_rules():
    with pytest.raises(ValueError):
        StrideSelection([("test", "something_invalid")]).filter([])
