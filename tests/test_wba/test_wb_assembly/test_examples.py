import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from statsmodels.compat.pandas import assert_series_equal

from mobgap.wba import IntervalParameterCriteria, MaxBreakCriteria, NStridesCriteria, StrideSelection, WbAssembly
from tests.test_wba.conftest import window


def naive_stride_list(start, stop, duration, foot=None, **paras):
    """A window list full of identical strides."""
    x = np.arange(start, stop + duration, duration, dtype="int64")
    start_end = zip(x[:-1], x[1:])

    return pd.DataFrame.from_records(
        [window(start=s, end=e, foot=foot, duration=duration, **paras) for i, (s, e) in enumerate(start_end)]
    ).set_index("s_id")


@pytest.mark.parametrize("consider_end_as_break", [True, False])
def test_simple_single_wb(consider_end_as_break):
    """Simple single WB test.

    Scenario:
        - Break at the beginning gait, break
        - No left right foot

    Rules:
        - BreakCriterium
        - MinStride Inclusion Rule

    """
    wb_start_time = 5
    n_strides = 10
    strides = pd.DataFrame.from_records(
        [window(wb_start_time + i, wb_start_time + i + 1) for i in range(n_strides)]
    ).set_index("s_id")
    rules = [
        ("break", MaxBreakCriteria(3, consider_end_as_break=consider_end_as_break)),
        ("n_strides", NStridesCriteria(4)),
    ]

    wba = WbAssembly(rules)
    wba.assemble(strides, sampling_rate_hz=1)

    assert len(wba.wbs_) == 1
    single_wb = next(iter(wba.wbs_.values()))
    single_wb_id = next(iter(wba.wbs_.keys()))
    assert len(single_wb) == n_strides
    assert_frame_equal(single_wb, strides)

    assert len(wba.excluded_stride_list_) == 0
    assert len(wba.excluded_wbs_) == 0
    assert len(wba.exclusion_reasons_) == 0
    assert len(wba.termination_reasons_) == 1

    assert wba.termination_reasons_.loc[single_wb_id, "rule_name"] == (
        "break" if consider_end_as_break else "end_of_list"
    )


@pytest.mark.parametrize("consider_end_as_break", [True, False])
def test_simple_break_center(consider_end_as_break):
    """Test gait sequence with a break in the center.

    Scenario:
        - long gait sequence with a break in the center
        - no left right

    Rules:
        - BreakCriterium
        - MinStride Inclusion Rule

    Outcome:
        - Two WBs (one for one after the break)
        - no strides discarded
    """
    wb_start_time = 5
    n_strides = 20
    strides = pd.DataFrame.from_records(
        [window(wb_start_time + i, wb_start_time + i + 1) for i in range(n_strides)]
    ).set_index("s_id")
    strides = strides.drop(strides.index[7:11])
    rules = [
        ("break", MaxBreakCriteria(3, consider_end_as_break=consider_end_as_break)),
        ("n_strides", NStridesCriteria(4)),
    ]

    wba = WbAssembly(rules)
    wba.assemble(strides, sampling_rate_hz=1)

    assert len(wba.excluded_wbs_) == 0
    assert len(wba.excluded_stride_list_) == 0
    assert len(wba.termination_reasons_) == 2
    assert len(wba.exclusion_reasons_) == 0

    assert len(wba.wbs_) == 2
    wbs = list(wba.wbs_.values())
    wb_ids = list(wba.wbs_.keys())
    assert wbs[0].iloc[0]["start"] == wb_start_time
    assert wbs[0].iloc[-1]["end"] == wb_start_time + 7
    assert wba.termination_reasons_.loc[wb_ids[0], "rule_name"] == "break"
    assert_frame_equal(wbs[0], strides.iloc[:7])

    assert wbs[1].iloc[0]["start"] == 16
    assert wbs[1].iloc[-1]["end"] == wb_start_time + n_strides
    assert_frame_equal(wbs[1], strides.iloc[7:])
    assert wba.termination_reasons_.loc[wb_ids[1], "rule_name"] == ("break" if consider_end_as_break else "end_of_list")


def test_full_complicated_example(snapshot):
    stride_list = [
        naive_stride_list(0, 5000, 100, foot="left"),
        naive_stride_list(50, 5050, 100, foot="right"),
        naive_stride_list(5000, 6020, 60, foot="left"),
        naive_stride_list(5050, 6070, 60, foot="right"),
        naive_stride_list(6020, 8000, 90, foot="left"),
        naive_stride_list(6070, 8050, 90, foot="right"),
    ]

    stride_list = pd.concat(stride_list).sort_values("start")
    # We want to have predictable ids
    stride_list = stride_list.reset_index(drop=True).rename_axis(index="s_id")
    # We add some additional parameters, we can use to filter later on.
    large_sl_ids = [10, 11, 12, 13, 14, 18, 19, 20, 21, 56, 90, 91, 121, 122, 176]
    stride_list["stride_length"] = 1
    stride_list.loc[stride_list.index[large_sl_ids], "stride_length"] = 2

    rules = [
        (
            "sl_thres",
            IntervalParameterCriteria("stride_length", lower_threshold=0.5, upper_threshold=1.5),
        )
    ]

    ss = StrideSelection(rules)
    ss.filter(stride_list, sampling_rate_hz=1)

    filtered_stride_list = ss.filtered_stride_list_

    rules = [
        (
            "max_break",
            MaxBreakCriteria(max_break_s=10, remove_last_ic="per_foot", consider_end_as_break=True),
        ),
        ("min_strides", NStridesCriteria(min_strides=5)),
    ]

    wb_assembly = WbAssembly(rules)
    wb_assembly.assemble(filtered_stride_list, sampling_rate_hz=1)

    snapshot.assert_match(wb_assembly.annotated_stride_list_, "wba")
    snapshot.assert_match(ss.filtered_stride_list_, "stride_selection")


def test_n_initial_contacts():
    wb_start_time = 5
    n_strides = 20
    strides = pd.DataFrame.from_records(
        [window(wb_start_time + i, wb_start_time + i + 1) for i in range(n_strides)]
    ).set_index("s_id")
    strides = strides.drop(strides.index[7:11])

    # WB 1: 5-12, WB 2: 16-25
    raw_initial_contacts = pd.DataFrame({"ic": [0, 1, 2, 3, 5, 6, 7, 8, 9, 16, 17, 18]})
    rules = [
        ("break", MaxBreakCriteria(3, consider_end_as_break=True)),
        ("n_strides", NStridesCriteria(4)),
    ]

    wba = WbAssembly(rules)
    wba.assemble(strides, raw_initial_contacts=raw_initial_contacts, sampling_rate_hz=1)
    assert len(wba.wbs_) == 2
    assert_series_equal(
        wba.wb_meta_parameters_["n_raw_initial_contacts"],
        pd.Series([5, 3], index=wba.wb_meta_parameters_.index, name="n_raw_initial_contacts", dtype="Int64"),
    )
