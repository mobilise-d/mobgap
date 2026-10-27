from dataclasses import dataclass

import pandas as pd
from pandas.testing import assert_frame_equal

from mobgap.pipeline import RegionIterator


@dataclass
class Summaries:
    statistics: pd.Series
    quality: pd.Series


def test_custom_series_fields_are_assembled_as_independent_tables():
    regions = pd.DataFrame({"start": [0, 2], "end": [2, 4]}, index=pd.Index(["a", "b"], name="segment"))
    iterator = RegionIterator(Summaries).clone()
    for (_region, window), result in iterator.iterate(pd.DataFrame({"signal": [1, 3, 8, 10]}), regions):
        result.statistics = pd.Series({"mean": window.signal.mean(), "peak": window.signal.max()}, name="ignored")
        result.quality = pd.Series({"complete": len(window) == 2})

    assert_frame_equal(
        iterator.results_.statistics,
        pd.DataFrame({"mean": [2.0, 9.0], "peak": [3.0, 10.0]}, index=regions.index),
    )
    assert_frame_equal(iterator.results_.quality, pd.DataFrame({"complete": [True, True]}, index=regions.index))


@dataclass
class Values:
    values: pd.Series


def test_windows_preserve_order_and_metadata_stays_in_source_table():
    data = pd.DataFrame({"signal": [1, 2, 3, 4, 5, 6]}, index=range(100, 106))
    regions = pd.DataFrame(
        {"start": [3, 0], "end": [6, 2], "scale": [2, -1]},
        index=pd.Index(["later", "earlier"], name="region_id"),
    )
    iterator = RegionIterator(Values)
    for (region, window), result in iterator.iterate(data, regions):
        result.values = pd.Series({"value": window.signal.sum() * regions.loc[region.id, "scale"]}, name="ignored")
    assert_frame_equal(iterator.results_.values, pd.DataFrame({"value": [30, -3]}, index=regions.index))
    assert_frame_equal(iterator.raw_results_[0].input.data, data.iloc[3:6])


def test_composite_ids_and_overlapping_regions_preserve_identity_without_offsets():
    regions = pd.DataFrame(
        {"start": [4, 1, 3], "end": [6, 3, 5]},
        index=pd.MultiIndex.from_tuples([(8, 0), (2, 0), (8, 1)], names=["recording_id", "event_id"]),
    )
    iterator = RegionIterator(Values)
    seen = []
    for (region, window), result in iterator.iterate(pd.DataFrame({"signal": [0, 10, 20, 30, 40, 50]}), regions):
        seen.append(region.id)
        result.values = pd.Series({"peak": window.signal.max(), "start": 0})
    assert seen == [(8, 0), (2, 0), (8, 1)]
    assert_frame_equal(
        iterator.results_.values,
        pd.DataFrame({"peak": [50, 20, 40], "start": [0, 0, 0]}, index=regions.index),
    )


def test_unset_series_are_omitted_independently_for_each_result_field():
    regions = pd.DataFrame({"start": [0, 1, 2], "end": [1, 2, 3]}, index=pd.Index([5, 2, 8], name="region_id"))
    iterator = RegionIterator(Summaries)
    for (region, window), result in iterator.iterate(pd.DataFrame({"signal": [10, 20, 30]}), regions):
        if region.id != 2:
            result.statistics = pd.Series({"first": window.signal.iloc[0]})
        if region.id != 5:
            result.quality = pd.Series({"complete": True})
    assert_frame_equal(
        iterator.results_.statistics,
        pd.DataFrame({"first": [10, 30]}, index=pd.Index([5, 8], name="region_id")),
    )
    assert_frame_equal(
        iterator.results_.quality,
        pd.DataFrame({"complete": [True, True]}, index=pd.Index([2, 8], name="region_id")),
    )


def test_empty_reuse_clears_cached_results_and_preserves_composite_index():
    iterator = RegionIterator(Values)
    data = pd.DataFrame({"signal": [1]})
    for _, result in iterator.iterate(data, pd.DataFrame({"start": [0], "end": [1]})):
        result.values = pd.Series({"value": 1})
    assert not iterator.results_.values.empty

    empty_regions = pd.DataFrame(
        {"start": pd.Series(dtype=int), "end": pd.Series(dtype=int)},
        index=pd.MultiIndex.from_arrays([[], []], names=["recording_id", "event_id"]),
    )
    list(iterator.iterate(data, empty_regions))
    assert_frame_equal(iterator.results_.values, pd.DataFrame(index=empty_regions.index))


def test_series_labels_align_as_columns_and_names_do_not_replace_region_ids():
    regions = pd.DataFrame({"start": [0, 1], "end": [1, 2]}, index=pd.Index(["z", "a"], name="region_id"))
    iterator = RegionIterator(Values)
    for (region, _), result in iterator.iterate(pd.DataFrame({"signal": [0, 0]}), regions):
        result.values = pd.Series({"left": 1, "right": 2} if region.id == "z" else {"right": 3}, name="same")
    assert_frame_equal(
        iterator.results_.values,
        pd.DataFrame({"left": [1.0, float("nan")], "right": [2.0, 3.0]}, index=regions.index),
    )


def test_keyed_series_keep_two_level_columns_and_repeated_parameter_names():
    regions = pd.DataFrame({"start": [0, 2], "end": [2, 4]}, index=pd.Index([7, 3], name="region_id"))
    iterator = RegionIterator(Values)
    for (_region, window), result in iterator.iterate(pd.DataFrame({"signal": [1, 3, 4, 8]}), regions):
        result.values = pd.concat(
            {
                "offset": pd.Series({"value": window.signal.iloc[0]}),
                "spread": pd.Series({"value": window.signal.iloc[-1] - window.signal.iloc[0]}),
            },
            names=["algorithm", "parameter"],
        )
    expected = pd.DataFrame(
        [[1, 2], [4, 4]],
        index=regions.index,
        columns=pd.MultiIndex.from_tuples([("offset", "value"), ("spread", "value")], names=["algorithm", "parameter"]),
    )
    assert_frame_equal(iterator.results_.values, expected)
