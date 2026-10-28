import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from mobgap.utils.intervals import map_intervals_to_regions


def test_containment_preserves_composite_ids_and_query_order():
    sequences = pd.DataFrame(
        {"start": [22, 0, 5, 8, 10, 30], "end": [25, 10, 9, 10, 20, 31]},
        index=pd.MultiIndex.from_tuples(
            [("b", 3), ("a", 1), ("a", 4), ("b", 1), ("a", 2), ("c", 0)], names=["trial", "wb_id"]
        ),
    )
    regions = pd.DataFrame(
        {"start": [20, 0], "end": [30, 10]},
        index=pd.MultiIndex.from_tuples([("session", 2), ("session", 1)], names=["recording", "gs_id"]),
    )
    original_sequences, original_regions = sequences.copy(deep=True), regions.copy(deep=True)
    expected = pd.Series(
        [("session", 2), ("session", 1), ("session", 1), ("session", 1), pd.NA, pd.NA],
        index=sequences.index,
        dtype=object,
    )

    assert_series_equal(map_intervals_to_regions(sequences, regions), expected)
    assert_frame_equal(sequences, original_sequences)
    assert_frame_equal(regions, original_regions)


def test_ambiguous_containment_raises():
    sequences = pd.DataFrame({"start": [5], "end": [8]})
    regions = pd.DataFrame({"start": [0, 4], "end": [10, 9]}, index=["outer", "inner"])
    with pytest.raises(ValueError, match="multiple regions"):
        map_intervals_to_regions(sequences, regions)


@pytest.mark.parametrize("empty", ["sequences", "regions"])
def test_empty_inputs(empty):
    sequences = pd.DataFrame({"start": [1], "end": [2]}, index=pd.Index(["wb"], name="wb_id"))
    regions = pd.DataFrame({"start": [0], "end": [3]}, index=["gs"])
    if empty == "sequences":
        sequences = sequences.iloc[:0]
    else:
        regions = regions.iloc[:0]
    assert_series_equal(
        map_intervals_to_regions(sequences, regions), pd.Series(pd.NA, index=sequences.index, dtype=object)
    )


def test_region_start_and_end_both_determine_containment():
    regions = pd.DataFrame({"start": [5, 0, 2, 1], "end": [20, 8, 3, 4]}, index=["late", "early", "small", "short"])
    sequences = pd.DataFrame({"start": [0, 5, 8, 21], "end": [8, 15, 9, 22]}, index=[3, 1, 4, 2])
    assert_series_equal(
        map_intervals_to_regions(sequences, regions),
        pd.Series(["early", "late", "late", pd.NA], index=sequences.index, dtype=object),
    )
