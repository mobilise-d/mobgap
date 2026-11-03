import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mobgap.re_orientation import get_orientation_for_sequences


def test_selects_opaque_values_and_explicit_default_without_copying():
    correction = {"swap": ["x", "y"]}
    default = {"swap": []}
    sections = pd.DataFrame(
        {"start": [20, 0], "end": [30, 10], "orientation": [object(), correction], "correction_applied": [False, True]},
        index=pd.MultiIndex.from_tuples([("a", 2), ("a", 1)], names=["recording", "section"]),
    )
    sequences = pd.DataFrame(
        {"start": [25, 0, 2, 10, 30, -5], "end": [35, 10, 7, 20, 40, 0]},
        index=pd.MultiIndex.from_tuples([("b", i) for i in [4, 2, 7, 0, 5, 6]], names=["trial", "wb_id"]),
    )
    original_sequences, original_sections = sequences.copy(deep=True), sections.copy(deep=True)
    result = get_orientation_for_sequences(sequences, sections, default=default)
    assert result.name == "orientation"
    assert result.index.equals(sequences.index)
    for actual, expected in zip(result, [default, correction, correction, default, default, default], strict=True):
        assert actual is expected
    assert_frame_equal(sequences, original_sequences)
    assert_frame_equal(sections, original_sections)


@pytest.mark.parametrize("bounds", [(-5, 5), (5, 15), (-5, 15), (0, 15), (-5, 10)])
def test_any_partial_overlap_with_required_correction_raises(bounds):
    sections = pd.DataFrame({"start": [0], "end": [10], "orientation": ["swap"], "correction_applied": [True]})
    sequences = pd.DataFrame([bounds], columns=["start", "end"])
    with pytest.raises(ValueError, match="partially overlaps"):
        get_orientation_for_sequences(sequences, sections, default="identity")


def test_partial_overlap_is_rejected_even_when_another_required_section_contains_query():
    sections = pd.DataFrame(
        {"start": [0, 5], "end": [30, 15], "orientation": ["outer", "inner"], "correction_applied": [True, True]}
    )
    with pytest.raises(ValueError, match="partially overlaps"):
        get_orientation_for_sequences(pd.DataFrame({"start": [0], "end": [10]}), sections, default="identity")


def test_ambiguous_nominal_containment_raises():
    sections = pd.DataFrame(
        {"start": [0, 2], "end": [10, 8], "orientation": [None, None], "correction_applied": [False, False]}
    )
    with pytest.raises(ValueError, match="multiple regions"):
        get_orientation_for_sequences(pd.DataFrame({"start": [3], "end": [5]}), sections, default="identity")


def test_empty_sections_and_sequences():
    sections = pd.DataFrame(
        {
            "start": pd.Series(dtype=int),
            "end": pd.Series(dtype=int),
            "orientation": [],
            "correction_applied": pd.Series(dtype=bool),
        }
    )
    sequences = pd.DataFrame({"start": [0], "end": [10]})
    default = object()
    assert get_orientation_for_sequences(sequences, sections, default=default).iat[0] is default
    result = get_orientation_for_sequences(sequences.iloc[:0], sections, default=default)
    assert result.empty
    assert result.name == "orientation"
    assert result.index.equals(sequences.iloc[:0].index)


def test_tuple_identifiers_in_a_single_level_index_preserve_array_payload():
    import numpy as np

    correction = np.array([[0, 1], [1, 0]])
    sections = pd.DataFrame(
        {"start": [0], "end": [10], "orientation": [correction], "correction_applied": [True]},
        index=pd.Index([("session", 1)], tupleize_cols=False),
    )
    sequences = pd.DataFrame({"start": [2], "end": [7]})
    assert get_orientation_for_sequences(sequences, sections, default=None).iloc[0] is correction
