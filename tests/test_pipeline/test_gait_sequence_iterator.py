from dataclasses import dataclass

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gaitlink.pipeline import GsIterator, create_aggregate_df, iter_gs


class TestGsIterationFunc:
    def test_simple(self):
        dummy_data = pd.DataFrame({"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        dummy_sections = pd.DataFrame({"start": [0, 5], "end": [5, 10], "s_id": ["s1", "s2"]}).set_index("s_id")

        iterator = iter_gs(dummy_data, dummy_sections)

        first = next(iterator)
        assert first[0] == ("s1", 0, 5)
        assert_frame_equal(first[1], pd.DataFrame({"data": [1, 2, 3, 4, 5]}, index=[0, 1, 2, 3, 4]))

        second = next(iterator)
        assert second[0] == ("s2", 5, 10)
        assert_frame_equal(second[1], pd.DataFrame({"data": [6, 7, 8, 9, 10]}, index=[5, 6, 7, 8, 9]))


class TestGsIterator:
    def test_no_agg(self):
        @dataclass
        class DummyResultType:
            n_samples: int
            s_id: str

        dummy_data = pd.DataFrame({"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        dummy_sections = pd.DataFrame({"start": [0, 5], "end": [5, 10], "s_id": ["s1", "s2"]}).set_index("s_id")

        iterator = GsIterator(DummyResultType)

        for ((s_id, *_), d), r in iterator.iterate(dummy_data, dummy_sections):
            r.n_samples = len(d)
            r.s_id = s_id

        raw_results = iterator.raw_results_
        assert len(raw_results) == 2
        assert raw_results[0].n_samples == 5
        assert raw_results[0].s_id == "s1"
        assert raw_results[1].n_samples == 5
        assert raw_results[1].s_id == "s2"

        assert iterator.n_samples_ == [5, 5]
        assert iterator.s_id_ == ["s1", "s2"]

        inputs, input_dfs = zip(*iterator.inputs_)
        assert inputs == (("s1", 0, 5), ("s2", 5, 10))
        assert_frame_equal(input_dfs[0], pd.DataFrame({"data": [1, 2, 3, 4, 5]}, index=[0, 1, 2, 3, 4]))
        assert_frame_equal(input_dfs[1], pd.DataFrame({"data": [6, 7, 8, 9, 10]}, index=[5, 6, 7, 8, 9]))

    def test_agg(self):
        @dataclass
        class DummyResultType:
            n_samples: int
            s_id: str

        aggregations = [("n_samples", lambda _, r: sum(r))]

        dummy_data = pd.DataFrame({"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        dummy_sections = pd.DataFrame({"start": [0, 5], "end": [5, 10], "s_id": ["s1", "s2"]}).set_index("s_id")

        iterator = GsIterator(DummyResultType, aggregations=aggregations)

        for (_, d), r in iterator.iterate(dummy_data, dummy_sections):
            r.n_samples = len(d)

        assert iterator.n_samples_ == 10

    def test_default_agg_offsets(self):
        dummy_sections = pd.DataFrame({"start": [0, 5], "end": [5, 10], "wb_id": ["s1", "s2"]}).set_index("wb_id")
        dummy_data = pd.DataFrame({"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        iterator = GsIterator()

        for (s, d), r in iterator.iterate(dummy_data, dummy_sections):
            # We set the values "relative" to the start of the section, but expect the aggregation to make the values
            # relative to the start of the recording.
            r.ic_list = pd.DataFrame({"ic": [0, s.end - s.start]}).rename_axis("s_id")

        assert_frame_equal(
            iterator.results_.ic_list,
            pd.DataFrame(
                {"ic": [0, 5, 5, 10]},
                index=pd.MultiIndex.from_tuples([("s1", 0), ("s1", 1), ("s2", 0), ("s2", 1)], names=["wb_id", "s_id"]),
            ),
        )


class TestAggregateDf:
    """Some isolated tests for the create_aggregate_df function."""

    @pytest.mark.parametrize("fix_cols", [[], ["start", "end"], ["start"]])
    def test_basic_aggregation(self, fix_cols):
        aggregate_df = create_aggregate_df(fix_gs_offset_cols=fix_cols)

        example_sequences = pd.DataFrame({"start": [0, 5], "end": [5, 10]}, index=["s1", "s2"]).rename_axis(
            index="wb_id"
        )
        example_inputs = [(s, pd.DataFrame()) for s in example_sequences.reset_index().itertuples(index=False)]

        example_results = [
            pd.DataFrame({"start": [1, 3], "end": [3, 5]}),
            pd.DataFrame({"start": [1, 3], "end": [3, 5]}),
        ]

        aggregated = aggregate_df(example_inputs, example_results)

        expected_start = [1, 3, 6, 8] if "start" in fix_cols else [1, 3, 1, 3]
        expected_end = [3, 5, 8, 10] if "end" in fix_cols else [3, 5, 3, 5]

        assert_frame_equal(
            aggregated,
            pd.DataFrame(
                {"start": expected_start, "end": expected_end},
                index=pd.MultiIndex.from_product((example_sequences.index, [0, 1]), names=["wb_id", None]),
            ),
        )

    def test_wrong_datatype_input(self):
        example_sequences = pd.DataFrame({"start": [0, 5], "end": [5, 10]}, index=["s1", "s2"]).rename_axis(
            index="wb_id"
        )
        example_inputs = [(s, pd.DataFrame()) for s in example_sequences.reset_index().itertuples(index=False)]

        with pytest.raises(TypeError):
            create_aggregate_df()(example_inputs, "not a df")
