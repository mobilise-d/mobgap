from dataclasses import dataclass

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gaitlink.pipeline import GsIterator, create_aggregate_df, iter_gs
from gaitlink.pipeline._gs_iterator import GaitSequence, WalkingBout


class TestGsIterationFunc:
    def test_simple(self):
        dummy_data = pd.DataFrame({"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        dummy_sections = pd.DataFrame({"start": [0, 5], "end": [5, 10], "gs_id": ["s1", "s2"]}).set_index("gs_id")

        iterator = iter_gs(dummy_data, dummy_sections)

        first = next(iterator)
        assert first[0] == ("s1", 0, 5)
        assert_frame_equal(first[1], pd.DataFrame({"data": [1, 2, 3, 4, 5]}, index=[0, 1, 2, 3, 4]))

        second = next(iterator)
        assert second[0] == ("s2", 5, 10)
        assert_frame_equal(second[1], pd.DataFrame({"data": [6, 7, 8, 9, 10]}, index=[5, 6, 7, 8, 9]))

    @pytest.mark.parametrize("col_name", ["wb_id", "gs_id"])
    @pytest.mark.parametrize("as_index", [True, False])
    def test_correct_dtype(self, col_name, as_index):
        dummy_data = pd.DataFrame({"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        dummy_sections = pd.DataFrame({"start": [0, 5], "end": [5, 10], col_name: ["s1", "s2"]})
        if as_index:
            dummy_sections = dummy_sections.set_index(col_name)

        iterator = iter_gs(dummy_data, dummy_sections)

        first = next(iterator)[0]
        if col_name == "wb_id":
            assert isinstance(first, WalkingBout)
            assert first.wb_id == first.id == "s1"
        else:
            assert isinstance(first, GaitSequence)
            assert first.gs_id == first.id == "s1"


class TestGsIterator:
    def test_no_agg(self):
        @dataclass
        class DummyResultType:
            n_samples: int
            s_id: str

        dummy_data = pd.DataFrame({"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        dummy_sections = pd.DataFrame({"start": [0, 5], "end": [5, 10], "gs_id": ["s1", "s2"]}).set_index("gs_id")

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
        assert_frame_equal(
            input_dfs[1],
            pd.DataFrame({"data": [6, 7, 8, 9, 10]}, index=[5, 6, 7, 8, 9]),
        )

    def test_agg(self):
        @dataclass
        class DummyResultType:
            n_samples: int
            s_id: str

        aggregations = [("n_samples", lambda _, r: sum(r))]

        dummy_data = pd.DataFrame({"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        dummy_sections = pd.DataFrame({"start": [0, 5], "end": [5, 10], "gs_id": ["s1", "s2"]}).set_index("gs_id")

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
            r.ic_list = pd.DataFrame({"ic": [0, s.end - s.start]}).rename_axis("step_id")

        assert_frame_equal(
            iterator.results_.ic_list,
            pd.DataFrame(
                {"ic": [0, 5, 5, 10]},
                index=pd.MultiIndex.from_tuples(
                    [("s1", 0), ("s1", 1), ("s2", 0), ("s2", 1)],
                    names=["wb_id", "step_id"],
                ),
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

    @pytest.mark.parametrize("fix_index_offset", [True, False])
    def test_index_offset(self, fix_index_offset):
        example_sequences = pd.DataFrame({"start": [1, 5], "end": [5, 10]}, index=["s1", "s2"]).rename_axis(
            index="wb_id"
        )
        example_inputs = [(s, pd.DataFrame()) for s in example_sequences.reset_index().itertuples(index=False)]

        example_results = [
            pd.DataFrame({"value": [1, 3]}, index=[1, 2]),
            pd.DataFrame({"value": [2, 4]}, index=[3, 4]),
        ]

        aggregate_df = create_aggregate_df(fix_gs_offset_index=fix_index_offset)
        aggregated = aggregate_df(example_inputs, example_results)

        if fix_index_offset:
            expected_index = [2, 3, 8, 9]
        else:
            expected_index = [1, 2, 3, 4]

        expected_index = pd.MultiIndex.from_tuples(
            zip(*(["s1", "s1", "s2", "s2"], expected_index)), names=["wb_id", None]
        )
        assert_frame_equal(
            aggregated,
            pd.DataFrame(
                {"value": [1, 3, 2, 4]},
                index=expected_index,
            ),
        )

    def test_wrong_datatype_input(self):
        example_sequences = pd.DataFrame({"start": [0, 5], "end": [5, 10]}, index=["s1", "s2"]).rename_axis(
            index="wb_id"
        )
        example_inputs = [(s, pd.DataFrame()) for s in example_sequences.reset_index().itertuples(index=False)]

        with pytest.raises(TypeError):
            create_aggregate_df()(example_inputs, "not a df")
