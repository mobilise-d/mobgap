from dataclasses import dataclass, replace
from typing import Any

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mobgap.pipeline import GsIterator, create_aggregate_df, iter_gs
from mobgap.pipeline._gs_iterator import Region, RegionDataTuple
from mobgap.pipeline._overwrite_typed_iterator import TypedIteratorResultTuple


class TestGsIterationFunc:
    def test_simple(self):
        dummy_data = pd.DataFrame({"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        dummy_sections = pd.DataFrame({"start": [0, 5], "end": [5, 10], "gs_id": ["s1", "s2"]}).set_index("gs_id")

        iterator = iter_gs(dummy_data, dummy_sections)

        first = next(iterator)
        assert first[0][:3] == ("s1", 0, 5)
        assert_frame_equal(first[1], pd.DataFrame({"data": [1, 2, 3, 4, 5]}, index=[0, 1, 2, 3, 4]))

        second = next(iterator)
        assert second[0][:3] == ("s2", 5, 10)
        assert_frame_equal(second[1], pd.DataFrame({"data": [6, 7, 8, 9, 10]}, index=[5, 6, 7, 8, 9]))

    @pytest.mark.parametrize(
        "col_name, as_index", [("wb_id", True), ("wb_id", False), ("gs_id", True), ("gs_id", False), ("bla", True)]
    )
    def test_correct_dtype(self, col_name, as_index):
        dummy_data = pd.DataFrame({"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        dummy_sections = pd.DataFrame({"start": [0, 5], "end": [5, 10], col_name: ["s1", "s2"]})
        if as_index:
            dummy_sections = dummy_sections.set_index(col_name)

        iterator = iter_gs(dummy_data, dummy_sections)

        first = next(iterator)[0]
        assert isinstance(first, Region)
        assert first.id == "s1"
        assert first.id_origin == col_name

    def test_custom_id_col(self):
        custom_name = "custom_id"
        dummy_data = pd.DataFrame({"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        dummy_sections = pd.DataFrame({"start": [0, 5], "end": [5, 10], custom_name: ["s1", "s2"]})

        # Just as dummy, check that this would fail without specifying the custom id column
        with pytest.raises(ValueError):
            list(iter_gs(dummy_data, dummy_sections))

        iterator = iter_gs(dummy_data, dummy_sections, id_col=custom_name)

        first = next(iterator)[0]
        assert first.id == "s1"
        assert first.id_origin == custom_name


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
        assert raw_results[0].result.n_samples == 5
        assert raw_results[0].result.s_id == "s1"
        assert raw_results[1].result.n_samples == 5
        assert raw_results[1].result.s_id == "s2"

        assert iterator.results_.n_samples == [5, 5]
        assert iterator.results_.s_id == ["s1", "s2"]

        inputs, input_dfs = zip(*(v.input for v in iterator.raw_results_))
        assert inputs == (("s1", 0, 5, "gs_id"), ("s2", 5, 10, "gs_id"))
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

        aggregations = [
            ("n_samples", lambda r: sum(v.result for v in GsIterator.filter_iterator_results(r, "n_samples")))
        ]

        dummy_data = pd.DataFrame({"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        dummy_sections = pd.DataFrame({"start": [0, 5], "end": [5, 10], "gs_id": ["s1", "s2"]}).set_index("gs_id")

        iterator = GsIterator(DummyResultType, aggregations=aggregations)

        for (_, d), r in iterator.iterate(dummy_data, dummy_sections):
            r.n_samples = len(d)

        assert iterator.results_.n_samples == 10

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
    @pytest.mark.parametrize("parent_region", [None, Region("0", 5, 10, "gs_id"), Region("1", 10, 10, "wb_id")])
    def test_basic_aggregation(self, fix_cols, parent_region):
        aggregate_df = create_aggregate_df("test", fix_offset_cols=fix_cols)

        example_sequences = pd.DataFrame({"start": [0, 5], "end": [5, 10]}, index=["s1", "s2"]).rename_axis(index="id")
        example_inputs = [
            RegionDataTuple(Region(*s, "gs_id"), pd.DataFrame())
            for s in example_sequences.reset_index().itertuples(index=False)
        ]

        @dataclass
        class DummyResultType:
            test: Any

        context = {} if parent_region is None else {"parent_region": parent_region}
        iter_type = "__main__" if parent_region is None else "__sub_iter__"

        example_results = [
            TypedIteratorResultTuple(
                iter_type, example_inputs[0], DummyResultType(pd.DataFrame({"start": [1, 3], "end": [3, 5]})), context
            ),
            TypedIteratorResultTuple(
                iter_type, example_inputs[1], DummyResultType(pd.DataFrame({"start": [1, 3], "end": [3, 5]})), context
            ),
        ]

        aggregated = aggregate_df(example_results)

        expected_start = [1, 3, 6, 8] if "start" in fix_cols else [1, 3, 1, 3]
        expected_end = [3, 5, 8, 10] if "end" in fix_cols else [3, 5, 3, 5]

        expected_df = pd.DataFrame(
            {"start": expected_start, "end": expected_end},
        )
        index = pd.MultiIndex.from_product((example_sequences.index, [0, 1]), names=["gs_id", None])
        if parent_region:
            expected_df[fix_cols] += parent_region.start
            index = pd.MultiIndex.from_product(
                ([parent_region.id], example_sequences.index, [0, 1]), names=[parent_region.id_origin, "gs_id", None]
            )

        expected_df.index = index

        assert_frame_equal(aggregated, expected_df)

    @pytest.mark.parametrize("fix_index_offset", [True, False])
    def test_index_offset(self, fix_index_offset):
        example_sequences = pd.DataFrame({"start": [1, 5], "end": [5, 10]}, index=["s1", "s2"]).rename_axis(index="id")
        example_inputs = [
            RegionDataTuple(Region(*s, "gs_id"), pd.DataFrame())
            for s in example_sequences.reset_index().itertuples(index=False)
        ]

        @dataclass
        class DummyResultType:
            test: Any

        example_results = [
            TypedIteratorResultTuple(
                "__main__", example_inputs[0], DummyResultType(pd.DataFrame({"value": [1, 3]}, index=[1, 2])), {}
            ),
            TypedIteratorResultTuple(
                "__main__", example_inputs[1], DummyResultType(pd.DataFrame({"value": [2, 4]}, index=[3, 4])), {}
            ),
        ]

        aggregate_df = create_aggregate_df("test", fix_offset_index=fix_index_offset)
        aggregated = aggregate_df(example_results)

        if fix_index_offset:
            expected_index = [2, 3, 8, 9]
        else:
            expected_index = [1, 2, 3, 4]

        expected_index = pd.MultiIndex.from_tuples(
            zip(*(["s1", "s1", "s2", "s2"], expected_index)), names=["gs_id", None]
        )
        assert_frame_equal(
            aggregated,
            pd.DataFrame(
                {"value": [1, 3, 2, 4]},
                index=expected_index,
            ),
        )

    def test_agg_with_not_set(self):
        example_sequences = pd.DataFrame(
            {"start": [0, 5, 10], "end": [5, 10, 15]}, index=["s1", "s2", "s3"]
        ).rename_axis(index="id")
        example_inputs = [
            RegionDataTuple(Region(*s, "gs_id"), pd.DataFrame())
            for s in example_sequences.reset_index().itertuples(index=False)
        ]

        @dataclass
        class DummyResultType:
            test1: Any
            test2: Any

        base_result = DummyResultType(
            pd.DataFrame({"start": [1, 3], "end": [3, 5]}), pd.DataFrame({"start": [1, 3], "end": [3, 5]})
        )

        example_results = [
            TypedIteratorResultTuple("__main__", example_inputs[0], base_result, {}),
            TypedIteratorResultTuple(
                "__main__", example_inputs[1], replace(base_result, test1=GsIterator.NULL_VALUE), {}
            ),
            TypedIteratorResultTuple(
                "__main__", example_inputs[2], replace(base_result, test2=GsIterator.NULL_VALUE), {}
            ),
        ]

        agg_test1 = create_aggregate_df("test1", fix_offset_cols=[])
        test_1_agg = agg_test1(example_results)

        assert set(test_1_agg.index.get_level_values("gs_id")) == {"s1", "s3"}

        agg_test2 = create_aggregate_df("test2", fix_offset_cols=[])
        test_2_agg = agg_test2(example_results)

        assert set(test_2_agg.index.get_level_values("gs_id")) == {"s1", "s2"}

    # TODO: Add tests for subiteration
    # TODO: Add tests for empty wbs.
