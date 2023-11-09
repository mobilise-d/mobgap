from dataclasses import dataclass

import pandas as pd
from pandas.testing import assert_frame_equal

from gaitlink.pipeline import GsIterator, iter_gs


class TestGsIterationFunc:
    def test_simple(self):
        dummy_data = pd.DataFrame({"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        dummy_sections = pd.DataFrame({"start": [0, 5], "end": [5, 10], "s_id": ["s1", "s2"]}).set_index("s_id")

        iterator = iter_gs(dummy_data, dummy_sections)

        first = next(iterator)
        assert first[0] == "s1"
        assert_frame_equal(first[1], pd.DataFrame({"data": [1, 2, 3, 4, 5]}, index=[0, 1, 2, 3, 4]))

        second = next(iterator)
        assert second[0] == "s2"
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

        for (s_id, d), r in iterator.iterate(dummy_data, dummy_sections):
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

        input_sids, input_dfs = zip(*iterator.inputs_)
        assert input_sids == ("s1", "s2")
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

        for (s_id, d), r in iterator.iterate(dummy_data, dummy_sections):
            r.n_samples = len(d)
            r.s_id = s_id

        assert iterator.n_samples_ == 10
