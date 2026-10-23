import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.wba import MaxBreakCriteria, WbAssembly
from tests.test_wba.conftest import window


class TestWBAssemblyMeta(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = WbAssembly

    @pytest.fixture
    def after_action_instance(self, naive_stride_list) -> WbAssembly:
        return self.ALGORITHM_CLASS([("break", MaxBreakCriteria(3))]).assemble(naive_stride_list, sampling_rate_hz=1)


class TestWBAssembly:
    @pytest.mark.parametrize("rules", ([("test", "something_invalid")]))
    def test_invalid_rules(self, rules):
        with pytest.raises(ValueError):
            WbAssembly(rules).assemble(pd.DataFrame([]), sampling_rate_hz=1)

    def test_assemble_no_rules(self):
        stride_list = pd.DataFrame.from_records([window(0, 0) for _ in range(10)]).set_index("s_id")

        wba = WbAssembly()
        wba.assemble(stride_list, sampling_rate_hz=1)

        assert_frame_equal(next(iter(wba.wbs_.values())), stride_list)
        assert len(wba.wbs_) == 1

    @pytest.mark.parametrize(
        ("contacts", "expected"),
        [
            (pd.Series([22, 0, 2, 2, 20, 21, 9]), [3, 3]),
            (pd.Series([], dtype="int64"), [0, 0]),
            (pd.Series([22, None, 0, 2, 2, 20], dtype="Int64"), [3, 2]),
        ],
    )
    def test_raw_contact_counts(self, contacts, expected):
        strides = pd.DataFrame({"start": [0, 1, 20, 21], "end": [1, 2, 21, 22]}).rename_axis("s_id")
        wba = WbAssembly([("break", MaxBreakCriteria(3))]).assemble(
            strides, raw_initial_contacts=contacts.to_frame("ic"), sampling_rate_hz=1
        )

        result = wba.wb_meta_parameters_["n_raw_initial_contacts"]
        assert_series_equal(result, pd.Series(expected, index=result.index, name=result.name, dtype="Int64"))

    def test_raw_contact_counts_reflect_input_mutations(self):
        strides = pd.DataFrame({"start": [0, 1, 20, 21], "end": [1, 2, 21, 22]}).rename_axis("s_id")
        contacts = pd.DataFrame({"ic": [0, 1, 20]})
        wba = WbAssembly([("break", MaxBreakCriteria(3))]).assemble(
            strides, raw_initial_contacts=contacts, sampling_rate_hz=1
        )
        assert wba.wb_meta_parameters_["n_raw_initial_contacts"].tolist() == [2, 1]

        contacts.loc[0, "ic"] = 21

        assert wba.wb_meta_parameters_["n_raw_initial_contacts"].tolist() == [1, 2]

    def test_raw_contact_counts_preserve_large_integer_boundaries(self):
        offset = 2**53
        strides = pd.DataFrame({"start": [offset + 1], "end": [offset + 2]}).rename_axis("s_id")
        contacts = pd.DataFrame({"ic": [offset + 3, offset, offset + 1, offset + 2]})
        wba = WbAssembly(rules=None).assemble(strides, raw_initial_contacts=contacts, sampling_rate_hz=1)

        assert wba.wb_meta_parameters_["n_raw_initial_contacts"].tolist() == [2]
