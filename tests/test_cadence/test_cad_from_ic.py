import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.cadence import CadFromIc, CadFromIcDetector
from mobgap.consts import BF_SENSOR_COLS
from mobgap.data import LabExampleDataset
from mobgap.initial_contacts.base import BaseIcDetector
from mobgap.pipeline import GsIterator


class TestMetaCadFromIc(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = CadFromIc

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().calculate(
            pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS),
            initial_contacts=pd.DataFrame({"ic": np.arange(0, 100, 5)}),
            sampling_rate_hz=40.0,
        )


class TestMetaCadFromIcDetector(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = CadFromIcDetector

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS(silence_ic_warning=True).calculate(
            pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS),
            initial_contacts=pd.DataFrame({"ic": np.arange(0, 100, 5)}),
            sampling_rate_hz=40.0,
        )


# TODO: - Tests with non-uniform data
#       - Really test that we perform linear interpolation
#       - Test the smoothing


class TestCadFromIc:
    @pytest.mark.parametrize("sampling_rate_hz", [10.0, 20.0, 40.0])
    @pytest.mark.parametrize("fixed_step_size", [5, 10, 20])
    def test_naive(self, sampling_rate_hz, fixed_step_size):
        n_samples = 100

        data = pd.DataFrame(np.zeros((n_samples, 6)), columns=BF_SENSOR_COLS)
        initial_contacts = pd.DataFrame({"ic": np.arange(0, n_samples + 1, fixed_step_size)})
        data = data.iloc[: initial_contacts["ic"].iloc[-1]]

        cad = CadFromIc().calculate(data, initial_contacts=initial_contacts, sampling_rate_hz=sampling_rate_hz)
        cadence = cad.cadence_per_sec_

        expected_index = pd.Index(
            np.arange(0.5 * sampling_rate_hz, n_samples + 0.5 * sampling_rate_hz, sampling_rate_hz),
            name="sec_center_samples",
        ).astype("int64")

        assert len(cadence) == np.ceil(len(data) / sampling_rate_hz)
        assert_frame_equal(
            cadence,
            pd.DataFrame(
                {"cadence_spm": np.ones(len(cadence)) * 1 / (fixed_step_size / sampling_rate_hz) * 60},
                index=expected_index,
            ),
        )

    def test_large_gap_no_interpolation(self):
        sampling_rate_hz = 40.0
        fixed_step_size = 5
        n_samples = 300

        data = pd.DataFrame(np.zeros((n_samples, 6)), columns=BF_SENSOR_COLS)
        initial_contacts = pd.DataFrame({"ic": np.arange(0, n_samples, fixed_step_size)})
        max_gap_s = 1

        n_ics_to_drop = max_gap_s * 3 * sampling_rate_hz // fixed_step_size

        # We introduce a gap in the data that is not covered by the initial contacts
        initial_contacts = initial_contacts.drop(initial_contacts.index[2 : 2 + int(n_ics_to_drop)])

        data = data.iloc[: initial_contacts["ic"].iloc[-1]]

        cad = CadFromIc(max_interpolation_gap_s=max_gap_s).calculate(
            data, initial_contacts=initial_contacts, sampling_rate_hz=sampling_rate_hz
        )
        cadence = cad.cadence_per_sec_

        expected_index = pd.Index(
            np.arange(0.5 * sampling_rate_hz, n_samples + 0.5 * sampling_rate_hz, sampling_rate_hz),
            name="sec_center_samples",
        ).astype("int64")

        assert len(cadence) == np.ceil(len(data) / sampling_rate_hz)
        expected_output = np.ones(len(cadence)) * 1 / (fixed_step_size / sampling_rate_hz) * 60
        expected_output[1:3] = np.nan

        assert_frame_equal(cadence, pd.DataFrame({"cadence_spm": expected_output}, index=expected_index))

    def test_small_gap_interpolation(self):
        sampling_rate_hz = 40.0
        fixed_step_size = 5
        n_samples = 300

        data = pd.DataFrame(np.zeros((n_samples, 6)), columns=BF_SENSOR_COLS)
        initial_contacts = pd.DataFrame({"ic": np.arange(0, n_samples, fixed_step_size)})
        max_gap_s = 6

        # We drop less than max_gap_s -> should get interpolated
        n_secs_to_drop = max_gap_s * 0.5
        assert n_secs_to_drop > 2
        n_ics_to_drop = n_secs_to_drop * sampling_rate_hz // fixed_step_size

        # We introduce a gap in the data that is not covered by the initial contacts
        initial_contacts = initial_contacts.drop(initial_contacts.index[2 : 2 + int(n_ics_to_drop)])

        data = data.iloc[: initial_contacts["ic"].iloc[-1]]

        cad = CadFromIc(max_interpolation_gap_s=max_gap_s).calculate(
            data, initial_contacts=initial_contacts, sampling_rate_hz=sampling_rate_hz
        )
        cadence = cad.cadence_per_sec_

        expected_index = pd.Index(
            np.arange(0.5 * sampling_rate_hz, n_samples + 0.5 * sampling_rate_hz, sampling_rate_hz),
            name="sec_center_samples",
        ).astype("int64")

        assert len(cadence) == np.ceil(len(data) / sampling_rate_hz)
        expected_output = np.ones(len(cadence)) * 1 / (fixed_step_size / sampling_rate_hz) * 60
        assert_frame_equal(cadence, pd.DataFrame({"cadence_spm": expected_output}, index=expected_index))

    def test_no_extrapolation(self):
        # We also test the warning here
        sampling_rate_hz = 40.0
        fixed_step_size = 5
        n_samples = 300

        data = pd.DataFrame(np.zeros((n_samples, 6)), columns=BF_SENSOR_COLS)
        # We remove ICs at the beginning and end
        initial_contacts = pd.DataFrame(
            {
                "ic": np.arange(
                    2 * sampling_rate_hz,
                    n_samples - 2 * sampling_rate_hz,
                    fixed_step_size,
                )
            }
        )

        with pytest.warns(UserWarning) as w:
            cad = CadFromIc().calculate(data, initial_contacts=initial_contacts, sampling_rate_hz=sampling_rate_hz)
        cadence = cad.cadence_per_sec_

        assert len(w) == 1
        assert "gait sequences are cut to the first and last detected initial" in str(w[0])

        assert len(cadence) == np.ceil(len(data) / sampling_rate_hz)
        expected_output = np.ones(len(cadence)) * 1 / (fixed_step_size / sampling_rate_hz) * 60
        expected_output[0] = np.nan
        expected_output[-2:] = np.nan

        expected_index = pd.Index(
            np.arange(0.5 * sampling_rate_hz, n_samples + 0.5 * sampling_rate_hz, sampling_rate_hz),
            name="sec_center_samples",
        ).astype("int64")

        assert_frame_equal(cadence, pd.DataFrame({"cadence_spm": expected_output}, index=expected_index))

    def test_not_enough_ics(self):
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        initial_contacts = pd.DataFrame({"ic": np.arange(0, 100, 5)})
        # We only keep the first IC -> Not possible to calculate cadence
        initial_contacts = initial_contacts.iloc[:1]

        with pytest.warns(UserWarning) as w:
            cad = CadFromIc().calculate(data, initial_contacts=initial_contacts, sampling_rate_hz=40.0)

        assert len(w) == 2
        assert "Can not calculate cadence with only one or zero initial contacts" in str(w[1])

        assert len(cad.cadence_per_sec_) == np.ceil(len(data) / 40)
        assert cad.cadence_per_sec_["cadence_spm"].isna().all()

    @pytest.mark.parametrize("n_ics", [2, 3, 4])
    def test_small_n_ics(self, n_ics):
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        initial_contacts = pd.DataFrame({"ic": np.arange(0, 100, 5)})
        initial_contacts = initial_contacts.iloc[:n_ics]

        with pytest.warns(UserWarning) as w:
            cad = CadFromIc().calculate(data, initial_contacts=initial_contacts, sampling_rate_hz=40.0)

        assert len(w) == 1

        assert len(cad.cadence_per_sec_) == np.ceil(len(data) / 40)
        # We just test that not all values are NaN
        assert not cad.cadence_per_sec_["cadence_spm"].isna().all()

    def test_raise_non_sorted_ics(self):
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        initial_contacts = pd.DataFrame({"ic": np.arange(0, 100, 5)})
        # We shuffle the ICs
        initial_contacts = initial_contacts.sample(frac=1, random_state=2)

        with pytest.raises(ValueError) as e:
            CadFromIc().calculate(data, initial_contacts=initial_contacts, sampling_rate_hz=40.0)

        assert "Initial contacts must be sorted" in str(e.value)

    def test_no_ics_result_all_nan(self):
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        initial_contacts = pd.DataFrame({"ic": []})
        sampling_rate_hz = 40.0
        cad = CadFromIc().calculate(data, initial_contacts=initial_contacts, sampling_rate_hz=sampling_rate_hz)
        assert cad.cadence_per_sec_["cadence_spm"].isna().all()
        assert len(cad.cadence_per_sec_) == np.ceil(len(data) / sampling_rate_hz)

    def test_regression_on_longer_data(self, snapshot):
        dp = LabExampleDataset(reference_system="INDIP").get_subset(
            cohort="HA", participant_id="001", test="Test11", trial="Trial1"
        )

        gs_iterator = GsIterator()

        ref_data = dp.reference_parameters_relative_to_wb_

        for (gs, data), r in gs_iterator.iterate(dp.data_ss, ref_data.wb_list):
            cad = CadFromIc().calculate(
                data,
                initial_contacts=ref_data.ic_list.loc[gs.id],
                sampling_rate_hz=dp.sampling_rate_hz,
            )
            r.cadence_per_sec = cad.cadence_per_sec_

        snapshot.assert_match(gs_iterator.results_.cadence_per_sec)


class _DummyIcDetector(BaseIcDetector):
    def __init__(self, ics):
        self.ics = ics

    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_):
        self.ic_list_ = self.ics.copy()
        return self


class TestCadFromIcDetector:
    # Note these are bascially the same tests as for CadFromIc, but we use a "dummy" IcDetector that just returns some
    # predefined ICs.
    @pytest.mark.parametrize("sampling_rate_hz", [10.0, 20.0, 40.0])
    @pytest.mark.parametrize("fixed_step_size", [5, 10, 20])
    def test_naive(self, sampling_rate_hz, fixed_step_size):
        n_samples = 100

        data = pd.DataFrame(np.zeros((n_samples, 6)), columns=BF_SENSOR_COLS)
        initial_contacts = pd.DataFrame({"ic": np.arange(0, n_samples + 1, fixed_step_size)})
        data = data.iloc[: initial_contacts["ic"].iloc[-1]]

        icd = _DummyIcDetector(initial_contacts)
        cad = CadFromIcDetector(icd, silence_ic_warning=True).calculate(
            data, initial_contacts=initial_contacts, sampling_rate_hz=sampling_rate_hz
        )
        cadence = cad.cadence_per_sec_

        expected_index = pd.Index(
            np.arange(0.5 * sampling_rate_hz, n_samples + 0.5 * sampling_rate_hz, sampling_rate_hz),
            name="sec_center_samples",
        ).astype("int64")

        assert len(cadence) == np.ceil(len(data) / sampling_rate_hz)
        assert_frame_equal(
            cadence,
            pd.DataFrame(
                {"cadence_spm": np.ones(len(cadence)) * 1 / (fixed_step_size / sampling_rate_hz) * 60},
                index=expected_index,
            ),
        )

    def test_large_gap_no_interpolation(self):
        sampling_rate_hz = 40.0
        fixed_step_size = 5
        n_samples = 300

        data = pd.DataFrame(np.zeros((n_samples, 6)), columns=BF_SENSOR_COLS)
        initial_contacts = pd.DataFrame({"ic": np.arange(0, n_samples, fixed_step_size)})
        max_gap_s = 1

        n_ics_to_drop = max_gap_s * 3 * sampling_rate_hz // fixed_step_size

        # We introduce a gap in the data that is not covered by the initial contacts
        initial_contacts = initial_contacts.drop(initial_contacts.index[2 : 2 + int(n_ics_to_drop)])

        data = data.iloc[: initial_contacts["ic"].iloc[-1]]

        icd = _DummyIcDetector(initial_contacts)
        cad = CadFromIcDetector(icd, max_interpolation_gap_s=max_gap_s, silence_ic_warning=True).calculate(
            data, initial_contacts=initial_contacts, sampling_rate_hz=sampling_rate_hz
        )
        cadence = cad.cadence_per_sec_

        expected_index = pd.Index(
            np.arange(0.5 * sampling_rate_hz, n_samples + 0.5 * sampling_rate_hz, sampling_rate_hz),
            name="sec_center_samples",
        ).astype("int64")

        assert len(cadence) == np.ceil(len(data) / sampling_rate_hz)
        expected_output = np.ones(len(cadence)) * 1 / (fixed_step_size / sampling_rate_hz) * 60
        expected_output[1:3] = np.nan

        assert_frame_equal(cadence, pd.DataFrame({"cadence_spm": expected_output}, index=expected_index))

    def test_small_gap_interpolation(self):
        sampling_rate_hz = 40.0
        fixed_step_size = 5
        n_samples = 300

        data = pd.DataFrame(np.zeros((n_samples, 6)), columns=BF_SENSOR_COLS)
        initial_contacts = pd.DataFrame({"ic": np.arange(0, n_samples, fixed_step_size)})
        max_gap_s = 6

        # We drop less than max_gap_s -> should get interpolated
        n_secs_to_drop = max_gap_s * 0.5
        assert n_secs_to_drop > 2
        n_ics_to_drop = n_secs_to_drop * sampling_rate_hz // fixed_step_size

        # We introduce a gap in the data that is not covered by the initial contacts
        initial_contacts = initial_contacts.drop(initial_contacts.index[2 : 2 + int(n_ics_to_drop)])

        data = data.iloc[: initial_contacts["ic"].iloc[-1]]

        icd = _DummyIcDetector(initial_contacts)
        cad = CadFromIcDetector(icd, max_interpolation_gap_s=max_gap_s, silence_ic_warning=True).calculate(
            data, initial_contacts=initial_contacts, sampling_rate_hz=sampling_rate_hz
        )
        cadence = cad.cadence_per_sec_

        expected_index = pd.Index(
            np.arange(0.5 * sampling_rate_hz, n_samples + 0.5 * sampling_rate_hz, sampling_rate_hz),
            name="sec_center_samples",
        ).astype("int64")

        assert len(cadence) == np.ceil(len(data) / sampling_rate_hz)
        expected_output = np.ones(len(cadence)) * 1 / (fixed_step_size / sampling_rate_hz) * 60
        assert_frame_equal(cadence, pd.DataFrame({"cadence_spm": expected_output}, index=expected_index))

    def test_no_extrapolation(self):
        # We also test the warning here
        sampling_rate_hz = 40.0
        fixed_step_size = 5
        n_samples = 300

        data = pd.DataFrame(np.zeros((n_samples, 6)), columns=BF_SENSOR_COLS)
        # We remove ICs at the beginning and end
        initial_contacts = pd.DataFrame(
            {
                "ic": np.arange(
                    2 * sampling_rate_hz,
                    n_samples - 2 * sampling_rate_hz,
                    fixed_step_size,
                )
            }
        )

        icd = _DummyIcDetector(initial_contacts)
        cad = CadFromIcDetector(icd, silence_ic_warning=True).calculate(
            data, initial_contacts=initial_contacts, sampling_rate_hz=sampling_rate_hz
        )
        cadence = cad.cadence_per_sec_

        assert len(cadence) == np.ceil(len(data) / sampling_rate_hz)
        expected_output = np.ones(len(cadence)) * 1 / (fixed_step_size / sampling_rate_hz) * 60
        expected_output[0] = np.nan
        expected_output[-2:] = np.nan

        expected_index = pd.Index(
            np.arange(0.5 * sampling_rate_hz, n_samples + 0.5 * sampling_rate_hz, sampling_rate_hz),
            name="sec_center_samples",
        ).astype("int64")

        assert_frame_equal(cadence, pd.DataFrame({"cadence_spm": expected_output}, index=expected_index))

    def test_not_enough_ics(self):
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        initial_contacts = pd.DataFrame({"ic": np.arange(0, 100, 5)})
        # We only keep the first IC -> Not possible to calculate cadence
        initial_contacts = initial_contacts.iloc[:1]

        with pytest.warns(UserWarning) as w:
            icd = _DummyIcDetector(initial_contacts)
            cad = CadFromIcDetector(icd, silence_ic_warning=True).calculate(
                data, initial_contacts=initial_contacts, sampling_rate_hz=40.0
            )

        assert len(w) == 1
        assert "Can not calculate cadence with only one or zero initial contacts" in str(w[0])

        assert len(cad.cadence_per_sec_) == np.ceil(len(data) / 40)
        assert cad.cadence_per_sec_["cadence_spm"].isna().all()

    @pytest.mark.parametrize("n_ics", [2, 3, 4])
    def test_small_n_ics(self, n_ics):
        """We test that things work with a small number of ICs.

        This could likely trigger some edge cases in the code.
        """
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        initial_contacts = pd.DataFrame({"ic": np.arange(0, 100, 5)})
        initial_contacts = initial_contacts.iloc[:n_ics]

        with pytest.warns(UserWarning) as w:
            icd = _DummyIcDetector(initial_contacts)
            cad = CadFromIcDetector(icd).calculate(data, initial_contacts=initial_contacts, sampling_rate_hz=40.0)

        assert len(w) == 1

        assert len(cad.cadence_per_sec_) == np.ceil(len(data) / 40)
        # We just test that not all values are NaN
        assert not cad.cadence_per_sec_["cadence_spm"].isna().all()

    def test_raise_non_sorted_ics(self):
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        initial_contacts = pd.DataFrame({"ic": np.arange(0, 100, 5)})
        # We shuffle the ICs
        initial_contacts = initial_contacts.sample(frac=1, random_state=2)

        with pytest.raises(ValueError) as e:
            icd = _DummyIcDetector(initial_contacts)
            _ = CadFromIcDetector(icd).calculate(data, initial_contacts=initial_contacts, sampling_rate_hz=40.0)

        assert "Initial contacts must be sorted" in str(e.value)

    def test_no_ics_result_all_nan(self):
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        initial_contacts = pd.DataFrame({"ic": []})
        icd = _DummyIcDetector(initial_contacts)
        sampling_rate_hz = 40.0
        cad = CadFromIcDetector(icd).calculate(
            data, initial_contacts=initial_contacts, sampling_rate_hz=sampling_rate_hz
        )
        assert cad.cadence_per_sec_["cadence_spm"].isna().all()
        assert len(cad.cadence_per_sec_) == np.ceil(len(data) / sampling_rate_hz)

    def test_regression_on_longer_data(self, snapshot):
        dp = LabExampleDataset(reference_system="INDIP").get_subset(
            cohort="HA", participant_id="001", test="Test11", trial="Trial1"
        )

        gs_iterator = GsIterator()

        ref_data = dp.reference_parameters_relative_to_wb_

        for (gs, data), r in gs_iterator.iterate(dp.data_ss, ref_data.wb_list):
            cad = CadFromIcDetector(silence_ic_warning=True).calculate(
                data,
                initial_contacts=ref_data.ic_list.loc[gs.id],
                sampling_rate_hz=dp.sampling_rate_hz,
            )
            r.cadence_per_sec = cad.cadence_per_sec_

        snapshot.assert_match(gs_iterator.results_.cadence_per_sec)
