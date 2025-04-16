import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
from mobgap.data import LabExampleDataset
from mobgap.pipeline import GsIterator
from mobgap.stride_length import SlZijlstra
from mobgap.utils.conversions import to_body_frame


class TestMetaSlZijlstra(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = SlZijlstra

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().calculate(
            data=pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS),
            initial_contacts=pd.DataFrame({"ic": np.arange(0, 100, 5)}),
            sensor_height_m=0.95,
            sampling_rate_hz=100.0,
        )


class TestSlZijlstra:
    """Tests for SlZijlstra.

    We just test the happy path and some potential edgecases.
    If people run into bugs when changing parameters, we can add more tests.
    """

    def test_not_enough_ics(self):
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        initial_contacts = pd.DataFrame({"ic": np.arange(0, 100, 5)})
        # We only keep the first IC -> Not possible to calculate step length
        initial_contacts = initial_contacts.iloc[:1]

        with pytest.warns(UserWarning) as w:
            sl_zijlstra = SlZijlstra().calculate(
                data=data, initial_contacts=initial_contacts, sensor_height_m=0.95, sampling_rate_hz=100
            )

        assert len(w) == 2
        assert "Can not calculate step length with only one or zero initial contacts" in w.list[1].message.args[0]
        assert (
            "Usually we assume that gait sequences are cut to the first and last detected initial"
            in w.list[0].message.args[0]
        )
        assert len(sl_zijlstra.stride_length_per_sec_) == np.ceil(data.shape[0] / 100)
        assert sl_zijlstra.stride_length_per_sec_["stride_length_m"].isna().all()
        assert sl_zijlstra.step_length_per_sec_["step_length_m"].isna().all()
        assert len(sl_zijlstra.raw_step_length_per_step_) == 0

    def test_raise_non_sorted_ics(self):
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        initial_contacts = pd.DataFrame({"ic": np.arange(0, 100, 5)})
        # We shuffle the ICs
        initial_contacts = initial_contacts.sample(frac=1, random_state=2)

        with pytest.raises(ValueError) as e:
            SlZijlstra().calculate(
                data=data, initial_contacts=initial_contacts, sensor_height_m=0.95, sampling_rate_hz=40.0
            )

        assert "Initial contacts must be sorted" in str(e.value)

    @pytest.mark.parametrize("n_ics", [2, 3, 4])
    def test_small_n_ics(self, n_ics):
        """We test that things work with a small number of ICs.

        This could likely trigger some edge cases in the code.
        """
        # Load real subject data
        dp = LabExampleDataset(reference_system="INDIP").get_subset(
            cohort="HA", participant_id="001", test="Test5", trial="Trial1"
        )
        reference_gs = dp.reference_parameters_relative_to_wb_.wb_list
        reference_ic = dp.reference_parameters_relative_to_wb_.ic_list
        gs_id = reference_gs.index[0]
        data_in_gs = dp.data["LowerBack"].iloc[reference_gs.start.iloc[0] : reference_gs.end.iloc[0]]
        ics_in_gs = reference_ic.loc[gs_id]  # reference initial contacts
        initial_contacts = ics_in_gs.iloc[:n_ics]
        sensor_height = dp.participant_metadata["sensor_height_m"]

        sl_zijlstra = SlZijlstra().calculate(
            data=to_body_frame(data_in_gs),
            initial_contacts=initial_contacts,
            sensor_height_m=sensor_height,
            sampling_rate_hz=100.0,
        )

        assert len(sl_zijlstra.stride_length_per_sec_) == np.ceil(data_in_gs.shape[0] / 100)
        # We just test that not all values are NaN
        assert not sl_zijlstra.stride_length_per_sec_["stride_length_m"].isna().all()
        assert_array_equal(
            sl_zijlstra.step_length_per_sec_["step_length_m"] * 2, sl_zijlstra.stride_length_per_sec_["stride_length_m"]
        )
        assert len(sl_zijlstra.raw_step_length_per_step_) == n_ics - 1

    def test_outlier_data(self):
        """Tests if the function handles outliers in accelerometer data."""
        # Load real subject data
        dp = LabExampleDataset(reference_system="INDIP").get_subset(
            cohort="HA", participant_id="001", test="Test5", trial="Trial1"
        )
        reference_gs = dp.reference_parameters_relative_to_wb_.wb_list
        reference_ic = dp.reference_parameters_relative_to_wb_.ic_list
        gs_id = reference_gs.index[0]
        data_in_gs = to_body_frame(
            dp.data["LowerBack"].iloc[reference_gs.start.iloc[0] : reference_gs.end.iloc[0]].copy()
        )
        ics_in_gs = reference_ic.loc[gs_id]  # reference initial contacts
        sensor_height = dp.participant_metadata["sensor_height_m"]

        # Introduce outliers in specific axis
        data_in_gs.loc[100, "acc_is"] = 10  # Very high value on x-axis
        data_in_gs.loc[400, "acc_is"] = -5  # Very low value on x-axis

        # Calculate stride lengths
        sl_zijlstra = SlZijlstra().calculate(
            data=data_in_gs, initial_contacts=ics_in_gs, sensor_height_m=sensor_height, sampling_rate_hz=100
        )

        # Assert that outliers don't cause unrealistic stride lengths
        assert not np.any(sl_zijlstra.stride_length_per_sec_["stride_length_m"] < 0)  # No negative stride lengths
        assert not np.any(
            sl_zijlstra.stride_length_per_sec_["stride_length_m"] > 5
        )  # Limit upper bound (adjust based on typical stride length)

    @pytest.mark.parametrize("noise_std", [0.1, 0.25, 0.5])
    def test_random_noise(self, noise_std):
        dp = LabExampleDataset(reference_system="INDIP").get_subset(
            cohort="HA", participant_id="001", test="Test5", trial="Trial1"
        )
        reference_gs = dp.reference_parameters_relative_to_wb_.wb_list
        reference_ic = dp.reference_parameters_relative_to_wb_.ic_list
        gs_id = reference_gs.index[0]
        data_in_gs_clean = to_body_frame(
            dp.data["LowerBack"].iloc[reference_gs.start.iloc[0] : reference_gs.end.iloc[0]]
        )
        ics_in_gs = reference_ic.loc[gs_id]  # reference initial contacts
        sensor_height = dp.participant_metadata["sensor_height_m"]
        # Add random white noise
        rng = np.random.default_rng(0)
        data_in_gs_noise = data_in_gs_clean + rng.normal(scale=noise_std, size=data_in_gs_clean.shape)

        # Calculate stride lengths with and without noise
        sl_zijlstra_clean = SlZijlstra().calculate(
            data=data_in_gs_clean,
            initial_contacts=ics_in_gs.copy(),
            sensor_height_m=sensor_height,
            sampling_rate_hz=100,
        )
        sl_zijlstra_noise = SlZijlstra().calculate(
            data=data_in_gs_noise, initial_contacts=ics_in_gs, sensor_height_m=sensor_height, sampling_rate_hz=100
        )

        # Compare stride lengths with and without noise
        # Here, we check the average difference is within a reasonable range
        average_difference = np.abs(
            sl_zijlstra_clean.stride_length_per_sec_["stride_length_m"]
            - sl_zijlstra_noise.stride_length_per_sec_["stride_length_m"]
        ).mean()
        assert average_difference < 0.05

    @pytest.mark.parametrize("n_ics", [0, 1])
    def test_no_ics_result_all_nan(self, n_ics):
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        initial_contacts = pd.DataFrame({"ic": np.linspace(0, 100, n_ics)})
        sampling_rate_hz = 100.0
        out = SlZijlstra().calculate(
            data, initial_contacts=initial_contacts, sensor_height_m=0.95, sampling_rate_hz=sampling_rate_hz
        )
        assert out.stride_length_per_sec_["stride_length_m"].isna().all()
        assert out.step_length_per_sec_["step_length_m"].isna().all()
        assert len(out.raw_step_length_per_step_) == np.clip(n_ics - 1, 0, None)
        assert len(out.stride_length_per_sec_) == np.ceil(len(data) / sampling_rate_hz)
        assert len(out.step_length_per_sec_) == np.ceil(len(data) / sampling_rate_hz)

    def test_regression_on_longer_data(self, snapshot):
        dp = LabExampleDataset(reference_system="INDIP").get_subset(
            cohort="HA", participant_id="001", test="Test11", trial="Trial1"
        )

        gs_iterator = GsIterator()

        ref_data = dp.reference_parameters_relative_to_wb_
        meta_data = dp.participant_metadata

        for (gs, data), r in gs_iterator.iterate(to_body_frame(dp.data_ss), ref_data.wb_list):
            sl = SlZijlstra().calculate(
                data,
                ref_data.ic_list.loc[gs.id],
                sampling_rate_hz=dp.sampling_rate_hz,
                **meta_data,
            )
            r.stride_length_per_sec = sl.stride_length_per_sec_

        snapshot.assert_match(gs_iterator.results_.stride_length_per_sec)
