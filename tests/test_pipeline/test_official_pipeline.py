import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin
from typing_extensions import Self

from mobgap.data import LabExampleDataset
from mobgap.initial_contacts.base import BaseIcDetector
from mobgap.gait_sequences.base import BaseGsDetector
from mobgap.signal_based.base import BaseSDMOCalculator
from mobgap.laterality.base import BaseLRClassifier
from mobgap.pipeline import (
    GenericMobilisedPipeline,
    MobilisedPipelineHealthy,
    MobilisedPipelineImpaired,
    MobilisedPipelineUniversal,
)


class _EmptyIcDetector(BaseIcDetector):
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_kwargs) -> Self:
        self.ic_list_ = pd.DataFrame({"ic": pd.Series(dtype="int64")}).rename_axis("step_id")
        return self


class _PassthroughLrc(BaseLRClassifier):
    def predict(self, data: pd.DataFrame, ic_list: pd.DataFrame, *, sampling_rate_hz: float, **_kwargs) -> Self:
        self.ic_lr_list_ = ic_list.copy()
        self.ic_lr_list_["lr_label"] = pd.Series(
            pd.Categorical(["left"] * len(ic_list), categories=["left", "right"]),
            index=ic_list.index,
        )
        return self


class TestMetaBaseMobilisedPipeline(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = GenericMobilisedPipeline
    ONLY_DEFAULT_PARAMS = False

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS(**self.ALGORITHM_CLASS.PredefinedParameters.regular_walking).run(
            LabExampleDataset()[0]
        )


class TestMetaMobilisedPipelineHealth(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = MobilisedPipelineHealthy

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().run(LabExampleDataset()[0])


class TestMetaMobilisedPipelineImpaired(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = MobilisedPipelineImpaired

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().run(LabExampleDataset()[0])


class TestMetaMobilisedMetaPipeline(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = MobilisedPipelineUniversal

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().run(LabExampleDataset()[0])


class TestFullPipelineRegression:
    @pytest.mark.parametrize("pipeline", [MobilisedPipelineHealthy, MobilisedPipelineImpaired])
    def test_full_pipeline(self, pipeline, snapshot):
        # Test the full pipeline with a sample dataset (note, that this is really short, and does not cover all edge
        # cases, the pipeline is ready to handle)
        dataset = LabExampleDataset(reference_system="INDIP").get_subset(
            cohort="MS", participant_id="001", test="Test11", trial="Trial1"
        )
        result = pipeline().run(dataset)

        snapshot.assert_match(result.per_stride_parameters_, name="per_stride_parameters")
        snapshot.assert_match(result.per_wb_parameters_.drop(columns="rule_obj"), name="per_wb_parameters")
        snapshot.assert_match(result.aggregated_parameters_, name="aggregated_parameters")


class TestFullPipelineEdgeCases:
    @pytest.fixture
    def dataset(self):
        return LabExampleDataset(reference_system="INDIP").get_subset(
            cohort="MS", participant_id="001", test="Test11", trial="Trial1"
        )

    def test_pipeline_without_sdmo_calculation_exposes_none_results(self, dataset):
        result = MobilisedPipelineImpaired(sdmo_calculation=None).run(dataset)

        assert result.per_wb_signal_based_parameters_ is None
        assert result.per_turn_parameters_ is None
        assert result.sdmo_calculation_ is None
        assert result.sdmo_aggregation_ is None

    def test_pipeline_without_sdmo_aggregation_keeps_per_wb_results(self, dataset):
        result = MobilisedPipelineImpaired(sdmo_aggregation=None).run(dataset)

        assert not result.per_wb_signal_based_parameters_.empty
        assert result.sdmo_calculation_ is not None
        assert result.sdmo_aggregation_ is None

    def test_impaired_pipeline_handles_no_detected_ics(self, dataset):
        result = MobilisedPipelineImpaired(
            initial_contact_detection=_EmptyIcDetector(),
            laterality_classification=_PassthroughLrc(),
        ).run(dataset)

        assert result.raw_ic_list_.empty
        assert result.raw_per_sec_parameters_.empty
        assert result.raw_per_stride_parameters_.empty
        assert result.per_stride_parameters_.empty
        assert result.per_wb_parameters_.empty
        assert result.per_wb_signal_based_parameters_.empty
        assert result.sdmo_calculation_ is None
        assert result.sdmo_aggregation_ is None


def test_turn_results_retain_all_gait_sequence_detections_without_accepted_wbs():
    from mobgap.signal_based import MobilisedSDMO, TurnSDMO

    dataset = LabExampleDataset(reference_system="INDIP").get_subset(
        cohort="MS", participant_id="001", test="Test11", trial="Trial1"
    )
    result = MobilisedPipelineImpaired(
        initial_contact_detection=_EmptyIcDetector(),
        laterality_classification=_PassthroughLrc(),
        sdmo_calculation=MobilisedSDMO(calculators=(("turn", TurnSDMO()),)),
    ).run(dataset)
    assert result.per_wb_parameters_.empty
    assert len(result.raw_turn_list_) > 0
    pd.testing.assert_frame_equal(result.per_turn_parameters_[result.raw_turn_list_.columns], result.raw_turn_list_)
    assert result.per_turn_parameters_.wb_id.isna().all()
    assert not any(c.startswith("turn_") for c in result.per_wb_signal_based_parameters_)
    assert result.sdmo_aggregation_ is None


def test_turn_results_preserve_detection_intervals_and_associate_containing_wbs():
    dataset = LabExampleDataset(reference_system="INDIP").get_subset(
        cohort="MS", participant_id="001", test="Test11", trial="Trial1"
    )
    result = MobilisedPipelineImpaired().run(dataset)
    turns = result.per_turn_parameters_
    pd.testing.assert_frame_equal(turns[result.raw_turn_list_.columns], result.raw_turn_list_)
    assert turns.index.is_unique
    assert turns.wb_id.notna().any()
    for turn in turns.itertuples():
        wbs = result.per_wb_parameters_
        contained = wbs[(wbs.start <= turn.start) & (wbs.end >= turn.end)]
        if len(contained) == 1:
            assert turn.wb_id == contained.index[0]
        else:
            assert pd.isna(turn.wb_id)
    assert not any("turn_" in c for c in result.aggregated_parameters_)


def test_single_sdmo_calculator_remains_supported():
    from mobgap.signal_based import RMS

    dataset = LabExampleDataset(reference_system="INDIP").get_subset(
        cohort="MS", participant_id="001", test="Test11", trial="Trial1"
    )
    result = MobilisedPipelineImpaired(sdmo_calculation=RMS(gyr_columns=["gyr_is"])).run(dataset)
    assert result.per_wb_signal_based_parameters_.rms_gyr_is.notna().all()
    assert isinstance(result.sdmo_calculation_, RMS)
    assert result.per_turn_parameters_.empty


def test_single_turn_calculator_preserves_all_detected_turns():
    from mobgap.signal_based import TurnSDMO

    dataset = LabExampleDataset(reference_system="INDIP").get_subset(
        cohort="MS", participant_id="001", test="Test11", trial="Trial1"
    )
    result = MobilisedPipelineImpaired(sdmo_calculation=TurnSDMO()).run(dataset)
    assert len(result.per_turn_parameters_) == len(result.raw_turn_list_)
    assert result.per_turn_parameters_.turn_peak_ang_vel.notna().all()
    assert list(result.per_wb_signal_based_parameters_) == ["duration_s"]


def test_sdmo_aggregation_is_independent_of_primary_aggregation():
    from mobgap.signal_based import RMS

    dataset = LabExampleDataset(reference_system="INDIP").get_subset(
        cohort="MS", participant_id="001", test="Test11", trial="Trial1"
    )
    result = MobilisedPipelineImpaired(dmo_aggregation=None, sdmo_calculation=RMS(gyr_columns=["gyr_is"])).run(dataset)
    assert result.dmo_aggregation_ is None
    assert result.sdmo_aggregation_ is not None
    pd.testing.assert_frame_equal(result.aggregated_parameters_, result.sdmo_aggregation_.aggregated_data_)
    assert result.aggregated_parameters_.loc["all_wbs", "all__rms_gyr_is__median"] > 0


def test_disabling_both_aggregators_returns_no_aggregated_result():
    from mobgap.signal_based import RMS

    dataset = LabExampleDataset(reference_system="INDIP").get_subset(
        cohort="MS", participant_id="001", test="Test11", trial="Trial1"
    )
    result = MobilisedPipelineImpaired(
        dmo_aggregation=None, sdmo_aggregation=None, sdmo_calculation=RMS(gyr_columns=["gyr_is"])
    ).run(dataset)
    assert result.aggregated_parameters_ is None
    assert result.dmo_aggregation_ is None
    assert result.sdmo_aggregation_ is None
    assert not result.per_wb_signal_based_parameters_.empty


class _EmptyGsDetector(BaseGsDetector):
    def detect(self, data, sampling_rate_hz, **kwargs):
        self.gs_list_ = pd.DataFrame({"start": pd.Series(dtype="int64"), "end": pd.Series(dtype="int64")}).rename_axis(
            "gs_id"
        )
        return self


class _TurnContextCalculator(BaseSDMOCalculator):
    def calculate(self, data, turn_list, **kwargs):
        self.data = data
        self.turn_list = turn_list
        self.signal_based_parameters_ = pd.DataFrame({"custom_metric": [1.0]})
        return self


@pytest.mark.parametrize("no_gs", [True, False])
def test_empty_recording_turn_result_has_recording_index(no_gs):
    dataset = LabExampleDataset(reference_system="INDIP").get_subset(
        cohort="MS", participant_id="001", test="Test11", trial="Trial1"
    )
    config = {"gait_sequence_detection": _EmptyGsDetector()} if no_gs else {"turn_detection": None}
    result = MobilisedPipelineImpaired(**config).run(dataset)
    assert result.per_turn_parameters_.empty
    assert result.per_turn_parameters_.index.names == ["gs_id", "turn_id"]
    assert "turn_peak_ang_vel" in result.per_turn_parameters_


def test_custom_wb_calculator_keeps_required_wb_local_turn_context():
    dataset = LabExampleDataset(reference_system="INDIP").get_subset(
        cohort="MS", participant_id="001", test="Test11", trial="Trial1"
    )
    result = MobilisedPipelineImpaired(sdmo_calculation=_TurnContextCalculator()).run(dataset)
    wb = result.per_wb_parameters_.iloc[-1]
    expected = result.raw_turn_list_[
        (result.raw_turn_list_.start >= wb.start) & (result.raw_turn_list_.end <= wb.end)
    ].copy()
    assert not expected.empty
    expected.loc[:, ["start", "end"]] -= wb.start
    pd.testing.assert_frame_equal(result.sdmo_calculation_.turn_list, expected)
    assert result.per_turn_parameters_.empty


def test_fourier_harmonic_ratio_uses_pipeline_stride_bounds_and_aggregates():
    from mobgap.signal_based import HarmonicRatioFourier
    from mobgap.utils.conversions import to_body_frame

    dataset = LabExampleDataset(reference_system="INDIP").get_subset(
        cohort="MS", participant_id="001", test="Test11", trial="Trial1"
    )
    result = MobilisedPipelineImpaired(
        sdmo_calculation=HarmonicRatioFourier(acc_columns=["acc_is", "acc_pa", "acc_ml"])
    ).run(dataset)
    wb_id = result.per_wb_parameters_.index[0]
    wb = result.per_wb_parameters_.loc[wb_id]
    strides = result.per_stride_parameters_.loc[wb_id].copy()
    strides.loc[:, ["start", "end"]] -= int(wb.start)
    expected = (
        HarmonicRatioFourier(acc_columns=["acc_is", "acc_pa", "acc_ml"])
        .calculate(
            to_body_frame(dataset.data_ss.iloc[int(wb.start) : int(wb.end)]),
            stride_list=strides,
            sampling_rate_hz=dataset.sampling_rate_hz,
        )
        .signal_based_parameters_.iloc[0]
    )
    pd.testing.assert_series_equal(
        result.per_wb_signal_based_parameters_.loc[wb_id, expected.index],
        expected,
        check_names=False,
    )
    assert expected.notna().all()
    assert "all__harmonic_ratio_fourier_acc_is__median" in result.aggregated_parameters_
