import gc
import weakref
from typing import Any, Self

import numpy as np

import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS, SF_ACC_COLS, SF_SENSOR_COLS
from mobgap.data import GaitDatasetFromData, LabExampleDataset
from mobgap.gait_sequences.base import BaseGsDetector
from mobgap.initial_contacts.base import BaseIcDetector
from mobgap.laterality import LrcBenMansour
from mobgap.signal_based.base import BaseSDMOCalculator
from mobgap.laterality.base import BaseLRClassifier
from mobgap.pipeline import (
    GenericMobilisedPipeline,
    MobilisedPipelineHealthy,
    MobilisedPipelineImpaired,
    MobilisedPipelineUniversal,
)
from mobgap.re_orientation import ReorientationMethodDM
from mobgap.re_orientation.base import BaseReorientationCorrector
from mobgap.stride_length.base import BaseSlCalculator
from mobgap.wba import StrideSelection, WbAssembly


class _FullRecordingGsDetector(BaseGsDetector):
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_kwargs: Any) -> Self:
        del sampling_rate_hz
        self.data_columns_ = tuple(data.columns)
        self.gs_list_ = pd.DataFrame({"start": [0], "end": [len(data)]}).rename_axis("gs_id")
        return self


class _SensorToBodyFrameReorientation(BaseReorientationCorrector[None]):
    def detect_correct(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_kwargs: Any) -> Self:
        del sampling_rate_hz
        self.input_columns_ = tuple(data.columns)
        self.corrected_data_ = self.apply_correction(data, None)
        self.result_ = self.input_columns_
        return self

    @staticmethod
    def apply_correction(data: pd.DataFrame, correction: None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "acc_is": data["acc_x"],
                "acc_ml": data["acc_y"],
                "acc_pa": data["acc_z"],
                "gyr_is": data["gyr_x"],
                "gyr_ml": data["gyr_y"],
                "gyr_pa": data["gyr_z"],
            },
            index=data.index,
        )


class _EmptyIcDetector(BaseIcDetector):
    def detect(self, _data: pd.DataFrame, *, sampling_rate_hz: float, **_kwargs: Any) -> Self:
        del sampling_rate_hz
        self.ic_list_ = pd.DataFrame({"ic": pd.Series(dtype="int64")}).rename_axis("step_id")
        return self


class _FixedIcDetector(BaseIcDetector):
    def detect(self, _data: pd.DataFrame, *, sampling_rate_hz: float, **_kwargs: Any) -> Self:
        del sampling_rate_hz
        self.ic_list_ = pd.DataFrame({"ic": [50, 150, 250]}).rename_axis("step_id")
        return self


class _PassthroughLrc(BaseLRClassifier):
    def predict(self, _data: pd.DataFrame, ic_list: pd.DataFrame, *, sampling_rate_hz: float, **_kwargs: Any) -> Self:
        del sampling_rate_hz
        self.ic_lr_list_ = ic_list.copy()
        self.ic_lr_list_["lr_label"] = pd.Series(
            pd.Categorical(["left"] * len(ic_list), categories=["left", "right"]),
            index=ic_list.index,
        )
        return self


class _MeanAccIsSlCalculator(BaseSlCalculator):
    def calculate(
        self, data: pd.DataFrame, initial_contacts: pd.DataFrame, *, sampling_rate_hz: float, **_kwargs: Any
    ) -> Self:
        self.data = data
        self.initial_contacts = initial_contacts
        self.sampling_rate_hz = sampling_rate_hz
        sec_center_samples = pd.Index(
            range(int(sampling_rate_hz // 2), len(data) + int(sampling_rate_hz // 2) + 1, int(sampling_rate_hz)),
            name="sec_center_samples",
        )
        self.stride_length_per_sec_ = pd.DataFrame({"stride_length_m": data["acc_is"].mean()}, index=sec_center_samples)
        return self


def _sensor_frame_test_dataset() -> GaitDatasetFromData:
    sensor_data = pd.DataFrame(
        {
            "acc_x": 9.81,
            "acc_y": 1.0,
            "acc_z": 2.0,
            "gyr_x": 0.0,
            "gyr_y": 0.0,
            "gyr_z": 0.0,
        },
        index=range(300),
    )[SF_SENSOR_COLS]
    return GaitDatasetFromData(
        {"test": {"LowerBack": sensor_data}},
        100.0,
        _participant_metadata={"test": {"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"}},
        _recording_metadata={"test": {"measurement_condition": "laboratory"}},
    )[0]


def _minimal_pipeline(**overrides: Any) -> GenericMobilisedPipeline:
    params = dict(GenericMobilisedPipeline.PredefinedParameters.regular_walking)
    params.update(
        {
            "gait_sequence_detection": _FullRecordingGsDetector(),
            "initial_contact_detection": _FixedIcDetector(),
            "laterality_classification": _PassthroughLrc(),
            "cadence_calculation": None,
            "stride_length_calculation": _MeanAccIsSlCalculator(),
            "walking_speed_calculation": None,
            "turn_detection": None,
            "stride_selection": StrideSelection(rules=None, incompatible_rules="raise"),
            "wba": WbAssembly(rules=None),
            "dmo_thresholds": None,
            "dmo_aggregation": None,
        }
    )
    params.update(overrides)
    return GenericMobilisedPipeline(**params)


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
    def test_acceleration_only_pipeline_with_acceleration_based_algorithms(self) -> None:
        example = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2")
        dataset = GaitDatasetFromData(
            {"test": {"LowerBack": example.data_ss[SF_ACC_COLS]}},
            example.sampling_rate_hz,
            _participant_metadata={"test": example.participant_metadata},
            _recording_metadata={"test": example.recording_metadata},
        )[0]
        pipeline = GenericMobilisedPipeline(
            **(
                GenericMobilisedPipeline.PredefinedParameters.regular_walking
                | {"laterality_classification": LrcBenMansour(), "turn_detection": None}
            )
        )

        result = pipeline.run(dataset)

        assert not result.gs_list_.empty
        assert not result.raw_ic_list_.empty
        assert not result.raw_per_sec_parameters_.empty

    def test_without_per_gs_reorientation_gsd_receives_body_frame_data(self) -> None:
        pipeline = _minimal_pipeline().run(_sensor_frame_test_dataset())

        assert pipeline.gait_sequence_detection_.data_columns_ == tuple(BF_SENSOR_COLS)

    def test_per_gs_reorientation_keeps_gsd_in_sensor_frame_and_downstream_data_in_body_frame(self) -> None:
        pipeline = _minimal_pipeline(per_gs_reorientation=_SensorToBodyFrameReorientation()).run(
            _sensor_frame_test_dataset()
        )

        assert pipeline.gait_sequence_detection_.data_columns_ == tuple(SF_SENSOR_COLS)
        assert pipeline.gs_iterator_.results_.reorientation_result[0] == tuple(SF_SENSOR_COLS)
        assert pipeline.raw_per_sec_parameters_["stride_length_m"].iloc[0] == pytest.approx(9.81)

    def test_healthy_pipeline_with_per_gs_reorientation_fails_because_gsd_iluz_requires_body_frame(self) -> None:
        dataset = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2")[0]

        with pytest.raises(AssertionError, match="no valid imu data in the body frame"):
            MobilisedPipelineHealthy(per_gs_reorientation=ReorientationMethodDM()).run(dataset)

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

    def test_per_gs_reorientation_is_used_for_refined_gs_data(self) -> None:
        sensor_data = pd.DataFrame(
            {
                "acc_x": 0.0,
                "acc_y": 9.81,
                "acc_z": 1.0,
                "gyr_x": 0.0,
                "gyr_y": 0.0,
                "gyr_z": 0.0,
            },
            index=range(300),
        )[SF_SENSOR_COLS]
        gs_list = pd.DataFrame({"start": [0], "end": [len(sensor_data)]}).rename_axis("gs_id")
        pipeline = GenericMobilisedPipeline(
            **(
                GenericMobilisedPipeline.PredefinedParameters.regular_walking
                | {
                    "per_gs_reorientation": ReorientationMethodDM(
                        correction_mode="full", pa_direction_detection_error_type="ignore"
                    ),
                    "initial_contact_detection": _FixedIcDetector(),
                    "laterality_classification": _PassthroughLrc(),
                    "cadence_calculation": None,
                    "stride_length_calculation": _MeanAccIsSlCalculator(),
                    "walking_speed_calculation": None,
                    "turn_detection": None,
                    "dmo_thresholds": None,
                    "dmo_aggregation": None,
                }
            )
        )

        result = pipeline._run_per_gs(gs_list, sensor_data, {"sampling_rate_hz": 100.0}).results_

        assert result.reorientation_result[0].family == "ml_up"
        assert result.stride_length_per_sec["stride_length_m"].iloc[0] == pytest.approx(9.81)


class TestIntermediateRetention:
    @pytest.mark.parametrize("pipeline_cls", [MobilisedPipelineHealthy, MobilisedPipelineImpaired])
    @pytest.mark.parametrize("aggregate", [True, False])
    def test_disabling_retention_preserves_tables(self, pipeline_cls, aggregate):
        dataset = LabExampleDataset(reference_system="INDIP").get_subset(
            cohort="MS", participant_id="001", test="Test11", trial="Trial1"
        )
        options = {} if aggregate else {"dmo_aggregation": None}
        retained = pipeline_cls(**options).safe_run(dataset)
        compact = pipeline_cls(retain_intermediate_results=False, **options).safe_run(dataset)

        for name in (
            "gs_list_",
            "raw_ic_list_",
            "raw_turn_list_",
            "raw_per_sec_parameters_",
            "raw_per_stride_parameters_",
            "per_stride_parameters_",
            "per_wb_parameters_",
            "per_wb_parameter_mask_",
            "aggregated_parameters_",
        ):
            expected, actual = getattr(retained, name), getattr(compact, name)
            if expected is None:
                assert actual is None
            else:
                pd.testing.assert_frame_equal(
                    actual.drop(columns="rule_obj", errors="ignore"),
                    expected.drop(columns="rule_obj", errors="ignore"),
                )
        assert retained.datapoint is dataset
        assert retained.gait_sequence_detection_.gs_list_ is retained.gs_list_
        assert retained.gs_iterator_.raw_results_
        for name in (
            "datapoint",
            "gait_sequence_detection_",
            "gs_iterator_",
            "stride_selection_",
            "wba_",
            "dmo_aggregation_",
        ):
            assert not hasattr(compact, name)
        assert compact.perf_["runtime_s"] > 0

    @pytest.mark.parametrize("retain", [True, False])
    def test_retention_controls_input_lifetime(self, retain):
        dataset = _sensor_frame_test_dataset()
        dataset_ref = weakref.ref(dataset)
        backing = dataset.data_ss.to_numpy()
        while isinstance(backing.base, np.ndarray):
            backing = backing.base
        backing_ref = weakref.ref(backing)
        pipeline = _minimal_pipeline(retain_intermediate_results=retain).safe_run(dataset)
        del dataset, backing
        gc.collect()

        assert (dataset_ref() is not None) is retain
        assert (backing_ref() is not None) is retain
        assert not pipeline.per_stride_parameters_.empty

    def test_disabling_retention_on_reused_pipeline_releases_previous_run(self):
        first = _sensor_frame_test_dataset()
        first_ref = weakref.ref(first)
        pipeline = _minimal_pipeline().run(first)
        del first
        assert first_ref() is not None

        pipeline.set_params(retain_intermediate_results=False).run(_sensor_frame_test_dataset())
        gc.collect()
        assert first_ref() is None
        assert not hasattr(pipeline, "datapoint")
        assert not hasattr(pipeline, "gs_iterator_")

        # Re-enabling retention on the same object restores the normal debugging API.
        third = _sensor_frame_test_dataset()
        pipeline.set_params(retain_intermediate_results=True).run(third)
        assert pipeline.datapoint is third
        assert pipeline.gs_iterator_.raw_results_

    def test_failed_run_does_not_leave_intermediate_attributes(self):
        # Missing cohort raises after GSD, per-GS processing and WB assembly have completed.
        dataset = _sensor_frame_test_dataset()
        dataset._participant_metadata["test"].pop("cohort")
        dataset_ref = weakref.ref(dataset)
        pipeline = _minimal_pipeline(
            retain_intermediate_results=False,
            dmo_thresholds=MobilisedPipelineHealthy().dmo_thresholds,
        )
        with pytest.raises(ValueError, match="cohort of the participant"):
            pipeline.run(dataset)
        del dataset
        gc.collect()
        assert dataset_ref() is None
        for name in ("datapoint", "gait_sequence_detection_", "gs_iterator_", "stride_selection_", "wba_"):
            assert not hasattr(pipeline, name)



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
        self.signal_based_parameters_ = pd.Series({"custom_metric": 1.0})
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
        .signal_based_parameters_
    )
    pd.testing.assert_series_equal(
        result.per_wb_signal_based_parameters_.loc[wb_id, expected.index],
        expected,
        check_names=False,
    )
    assert expected.notna().all()
    assert "all__harmonic_ratio_fourier_acc_is__median" in result.aggregated_parameters_
