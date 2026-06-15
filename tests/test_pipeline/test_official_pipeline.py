from typing import Any

import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin
from typing_extensions import Self

from mobgap.consts import BF_SENSOR_COLS, SF_SENSOR_COLS
from mobgap.data import GaitDatasetFromData, LabExampleDataset
from mobgap.gait_sequences.base import BaseGsDetector
from mobgap.initial_contacts.base import BaseIcDetector
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


class _SensorToBodyFrameReorientation(BaseReorientationCorrector):
    def detect_correct(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_kwargs: Any) -> Self:
        del sampling_rate_hz
        self.input_columns_ = tuple(data.columns)
        self.corrected_data_ = pd.DataFrame(
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
        self.result_ = self.input_columns_
        return self


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

    def test_impaired_pipeline_handles_no_detected_ics(self):
        dataset = LabExampleDataset(reference_system="INDIP").get_subset(
            cohort="MS", participant_id="001", test="Test11", trial="Trial1"
        )

        result = MobilisedPipelineImpaired(
            initial_contact_detection=_EmptyIcDetector(),
            laterality_classification=_PassthroughLrc(),
        ).run(dataset)

        assert result.raw_ic_list_.empty
        assert result.raw_per_sec_parameters_.empty
        assert result.raw_per_stride_parameters_.empty
        assert result.per_stride_parameters_.empty
        assert result.per_wb_parameters_.empty

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
