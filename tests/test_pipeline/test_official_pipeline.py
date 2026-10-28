import gc
import weakref
from enum import IntEnum
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


class _ReplayCorrection(IntEnum):
    ALIGNED = 1
    FLIPPED = -1


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
        self.correction_ = None
        self.correction_applied_ = False
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
    @pytest.mark.parametrize("run_first", [False, True])
    def test_disabled_aggregation_clears_previous_aggregator(self, run_first):
        dataset = LabExampleDataset(reference_system="INDIP").get_subset(
            cohort="MS", participant_id="001", test="Test11", trial="Trial1"
        )
        pipeline = MobilisedPipelineImpaired()
        if run_first:
            pipeline.run(dataset)
            assert pipeline.dmo_aggregation_ is not None
        pipeline.set_params(dmo_aggregation=None).run(dataset)
        assert pipeline.aggregated_parameters_ is None
        assert pipeline.dmo_aggregation_ is None

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
        assert pipeline.gs_list_.loc[0, "orientation"] is None
        assert not pipeline.gs_list_.loc[0, "correction_applied"]
        assert pipeline.raw_per_sec_parameters_["stride_length_m"].iloc[0] == pytest.approx(9.81)

    @pytest.mark.parametrize("retain", [True, False])
    def test_gs_list_exposes_replayable_correction_without_retaining_intermediates(self, retain):
        from scipy.spatial.transform import Rotation

        from mobgap.utils.conversions import to_body_frame
        from mobgap.utils.rotations import flip_dataset

        nominal = _sensor_frame_test_dataset()
        rotated = flip_dataset(nominal.data_ss, Rotation.from_euler("z", 90, degrees=True))
        dataset = GaitDatasetFromData(
            {"test": {"LowerBack": rotated}},
            100.0,
            _participant_metadata={"test": nominal.participant_metadata},
            _recording_metadata={"test": nominal.recording_metadata},
        )[0]
        pipeline = _minimal_pipeline(
            per_gs_reorientation=ReorientationMethodDM(
                correction_mode="trust_gravity", pa_direction_detection_error_type="ignore"
            ),
            retain_intermediate_results=retain,
        ).run(dataset)

        assert pipeline.gs_list_.loc[0, "correction_applied"]
        correction = pipeline.gs_list_.loc[0, "orientation"]
        replayed = pipeline.per_gs_reorientation.apply_correction(rotated, correction)
        pd.testing.assert_frame_equal(replayed, to_body_frame(nominal.data_ss))
        pd.testing.assert_frame_equal(
            pipeline.gs_list_[["start", "end"]],
            pd.DataFrame({"start": [0], "end": [300]}).rename_axis("gs_id"),
        )
        assert hasattr(pipeline, "gs_iterator_") is retain
        if retain:
            assert pipeline.gait_sequence_detection_.gs_list_.columns.tolist() == ["start", "end"]

    @pytest.mark.parametrize("column_ids", [False, True])
    def test_gs_corrections_keep_original_identifiers_and_recording_intervals(self, column_ids):
        from scipy.spatial.transform import Rotation

        from mobgap.re_orientation import get_orientation_for_sequences
        from mobgap.utils.conversions import to_body_frame
        from mobgap.utils.rotations import flip_dataset

        class SplitGsDetector(BaseGsDetector):
            def detect(self, data, *, sampling_rate_hz, **kwargs):
                self.gs_list_ = pd.DataFrame(
                    {"start": [0, 150], "end": [150, 300]}, index=pd.Index([42, 7], name="gs_id")
                )
                if column_ids:
                    self.gs_list_ = self.gs_list_.reset_index().set_axis(pd.Index([7, 42], name="detector_row"))
                return self

        nominal = _sensor_frame_test_dataset()
        data = pd.concat(
            [
                nominal.data_ss.iloc[:150],
                flip_dataset(nominal.data_ss.iloc[150:], Rotation.from_euler("z", 90, degrees=True)),
            ]
        )
        dataset = GaitDatasetFromData(
            {"test": {"LowerBack": data}},
            100.0,
            _participant_metadata={"test": nominal.participant_metadata},
            _recording_metadata={"test": nominal.recording_metadata},
        )[0]
        pipeline = _minimal_pipeline(
            gait_sequence_detection=SplitGsDetector(),
            initial_contact_detection=_EmptyIcDetector(),
            per_gs_reorientation=ReorientationMethodDM(pa_direction_detection_error_type="ignore"),
            retain_intermediate_results=False,
        ).run(dataset)

        expected_index = pd.Index([7, 42], name="detector_row") if column_ids else pd.Index([42, 7], name="gs_id")
        pd.testing.assert_index_equal(pipeline.gs_list_.index, expected_index)
        assert pipeline.gs_list_.correction_applied.tolist() == [False, True]
        sequences = pd.DataFrame({"start": [160], "end": [200]}, index=pd.Index(["query"]))
        correction = get_orientation_for_sequences(sequences, pipeline.gs_list_, default=None).loc["query"]
        pd.testing.assert_frame_equal(
            pipeline.per_gs_reorientation.apply_correction(data.iloc[160:200], correction),
            to_body_frame(nominal.data_ss.iloc[160:200]),
        )
        # A later run with correction disabled returns the unextended detector table.
        pipeline.set_params(per_gs_reorientation=None).run(dataset)
        assert pipeline.gs_list_.columns.tolist() == (["gs_id", "start", "end"] if column_ids else ["start", "end"])

    @pytest.mark.parametrize("retain", [True, False])
    @pytest.mark.parametrize("scalar_kind", ["enum", "large_int"])
    def test_gs_list_preserves_opaque_scalar_correction_types(self, retain, scalar_kind):
        from mobgap.utils.conversions import to_body_frame

        corrections = (
            [_ReplayCorrection.ALIGNED, _ReplayCorrection.FLIPPED] if scalar_kind == "enum" else [-1, 2**63 + 1]
        )

        class TwoGsDetector(BaseGsDetector):
            def detect(self, data, *, sampling_rate_hz, **kwargs):
                self.gs_list_ = pd.DataFrame({"start": [0, 150], "end": [150, 300]}).rename_axis("gs_id")
                return self

        class ScalarCorrector(BaseReorientationCorrector):
            def detect_correct(self, data, *, sampling_rate_hz, **kwargs):
                self.correction_ = corrections[int(data.index[0] >= 150)]
                self.correction_applied_ = data.index[0] >= 150
                self.corrected_data_ = self.apply_correction(data, self.correction_)
                return self

            @staticmethod
            def apply_correction(data, correction):
                sign = correction.value if scalar_kind == "enum" else {-1: 1, 2**63 + 1: -1}[correction]
                result = to_body_frame(data)
                result[["acc_ml", "acc_pa", "gyr_ml", "gyr_pa"]] *= sign
                return result

        dataset = _sensor_frame_test_dataset()
        pipeline = _minimal_pipeline(
            gait_sequence_detection=TwoGsDetector(),
            initial_contact_detection=_EmptyIcDetector(),
            per_gs_reorientation=ScalarCorrector(),
            retain_intermediate_results=retain,
        ).run(dataset)
        for gs_id, expected_correction in enumerate(corrections):
            correction = pipeline.gs_list_.loc[gs_id, "orientation"]
            assert correction == expected_correction
            if scalar_kind == "enum":
                assert correction is expected_correction
            start, end = [(0, 150), (150, 300)][gs_id]
            data = dataset.data_ss.iloc[start:end]
            expected = to_body_frame(data)
            if gs_id == 1:
                expected[["acc_ml", "acc_pa", "gyr_ml", "gyr_pa"]] *= -1
            pd.testing.assert_frame_equal(pipeline.per_gs_reorientation.apply_correction(data, correction), expected)

    def test_empty_gs_list_retains_reorientation_schema(self):
        class NoGsDetector(BaseGsDetector):
            def detect(self, data, *, sampling_rate_hz, **kwargs):
                self.gs_list_ = pd.DataFrame(
                    {"start": pd.Series(dtype="int64"), "end": pd.Series(dtype="int64")}
                ).rename_axis("gs_id")
                return self

        pipeline = _minimal_pipeline(
            gait_sequence_detection=NoGsDetector(),
            per_gs_reorientation=ReorientationMethodDM(),
            retain_intermediate_results=False,
        ).run(_sensor_frame_test_dataset())
        expected = pd.DataFrame(
            {
                "start": pd.Series(dtype="int64"),
                "end": pd.Series(dtype="int64"),
                "orientation": pd.Series(dtype=object),
                "correction_applied": pd.Series(dtype=bool),
            }
        ).rename_axis("gs_id")
        pd.testing.assert_frame_equal(pipeline.gs_list_, expected)

    def test_healthy_pipeline_with_per_gs_reorientation_fails_because_gsd_iluz_requires_body_frame(self) -> None:
        dataset = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2")[0]

        with pytest.raises(AssertionError, match="no valid imu data in the body frame"):
            MobilisedPipelineHealthy(per_gs_reorientation=ReorientationMethodDM()).run(dataset)

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
