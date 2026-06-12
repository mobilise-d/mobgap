from typing import Any

import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin
from typing_extensions import Self

from mobgap._gaitmap.utils.rotations import flip_dataset
from mobgap.consts import BF_SENSOR_COLS
from mobgap.data import LabExampleDataset
from mobgap.initial_contacts.base import BaseIcDetector
from mobgap.laterality.base import BaseLRClassifier
from mobgap.pipeline import (
    GenericMobilisedPipeline,
    MobilisedPipelineHealthy,
    MobilisedPipelineImpaired,
    MobilisedPipelineUniversal,
)
from mobgap.re_orientation import ReorientationMethodDM
from mobgap.re_orientation.pipeline import REORIENTATION_ROTATIONS
from mobgap.stride_length.base import BaseSlCalculator


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
        self.stride_length_per_sec_ = pd.DataFrame(
            {"stride_length_m": [data["acc_is"].mean()]},
            index=pd.Index([int(sampling_rate_hz // 2)], name="sec_center_samples"),
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

    def test_reorientation_correction_is_used_for_refined_gs_data(self) -> None:
        base_data = pd.DataFrame(
            {
                "acc_is": 9.81,
                "acc_ml": 0.0,
                "acc_pa": 1.0,
                "gyr_is": 0.0,
                "gyr_ml": 0.0,
                "gyr_pa": 0.0,
            },
            index=range(300),
        )[BF_SENSOR_COLS]
        rotated_data = flip_dataset(base_data, REORIENTATION_ROTATIONS["pa_normal__rot_pa_pos90"])
        gs_list = pd.DataFrame({"start": [0], "end": [len(rotated_data)]}).rename_axis("gs_id")
        pipeline = GenericMobilisedPipeline(
            **(
                GenericMobilisedPipeline.PredefinedParameters.regular_walking
                | {
                    "reorientation_correction": ReorientationMethodDM(correction_mode="full"),
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

        result = pipeline._run_per_gs(gs_list, rotated_data, {"sampling_rate_hz": 100.0}).results_

        assert result.reorientation_result[0].family == 3
        assert result.stride_length_per_sec["stride_length_m"].iloc[0] == pytest.approx(9.81)
