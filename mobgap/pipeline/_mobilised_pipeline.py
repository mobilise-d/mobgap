from types import MappingProxyType
from typing import Final, Optional

import pandas as pd
from tpcp import Pipeline, cf
from tpcp.misc import set_defaults
from typing_extensions import Self

from mobgap.aggregation import MobilisedAggregator, apply_thresholds, get_mobilised_dmo_thresholds
from mobgap.aggregation.base import BaseAggregator
from mobgap.cad import CadFromIcDetector
from mobgap.cad.base import BaseCadCalculator
from mobgap.data.base import BaseGaitDataset, ParticipantMetadata
from mobgap.gsd import GsdIluz, GsdIonescu
from mobgap.gsd.base import BaseGsDetector
from mobgap.icd import IcdHKLeeImproved, IcdIonescu, IcdShinImproved, refine_gs
from mobgap.icd.base import BaseIcDetector
from mobgap.lrc import LrcUllrich, strides_list_from_ic_lr_list
from mobgap.lrc.base import BaseLRClassifier
from mobgap.pipeline._gs_iterator import FullPipelinePerGsResult, GsIterator
from mobgap.stride_length import SlZijlstra
from mobgap.stride_length.base import BaseSlCalculator
from mobgap.turning import TdElGohary
from mobgap.turning.base import BaseTurnDetector
from mobgap.utils.array_handling import create_multi_groupby
from mobgap.utils.interpolation import naive_sec_paras_to_regions
from mobgap.walking_speed import WsNaive
from mobgap.walking_speed.base import BaseWsCalculator
from mobgap.wba import StrideSelection, WbAssembly


class MobilisedPipeline(Pipeline[BaseGaitDataset]):
    gait_sequence_detection: BaseGsDetector
    initial_contact_detection: BaseIcDetector
    laterality_classification: BaseLRClassifier
    cadence_calculation: Optional[BaseCadCalculator]
    stride_length_calculation: Optional[BaseSlCalculator]
    walking_speed_calculation: Optional[BaseWsCalculator]
    turn_detection: Optional[BaseTurnDetector]
    stride_selection: StrideSelection
    wba: WbAssembly
    dmo_thresholds: Optional[pd.DataFrame]
    dmo_aggregation: BaseAggregator

    # Algos with results
    gait_sequence_detection_: BaseGsDetector
    gs_iterator_: GsIterator[FullPipelinePerGsResult]
    stride_selection_: StrideSelection
    wba_: WbAssembly
    dmo_aggregation_: BaseAggregator

    # Intermediate results
    gs_list_: pd.DataFrame
    raw_ic_list_: pd.DataFrame
    raw_per_sec_parameters_: pd.DataFrame
    raw_per_stride_parameters_: pd.DataFrame

    # Final Results
    per_stride_parameters_: pd.DataFrame
    per_wb_parameters_: pd.DataFrame
    per_wb_parameter_mask_: Optional[pd.DataFrame]
    aggregated_parameters_: pd.DataFrame

    class PredefinedParameters:
        normal_walking: Final = MappingProxyType(
            {
                "gait_sequence_detection": GsdIluz(),
                "initial_contact_detection": IcdIonescu(),
                "laterality_classification": LrcUllrich(**LrcUllrich.PredefinedParameters.msproject_all),
                "cadence_calculation": CadFromIcDetector(IcdShinImproved()),
                "stride_length_calculation": SlZijlstra(),
                "walking_speed_calculation": WsNaive(),
                "turn_detection": TdElGohary(),
                "stride_selection": StrideSelection(),
                "wba": WbAssembly(),
                "dmo_thresholds": get_mobilised_dmo_thresholds(),
                "dmo_aggregation": MobilisedAggregator(groupby=None),
            }
        )

        impaired_walking: Final = MappingProxyType(
            {
                "gait_sequence_detection": GsdIonescu(),
                "initial_contact_detection": IcdIonescu(),
                "laterality_classification": LrcUllrich(**LrcUllrich.PredefinedParameters.msproject_all),
                "cadence_calculation": CadFromIcDetector(IcdHKLeeImproved()),
                "stride_length_calculation": SlZijlstra(),
                "walking_speed_calculation": WsNaive(),
                "turn_detection": TdElGohary(),
                "stride_selection": StrideSelection(),
                "wba": WbAssembly(),
                "dmo_thresholds": get_mobilised_dmo_thresholds(),
                "dmo_aggregation": MobilisedAggregator(groupby=None),
            }
        )

    def __init__(
        self,
        *,
        gait_sequence_detection: BaseGsDetector,
        initial_contact_detection: BaseIcDetector,
        laterality_classification: BaseLRClassifier,
        cadence_calculation: Optional[BaseCadCalculator],
        stride_length_calculation: Optional[BaseSlCalculator],
        walking_speed_calculation: Optional[BaseWsCalculator],
        turn_detection: Optional[BaseTurnDetector],
        stride_selection: StrideSelection,
        wba: WbAssembly,
        dmo_thresholds: Optional[pd.DataFrame],
        dmo_aggregation: BaseAggregator,
    ) -> None:
        self.gait_sequence_detection = gait_sequence_detection
        self.initial_contact_detection = initial_contact_detection
        self.laterality_classification = laterality_classification
        self.cadence_calculation = cadence_calculation
        self.stride_length_calculation = stride_length_calculation
        self.walking_speed_calculation = walking_speed_calculation
        self.turn_detection = turn_detection
        self.stride_selection = stride_selection
        self.wba = wba
        self.dmo_thresholds = dmo_thresholds
        self.dmo_aggregation = dmo_aggregation

    def run(self, datapoint: BaseGaitDataset) -> Self:
        imu_data = datapoint.data_ss
        sampling_rate_hz = datapoint.sampling_rate_hz

        self.gait_sequence_detection_ = self.gait_sequence_detection.clone().detect(
            imu_data, sampling_rate_hz=sampling_rate_hz
        )
        self.gs_list_ = self.gait_sequence_detection_.gs_list_
        self.gs_iterator_ = self._run_per_gs(self.gs_list_, imu_data, sampling_rate_hz, datapoint.participant_metadata)

        results = self.gs_iterator_.results_

        self.raw_per_sec_parameters_ = pd.concat(
            [
                results.cadence_per_sec,
                results.stride_length_per_sec,
                results.walking_speed_per_sec,
            ],
            axis=1,
        ).reset_index("r_gs_id", drop=True)
        self.raw_ic_list_ = results.ic_list
        self.raw_per_stride_parameters_ = self._sec_to_stride(
            self.raw_per_sec_parameters_, results.ic_list, sampling_rate_hz
        )

        flat_index = pd.Index(
            ["_".join(str(e) for e in s_id) for s_id in self.raw_per_stride_parameters_.index], name="s_id"
        )
        raw_per_stride_parameters = self.raw_per_stride_parameters_.reset_index("gs_id").rename(
            columns={"gs_id": "original_gs_id"}
        )
        raw_per_stride_parameters.index = flat_index

        self.stride_selection_ = self.stride_selection.clone().filter(
            raw_per_stride_parameters, sampling_rate_hz=sampling_rate_hz
        )
        self.wba_ = self.wba.clone().assemble(
            self.stride_selection_.filtered_stride_list_, sampling_rate_hz=sampling_rate_hz
        )

        self.per_stride_parameters_ = self.wba_.annotated_stride_list_
        self.per_wb_parameters_ = self._aggregate_per_wb(self.per_stride_parameters_, self.wba_.wb_meta_parameters_)
        if self.dmo_thresholds is None:
            self.per_wb_parameter_mask_ = None
        else:
            self.per_wb_parameter_mask_ = apply_thresholds(
                self.per_wb_parameters_,
                self.dmo_thresholds,
                cohort=datapoint.participant_metadata["cohort"],
                height_m=datapoint.participant_metadata["height_m"],
                measurement_condition=datapoint.recording_metadata["measurement_condition"],
            )

        self.dmo_aggregation_ = self.dmo_aggregation.clone().aggregate(
            self.per_wb_parameters_, wb_dmos_mask=self.per_wb_parameter_mask_
        )
        self.aggregated_parameters_ = self.dmo_aggregation_.aggregated_data_

        return self

    def _run_per_gs(
        self,
        gait_sequences: pd.DataFrame,
        imu_data: pd.DataFrame,
        sampling_rate_hz: float,
        participant_metadata: ParticipantMetadata,
    ) -> GsIterator:
        gs_iterator = GsIterator[FullPipelinePerGsResult]()
        # TODO: How to expose the individual algo instances of the algos that run in the loop?

        for (_, gs_data), r in gs_iterator.iterate(imu_data, gait_sequences):
            icd = self.initial_contact_detection.clone().detect(gs_data, sampling_rate_hz=sampling_rate_hz)
            lrc = self.laterality_classification.clone().predict(
                gs_data, icd.ic_list_, sampling_rate_hz=sampling_rate_hz
            )
            if self.turn_detection:
                r.ic_list = lrc.ic_lr_list_
                turn = self.turn_detection.clone().detect(gs_data, sampling_rate_hz=sampling_rate_hz)
                r.turn_list = turn.turn_list_

            refined_gs, refined_ic_list = refine_gs(r.ic_list)

            with gs_iterator.subregion(refined_gs) as ((_, refined_gs_data), rr):
                cad_r = None
                if self.cadence_calculation:
                    cad = self.cadence_calculation.clone().calculate(
                        refined_gs_data,
                        initial_contacts=refined_ic_list,
                        sampling_rate_hz=sampling_rate_hz,
                        **participant_metadata,
                    )
                    cad_r = cad.cadence_per_sec_
                    rr.cadence_per_sec = cad_r
                sl_r = None
                if self.stride_length_calculation:
                    sl = self.stride_length_calculation.clone().calculate(
                        refined_gs_data,
                        initial_contacts=refined_ic_list,
                        sampling_rate_hz=sampling_rate_hz,
                        **participant_metadata,
                    )
                    sl_r = sl.stride_length_per_sec_
                    rr.stride_length_per_sec = sl.stride_length_per_sec_
                if self.walking_speed_calculation:
                    ws = self.walking_speed_calculation.clone().calculate(
                        refined_gs_data,
                        initial_contacts=refined_ic_list,
                        cadence_per_sec=cad_r,
                        stride_length_per_sec=sl_r,
                        sampling_rate_hz=sampling_rate_hz,
                        **participant_metadata,
                    )
                    rr.walking_speed_per_sec = ws.walking_speed_per_sec_

        return gs_iterator

    def _sec_to_stride(
        self, sec_level_paras: pd.DataFrame, lr_ic_list: pd.DataFrame, sampling_rate_hz: float
    ) -> pd.DataFrame:
        stride_list = (
            lr_ic_list.groupby("gs_id", group_keys=False)
            .apply(strides_list_from_ic_lr_list)
            .assign(stride_duration_s=lambda df_: (df_.end - df_.start) / sampling_rate_hz)
        )

        stride_list = create_multi_groupby(
            stride_list,
            sec_level_paras,
            "gs_id",
            group_keys=False,
        ).apply(naive_sec_paras_to_regions, sampling_rate_hz=sampling_rate_hz)
        return stride_list

    def _aggregate_per_wb(self, per_stride_parameters: pd.DataFrame, wb_meta_parameters: pd.DataFrame) -> pd.DataFrame:
        # TODO: Make a class constant
        params_to_aggregate = [
            "stride_duration_s",
            "cadence_spm",
            "stride_length_m",
            "walking_speed_mps",
        ]
        return pd.concat(
            [
                wb_meta_parameters,
                per_stride_parameters.reindex(columns=params_to_aggregate)
                .groupby(["wb_id"])
                # TODO: Decide if we should use mean or trim_mean here!
                .mean(),
            ],
            axis=1,
        )


class MobilisedPipelineHealthy(MobilisedPipeline):
    @set_defaults(**{k: cf(v) for k, v in MobilisedPipeline.PredefinedParameters.normal_walking.items()})
    def __init__(
        self,
        *,
        gait_sequence_detection: BaseGsDetector,
        initial_contact_detection: BaseIcDetector,
        laterality_classification: BaseLRClassifier,
        cadence_calculation: Optional[BaseCadCalculator],
        stride_length_calculation: Optional[BaseSlCalculator],
        walking_speed_calculation: Optional[BaseWsCalculator],
        turn_detection: Optional[BaseTurnDetector],
        stride_selection: StrideSelection,
        wba: WbAssembly,
        dmo_thresholds: Optional[pd.DataFrame],
        dmo_aggregation: BaseAggregator,
    ) -> None:
        super().__init__(
            gait_sequence_detection=gait_sequence_detection,
            initial_contact_detection=initial_contact_detection,
            laterality_classification=laterality_classification,
            cadence_calculation=cadence_calculation,
            stride_length_calculation=stride_length_calculation,
            walking_speed_calculation=walking_speed_calculation,
            turn_detection=turn_detection,
            stride_selection=stride_selection,
            wba=wba,
            dmo_thresholds=dmo_thresholds,
            dmo_aggregation=dmo_aggregation,
        )


class MobilisedPipelineImpaired(MobilisedPipeline):
    @set_defaults(**{k: cf(v) for k, v in MobilisedPipeline.PredefinedParameters.impaired_walking.items()})
    def __init__(
        self,
        *,
        gait_sequence_detection: BaseGsDetector,
        initial_contact_detection: BaseIcDetector,
        laterality_classification: BaseLRClassifier,
        cadence_calculation: Optional[BaseCadCalculator],
        stride_length_calculation: Optional[BaseSlCalculator],
        walking_speed_calculation: Optional[BaseWsCalculator],
        turn_detection: Optional[BaseTurnDetector],
        stride_selection: StrideSelection,
        wba: WbAssembly,
        dmo_thresholds: Optional[pd.DataFrame],
        dmo_aggregation: BaseAggregator,
    ) -> None:
        super().__init__(
            gait_sequence_detection=gait_sequence_detection,
            initial_contact_detection=initial_contact_detection,
            laterality_classification=laterality_classification,
            cadence_calculation=cadence_calculation,
            stride_length_calculation=stride_length_calculation,
            walking_speed_calculation=walking_speed_calculation,
            turn_detection=turn_detection,
            stride_selection=stride_selection,
            wba=wba,
            dmo_thresholds=dmo_thresholds,
            dmo_aggregation=dmo_aggregation,
        )
