import warnings
from itertools import combinations
from types import MappingProxyType
from typing import Any, Final, Generic, Optional

import pandas as pd
from tpcp import cf
from tpcp.misc import set_defaults
from typing_extensions import Self

from mobgap._utils_internal.misc import timed_action_method
from mobgap.aggregation import MobilisedAggregator, apply_thresholds, get_mobilised_dmo_thresholds
from mobgap.aggregation.base import BaseAggregator
from mobgap.cadence import CadFromIcDetector
from mobgap.cadence.base import BaseCadCalculator
from mobgap.gait_sequences import GsdIluz, GsdIonescu
from mobgap.gait_sequences.base import BaseGsDetector
from mobgap.initial_contacts import IcdIonescu, refine_gs
from mobgap.initial_contacts.base import BaseIcDetector
from mobgap.laterality import LrcUllrich, strides_list_from_ic_lr_list
from mobgap.laterality.base import BaseLRClassifier, _unify_ic_lr_list_df
from mobgap.pipeline._gs_iterator import FullPipelinePerGsResult, GsIterator
from mobgap.pipeline.base import BaseGaitDatasetT, BaseMobilisedPipeline, mobilised_pipeline_docfiller
from mobgap.stride_length import SlZijlstra
from mobgap.stride_length.base import BaseSlCalculator
from mobgap.turning import TdElGohary
from mobgap.turning.base import BaseTurnDetector
from mobgap.utils.conversions import to_body_frame
from mobgap.utils.df_operations import MultiGroupByPrimaryDfEmptyError, create_multi_groupby
from mobgap.utils.interpolation import naive_sec_paras_to_regions
from mobgap.walking_speed import WsNaive
from mobgap.walking_speed.base import BaseWsCalculator
from mobgap.wba import StrideSelection, WbAssembly


@mobilised_pipeline_docfiller
class GenericMobilisedPipeline(BaseMobilisedPipeline[BaseGaitDatasetT], Generic[BaseGaitDatasetT]):
    """Pipeline structure of the Mobilise-D pipeline without any default algorithms.

    .. warning:: While this class implements the basic structure of the Mobilise-D pipeline, we only consider it "The
             Mobilise-D pipeline" if it is used with the predefined parameters/algorithms for the cohorts these
             parameters are evaluated for.

    This pipeline class can either be used with a custom set of algorithms instances or the "official" predefined
    parameters for healthy or impaired walking (see Examples).
    However, when using the predefined parameters it is recommended to use the separate classes instead
    (:class:`MobilisedPipelineHealthy` and :class:`MobilisedPipelineImpaired`).

    For detailed steps on how this pipeline works, check the Notes section and the dedicated examples.


    Parameters
    ----------
    %(core_parameters)s
    %(turn_detection)s
    %(wba_parameters)s
    %(aggregation_parameters)s
    %(additional_parameters)s

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(primary_results)s
    %(intermediate_results)s
    %(debug_results)s
    %(perf_)s

    Notes
    -----
    %(step_by_step)s

    See Also
    --------
    mobgap.pipeline.MobilisedPipelineHealthy : A predefined pipeline for healthy/mildly impaired walking.
    mobgap.pipeline.MobilisedPipelineImpaired : A predefined pipeline for impaired walking.

    """

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

    datapoint: BaseGaitDatasetT

    # Algos with results
    gait_sequence_detection_: BaseGsDetector
    gs_iterator_: GsIterator[FullPipelinePerGsResult]
    stride_selection_: StrideSelection
    wba_: WbAssembly
    dmo_aggregation_: Optional[BaseAggregator]

    # Intermediate results
    gs_list_: pd.DataFrame
    raw_ic_list_: pd.DataFrame
    raw_turn_list_: pd.DataFrame
    raw_per_sec_parameters_: pd.DataFrame
    raw_per_stride_parameters_: pd.DataFrame

    _all_action_kwargs: dict[str, Any]

    class PredefinedParameters:
        regular_walking: Final = MappingProxyType(
            {
                "gait_sequence_detection": GsdIluz(),
                "initial_contact_detection": IcdIonescu(),
                "laterality_classification": LrcUllrich(**LrcUllrich.PredefinedParameters.msproject_all),
                "cadence_calculation": CadFromIcDetector(
                    **dict(CadFromIcDetector.PredefinedParameters.regular_walking, silence_ic_warning=True)
                ),
                "stride_length_calculation": SlZijlstra(),
                "walking_speed_calculation": WsNaive(),
                "turn_detection": TdElGohary(),
                "stride_selection": StrideSelection(),
                "wba": WbAssembly(),
                "dmo_thresholds": get_mobilised_dmo_thresholds(),
                "dmo_aggregation": MobilisedAggregator(groupby=None),
                "recommended_cohorts": ("HA", "COPD", "CHF"),
            }
        )

        impaired_walking: Final = MappingProxyType(
            {
                "gait_sequence_detection": GsdIonescu(),
                "initial_contact_detection": IcdIonescu(),
                "laterality_classification": LrcUllrich(**LrcUllrich.PredefinedParameters.msproject_all),
                "cadence_calculation": CadFromIcDetector(
                    **dict(CadFromIcDetector.PredefinedParameters.impaired_walking, silence_ic_warning=True)
                ),
                "stride_length_calculation": SlZijlstra(),
                "walking_speed_calculation": WsNaive(),
                "turn_detection": TdElGohary(),
                "stride_selection": StrideSelection(),
                "wba": WbAssembly(),
                "dmo_thresholds": get_mobilised_dmo_thresholds(),
                "dmo_aggregation": MobilisedAggregator(groupby=None),
                "recommended_cohorts": ("PD", "MS", "PFF"),
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
        dmo_aggregation: Optional[BaseAggregator],
        recommended_cohorts: Optional[tuple[str, ...]] = None,
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
        self.recommended_cohorts = recommended_cohorts

    def get_recommended_cohorts(self) -> Optional[tuple[str, ...]]:
        """Get the recommended cohorts for this pipeline.

        Returns
        -------
        recommended_cohorts
            The recommended cohorts for this pipeline or None
        """
        return self.recommended_cohorts

    @timed_action_method
    @mobilised_pipeline_docfiller
    def run(self, datapoint: BaseGaitDatasetT) -> Self:
        """%(run_short)s.

        Parameters
        ----------
        %(run_para)s

        %(run_return)s
        """
        try:
            participant_metadata = datapoint.participant_metadata
        except AttributeError as e:
            raise ValueError(
                "The provided dataset does not provide any participant metadata. "
                "For the default algorithms, metadata is required for the ``stride_length_calculation`` "
                "and ``dmo_thresholds`` step. "
                "If you want to use this pipeline without metadata, please provide custom algorithms and"
                "at least implement the ``participant_metadata`` attribute on your dataset, even if it"
                "just returns an empty dictionary."
            ) from e

        participant_cohort = participant_metadata.get("cohort")

        recommended_cohorts = self.get_recommended_cohorts()

        if recommended_cohorts and participant_cohort not in recommended_cohorts:
            warnings.warn(
                f"The provided datapoint has data of a participant with the cohort {participant_cohort} is not part of "
                "the recommended "
                f"cohorts for this pipeline {type(self).__name__}.\n"
                f"Recommended cohorts are {recommended_cohorts}",
                stacklevel=1,
            )

        self.datapoint = datapoint

        self._all_action_kwargs = {
            **participant_metadata,
            **datapoint.recording_metadata,
            "dp_group": datapoint.group_label,
            "sampling_rate_hz": datapoint.sampling_rate_hz,
        }

        imu_data = to_body_frame(datapoint.data_ss)
        sampling_rate_hz = datapoint.sampling_rate_hz

        self.gait_sequence_detection_ = self.gait_sequence_detection.clone().detect(imu_data, **self._all_action_kwargs)
        self.gs_list_ = self.gait_sequence_detection_.gs_list_
        self.gs_iterator_ = self._run_per_gs(self.gs_list_, imu_data, self._all_action_kwargs)

        results = self.gs_iterator_.results_

        self.raw_per_sec_parameters_ = pd.concat(
            [
                results.cadence_per_sec,
                results.stride_length_per_sec,
                results.walking_speed_per_sec,
            ],
            axis=1,
        )

        if self.raw_per_sec_parameters_.empty:
            expected_results = [
                calc
                for calc, available in [
                    ("cadence_per_sec", self.cadence_calculation),
                    ("stride_length_per_sec", self.stride_length_calculation),
                    ("walking_speed_per_sec", self.walking_speed_calculation),
                ]
                if available
            ]
            index_names = ["gs_id", "r_gs_id", "sec_center_samples"]
            self.raw_per_sec_parameters_ = pd.DataFrame(columns=[*expected_results, *index_names]).set_index(
                index_names
            )

        self.raw_per_sec_parameters_ = self.raw_per_sec_parameters_.reset_index(
            "r_gs_id",
            drop=True,
        )

        if (ic_list := results.ic_list).empty:
            index_names = ["gs_id", "step_id"]
            ic_list = _unify_ic_lr_list_df(
                pd.DataFrame(columns=["ic", "lr_label", *index_names]).set_index(index_names)
            )
        self.raw_ic_list_ = ic_list
        self.raw_turn_list_ = results.turn_list
        self.raw_per_stride_parameters_ = self._sec_to_stride(
            self.raw_per_sec_parameters_, self.raw_ic_list_, sampling_rate_hz
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
            self.stride_selection_.filtered_stride_list_,
            raw_initial_contacts=ic_list,
            sampling_rate_hz=sampling_rate_hz,
        )

        self.per_stride_parameters_ = self.wba_.annotated_stride_list_
        self.per_wb_parameters_ = self._aggregate_per_wb(self.per_stride_parameters_, self.wba_.wb_meta_parameters_)
        if self.dmo_thresholds is None:
            self.per_wb_parameter_mask_ = None
        else:
            if participant_cohort is None:
                raise ValueError(
                    "The cohort of the participant is not provided. "
                    "Please provide the cohort in the participant metadata or set the dmo_thresholds to None."
                )
            assert participant_cohort is not None
            self.per_wb_parameter_mask_ = apply_thresholds(
                self.per_wb_parameters_,
                self.dmo_thresholds,
                cohort=participant_cohort,
                height_m=datapoint.participant_metadata["height_m"],
                measurement_condition=datapoint.recording_metadata["measurement_condition"],
            )

        if self.dmo_aggregation is None:
            self.aggregated_parameters_ = None
            return self

        self.dmo_aggregation_ = self.dmo_aggregation.clone().aggregate(
            self.per_wb_parameters_, wb_dmos_mask=self.per_wb_parameter_mask_
        )
        self.aggregated_parameters_ = self.dmo_aggregation_.aggregated_data_

        del self._all_action_kwargs
        return self

    def _run_per_gs(
        self,
        gait_sequences: pd.DataFrame,
        imu_data: pd.DataFrame,
        action_kwargs: dict[str, Any],
    ) -> GsIterator:
        gs_iterator = GsIterator[FullPipelinePerGsResult]()
        # TODO: How to expose the individual algo instances of the algos that run in the loop?

        for (_, gs_data), r in gs_iterator.iterate(imu_data, gait_sequences):
            icd = self.initial_contact_detection.clone().detect(gs_data, **action_kwargs)
            lrc = self.laterality_classification.clone().predict(gs_data, icd.ic_list_, **action_kwargs)
            r.ic_list = lrc.ic_lr_list_
            if self.turn_detection:
                turn = self.turn_detection.clone().detect(gs_data, **action_kwargs)
                r.turn_list = turn.turn_list_

            refined_gs, refined_ic_list = refine_gs(r.ic_list)

            with gs_iterator.subregion(refined_gs) as ((_, refined_gs_data), rr):
                cad_r = None
                if self.cadence_calculation:
                    cad = self.cadence_calculation.clone().calculate(
                        refined_gs_data,
                        initial_contacts=refined_ic_list,
                        **action_kwargs,
                    )
                    cad_r = cad.cadence_per_sec_
                    rr.cadence_per_sec = cad_r
                sl_r = None
                if self.stride_length_calculation:
                    sl = self.stride_length_calculation.clone().calculate(
                        refined_gs_data, initial_contacts=refined_ic_list, **action_kwargs
                    )
                    sl_r = sl.stride_length_per_sec_
                    rr.stride_length_per_sec = sl.stride_length_per_sec_
                if self.walking_speed_calculation:
                    ws = self.walking_speed_calculation.clone().calculate(
                        refined_gs_data,
                        initial_contacts=refined_ic_list,
                        cadence_per_sec=cad_r,
                        stride_length_per_sec=sl_r,
                        **action_kwargs,
                    )
                    rr.walking_speed_per_sec = ws.walking_speed_per_sec_

        return gs_iterator

    def _sec_to_stride(
        self, sec_level_paras: pd.DataFrame, lr_ic_list: pd.DataFrame, sampling_rate_hz: float
    ) -> pd.DataFrame:
        if lr_ic_list.empty:
            # We still call the function to get the correct index
            # We need to do that in a separate step, as the groupby is not working with an empty dataframe
            stride_list = strides_list_from_ic_lr_list(lr_ic_list)

        else:
            stride_list = lr_ic_list.groupby("gs_id", group_keys=False).apply(strides_list_from_ic_lr_list)
        stride_list = stride_list.assign(stride_duration_s=lambda df_: (df_.end - df_.start) / sampling_rate_hz)

        try:
            stride_list = create_multi_groupby(
                stride_list,
                sec_level_paras,
                "gs_id",
                group_keys=False,
            ).apply(naive_sec_paras_to_regions, sampling_rate_hz=sampling_rate_hz)
        except MultiGroupByPrimaryDfEmptyError:
            # If the stride_list is empty, we cannot create a multi-groupby
            # We still return an empty dataframe with the correct index
            stride_list = stride_list.reindex(columns=[*stride_list.columns, *sec_level_paras.columns])
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
                # TODO: Add "avg_" prefix to the columns
                .mean(),
            ],
            axis=1,
        )


@mobilised_pipeline_docfiller
class MobilisedPipelineHealthy(GenericMobilisedPipeline[BaseGaitDatasetT], Generic[BaseGaitDatasetT]):
    """Official Mobilise-D pipeline for healthy and mildly impaired gait (aka P1 pipeline).

    .. note:: When using this pipeline with its default parameters with healthy participants or participants with COPD
              or congestive heart failure, the use of the name "the Mobilise-D pipeline" is recommended.

    Based on the benchmarking performed in [1]_, the algorithms selected for this pipeline are the optimal choice for
    healthy and mildly impaired gait or more specifically for the cohorts "HA", "COPD", "CHF" within the Mobilise-D
    validation study.
    Performance metrics for the original implementation of this pipeline can be found in [2]_.
    This pipeline is referred to as the "P1" pipeline in the context of this and other publications.

    For detailed steps on how this pipeline works, check the Notes section and the dedicated examples.

    Parameters
    ----------
    %(core_parameters)s
    %(turn_detection)s
    %(wba_parameters)s
    %(aggregation_parameters)s
    %(additional_parameters)s

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(primary_results)s
    %(intermediate_results)s
    %(debug_results)s
    %(perf_)s

    Notes
    -----
    %(step_by_step)s

    .. [1] Micó-Amigo, M., Bonci, T., Paraschiv-Ionescu, A. et al. Assessing real-world gait with digital technology?
           Validation, insights and recommendations from the Mobilise-D consortium. J NeuroEngineering Rehabil 20, 78
           (2023). https://doi.org/10.1186/s12984-023-01198-5
    .. [2] Kirk, C., Küderle, A., Micó-Amigo, M.E. et al. Mobilise-D insights to estimate real-world walking speed in
           multiple conditions with a wearable device. Sci Rep 14, 1754 (2024).
           https://doi.org/10.1038/s41598-024-51766-5

    See Also
    --------
    mobgap.pipeline.BaseMobilisedPipeline : A version of the pipeline without any default algorithms or parameters.
    mobgap.pipeline.MobilisedPipelineImpaired : A predefined pipeline for impaired walking.

    """

    @set_defaults(**{k: cf(v) for k, v in GenericMobilisedPipeline.PredefinedParameters.regular_walking.items()})
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
        recommended_cohorts: Optional[tuple[str, ...]],
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
            recommended_cohorts=recommended_cohorts,
        )


@mobilised_pipeline_docfiller
class MobilisedPipelineImpaired(GenericMobilisedPipeline[BaseGaitDatasetT], Generic[BaseGaitDatasetT]):
    """Official Mobilise-D pipeline for impaired gait (aka P2 pipeline).

    .. note:: When using this pipeline with its default parameters with participants with MS, PD, PFF, the use of the
              name "the Mobilise-D pipeline" is recommended.

    Based on the benchmarking performed in [1]_, the algorithms selected for this pipeline are the optimal choice for
    healthy and mildly impaired gait or more specifically for the cohorts "PD", "MS", "PFF" within the Mobilise-D
    validation study.
    Performance metrics for the original implementation of this pipeline can be found in [2]_.
    This pipeline is referred to as the "P1" pipeline in the context of this and other publications.

    For detailed steps on how this pipeline works, check the Notes section and the dedicated examples.

    Parameters
    ----------
    %(core_parameters)s
    %(turn_detection)s
    %(wba_parameters)s
    %(aggregation_parameters)s
    %(additional_parameters)s

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(primary_results)s
    %(intermediate_results)s
    %(debug_results)s
    %(perf_)s

    Notes
    -----
    %(step_by_step)s

    .. [1] Micó-Amigo, M., Bonci, T., Paraschiv-Ionescu, A. et al. Assessing real-world gait with digital technology?
           Validation, insights and recommendations from the Mobilise-D consortium. J NeuroEngineering Rehabil 20, 78
           (2023). https://doi.org/10.1186/s12984-023-01198-5
    .. [2] Kirk, C., Küderle, A., Micó-Amigo, M.E. et al. Mobilise-D insights to estimate real-world walking speed in
           multiple conditions with a wearable device. Sci Rep 14, 1754 (2024).
           https://doi.org/10.1038/s41598-024-51766-5

    See Also
    --------
    mobgap.pipeline.BaseMobilisedPipeline : A version of the pipeline without any default algorithms or parameters.
    mobgap.pipeline.MobilisedPipelineImpaired : A predefined pipeline for impaired walking.

    """

    @set_defaults(**{k: cf(v) for k, v in GenericMobilisedPipeline.PredefinedParameters.impaired_walking.items()})
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
        recommended_cohorts: Optional[tuple[str, ...]],
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
            recommended_cohorts=recommended_cohorts,
        )


@mobilised_pipeline_docfiller
class MobilisedPipelineUniversal(BaseMobilisedPipeline[BaseGaitDatasetT], Generic[BaseGaitDatasetT]):
    """Metapipeline that can use a specific pipeline depending on the cohort of the participant.

    This uses the ``recommended_cohorts`` parameter of the pipelines to determine which pipeline to use.
    You can provide any list of pipelines with their names to this class.
    However, there must be no overlap in the recommended cohorts of the pipelines.

    The pipeline that is used (and all its results) can be accessed via the ``pipeline_`` attribute.

    Parameters
    ----------
    pipelines
        A list of tuples with the name of the pipeline and the pipeline instance.
        The pipeline that has the cohort in its recommended cohorts will be used.
        If multiple pipelines are recommended for the same cohort an ValueError will be raised.
        If no pipeline is found, a ValueError will be raised.
        By default, the :class:`MobilisedPipelineHealthy` and :class:`MobilisedPipelineImpaired` are used.

    Attributes
    ----------
    %(primary_results)s
    pipeline_
        The pipeline that was used for the provided data with all its results.
    pipeline_name_
        The name of the pipeline that was used.
    %(perf_)s


    Other Parameters
    ----------------
    %(other_parameters)s

    """

    _composite_params = ("pipelines",)

    pipelines: list[tuple[str, BaseMobilisedPipeline[BaseGaitDatasetT]]]

    datapoint: BaseGaitDatasetT

    pipeline_: BaseMobilisedPipeline[BaseGaitDatasetT]
    pipeline_name_: str

    def __init__(
        self,
        pipelines: list[tuple[str, BaseMobilisedPipeline[BaseGaitDatasetT]]] = cf(
            [("healthy", MobilisedPipelineHealthy()), ("impaired", MobilisedPipelineImpaired())]
        ),
    ) -> None:
        self.pipelines = pipelines

    def get_recommended_cohorts(self) -> Optional[tuple[str, ...]]:
        """Get the recommended cohorts for this pipeline.

        Returns
        -------
        recommended_cohorts
            The recommended cohorts for this pipeline or None
        """
        all_cohorts = set()
        for _, pipeline in self.pipelines:
            recommended_cohorts = pipeline.get_recommended_cohorts()
            if recommended_cohorts:
                all_cohorts.update(recommended_cohorts)
        if not all_cohorts:
            return None
        return tuple(all_cohorts)

    @property
    def per_stride_parameters_(self) -> pd.DataFrame:
        return self.pipeline_.per_stride_parameters_

    @property
    def per_wb_parameters_(self) -> pd.DataFrame:
        return self.pipeline_.per_wb_parameters_

    @property
    def per_wb_parameter_mask_(self) -> Optional[pd.DataFrame]:
        return self.pipeline_.per_wb_parameter_mask_

    @property
    def aggregated_parameters_(self) -> Optional[pd.DataFrame]:
        return self.pipeline_.aggregated_parameters_

    @timed_action_method
    @mobilised_pipeline_docfiller
    def run(self, datapoint: BaseGaitDatasetT) -> Self:
        """%(run_short)s.

        Parameters
        ----------
        %(run_para)s

        %(run_return)s
        """
        self.datapoint = datapoint
        # Check if there is overlap in the recommended cohorts
        # We want to find the first pair that has overlaps
        for p1, p2 in combinations(self.pipelines, 2):
            if p1[1].get_recommended_cohorts() and p2[1].get_recommended_cohorts():
                union = set(p1[1].get_recommended_cohorts()) & set(p2[1].get_recommended_cohorts())
                if union:
                    raise ValueError(
                        f"The provided pipelines with the names {p1[0]} and {p2[0]} have an overlap in the recommended "
                        f"cohorts: {union}"
                    )

        cohort = datapoint.participant_metadata["cohort"]
        for name, pipeline in self.pipelines:
            if pipeline.get_recommended_cohorts() and cohort in pipeline.get_recommended_cohorts():
                self.pipeline_ = pipeline.clone().run(datapoint)
                self.pipeline_name_ = name
                return self
        raise ValueError(
            f"Could not determine the correct pipeline for the cohort {cohort}. "
            "Check the ``RECOMMENDED_COHORTS`` attribute of the pipelines."
        )
