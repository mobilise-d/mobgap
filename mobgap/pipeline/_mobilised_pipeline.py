import warnings
from itertools import combinations
from types import MappingProxyType
from typing import Final, Generic, Optional, TypeVar

import pandas as pd
from tpcp import Pipeline, cf
from tpcp.misc import set_defaults
from typing_extensions import Self

from mobgap._docutils import make_filldoc
from mobgap.aggregation import MobilisedAggregator, apply_thresholds, get_mobilised_dmo_thresholds
from mobgap.aggregation.base import BaseAggregator
from mobgap.cadence import CadFromIcDetector
from mobgap.cadence.base import BaseCadCalculator
from mobgap.data.base import BaseGaitDataset, ParticipantMetadata
from mobgap.gait_sequences import GsdIluz, GsdIonescu
from mobgap.gait_sequences.base import BaseGsDetector
from mobgap.initial_contacts import IcdHKLeeImproved, IcdIonescu, IcdShinImproved, refine_gs
from mobgap.initial_contacts.base import BaseIcDetector
from mobgap.laterality import LrcUllrich, strides_list_from_ic_lr_list
from mobgap.laterality.base import BaseLRClassifier, _unify_ic_lr_list_df
from mobgap.pipeline._gs_iterator import FullPipelinePerGsResult, GsIterator
from mobgap.stride_length import SlZijlstra
from mobgap.stride_length.base import BaseSlCalculator
from mobgap.turning import TdElGohary
from mobgap.turning.base import BaseTurnDetector
from mobgap.utils.df_operations import create_multi_groupby
from mobgap.utils.interpolation import naive_sec_paras_to_regions
from mobgap.walking_speed import WsNaive
from mobgap.walking_speed.base import BaseWsCalculator
from mobgap.wba import StrideSelection, WbAssembly

mobilsed_pipeline_docfiller = make_filldoc(
    {
        "run_short": "Run the pipeline on the provided data.",
        "run_para": """
    datapoint
        The data to run the pipeline on.
        This needs to be a valid datapoint (i.e. a dataset with just a single row).
        The Dataset should be a child class of :class:`~mobgap.data.base.BaseGaitDataset` or implement all the same
        parameters and methods.
    """,
        "run_return": """
    Returns
    -------
    self
        The pipeline object itself with all the results stored in the attributes.
    """,
        "core_parameters": """
    gait_sequence_detection
        A valid instance of a gait sequence detection algorithm.
        This will get the entire raw data as input.
        The core output is available via the ``gs_list_`` attribute.
    initial_contact_detection
        A valid instance of an initial contact detection algorithm.
        This will run on each gait sequence individually.
        The concatenated raw ICs are available via the ``raw_ic_list_`` attribute.
    laterality_classification
        A valid instance of a laterality classification algorithm.
        This will run on each gait sequence individually, getting the predicted ICs from the IC detection algorithm as
        input.
        The concatenated raw ICs with L/R label are available via the ``raw_ic_list_`` attribute.
    cadence_calculation
        A valid instance of a cadence calculation algorithm.
        This will run on each "refined" gait sequence individually.
        This means the provided gait sequence will start and end at the first and last detected IC.
        The detected ICs (with L/R label) and all :class:`~mobgap.data.base.ParticipantMetadata` parameters are provided
        as keyword arguments.
        The concatenated raw cadence per second values are available via the ``raw_per_sec_parameters_`` attribute.
    stride_length_calculation
        A valid instance of a stride length calculation algorithm.
        This will run on each "refined" gait sequence individually.
        This means the provided gait sequence will start and end at the first and last detected IC.
        The detected ICs (with L/R label) and all :class:`~mobgap.data.base.ParticipantMetadata` parameters are provided
        as keyword arguments.
        The concatenated raw stride length per second values are available via the ``raw_per_sec_parameters_``
        attribute.
    walking_speed_calculation
        A valid instance of a walking speed calculation algorithm.
        This will run on each "refined" gait sequence individually.
        This means the provided gait sequence will start and end at the first and last detected IC.
        The detected ICs (with L/R label), cadence per second, stride length per second values and all
        :class:`~mobgap.data.base.ParticipantMetadata` parameters are provided as keyword arguments.
        The concatenated raw walking speed per second values are available via the ``raw_per_sec_parameters_``
        attribute.

        .. note :: If either cadence or stride length is not provided, ``None`` will be passed to the algorithm.
                   Depending on the algorithm, this might raise an error, as the information is required.
    """,
        "turn_detection": """
    turn_detection
        A valid instance of a turn detection algorithm.
        This will run on each gait sequence individually.
        The concatenated raw turn detections are available via the ``raw_turn_list_`` attribute.
    """,
        "wba_parameters": """
    stride_selection
        A valid instance of a stride selection algorithm.
        This will be called with all interpolated stride parameters (``raw_per_stride_parameters_``) across all gait
        sequences.
    wba
        A valid instance of a walking bout assembly algorithm.
        This will be called with the filtered stride list from the stride selection algorithm.
        The final list of strides that are part of a valid WB are available via the ``per_stride_parameters_``
        attribute.
        The aggregated parameters for each WB are available via the ``per_wb_parameters_`` attribute.
    """,
        "aggregation_parameters": """
    dmo_thresholds
        A DataFrame with the thresholds for the individual DMOs.
        To learn more about the required structure and the filtering process, please refer to the documentation of the
        :func:`~mobgap.aggregation.get_mobilised_dmo_thresholds` and :func:`~mobgap.aggregation.apply_thresholds`.
    dmo_aggregation
        A valid instance of a DMO aggregation algorithm.
        This will be called with the aggregated parameters for each WB and the mask of the DMOs.
        The final aggregated parameters are available via the ``aggregated_parameters_`` attribute.
    """,
        "additional_parameters": """
    recommended_cohorts
        A tuple of recommended cohorts for this pipeline.
        If a datapoint is provided with a cohort that is not part of this tuple, a warning will be raised.
        This can also be used in combination with the :class:`MobilisedMetaPipeline` to conditionally run a specific
        pipeline based on the cohort of the participant.
    """,
        "other_parameters": """
    datapoint
        The dataset instance passed to the run method.
    """,
        "primary_results": """
    per_stride_parameters_
        The final list of all strides including their parameters that are part of a valid WB.
        Note, that all per-stride parameters are interpolated based on the per-sec output of the other algorithms.
        Check out the pipeline examples to learn more about this.
    per_wb_parameters_
        Aggregated parameters for each WB.
        This contains "meta parameters" like the number of strides, duration of the WB and the average over all strides
        of cadence, stride length and walking speed (if calculated).
    per_wb_parameter_mask_
        A "valid" mask calculated using the :func:`~mobgap.aggregation.apply_thresholds` function.
        It indicates for each WB which DMOs are valid.
        NaN indicates that the value has not been checked
    aggregated_parameters_
        The final aggregated parameters.
        They are calculated based on the per WB parameters and the DMO mask.
        Invalid parameters are (depending on the implementation in the provided Aggregation algorithm) excluded.
        This output can either be a dataframe with a single row (all WBs were aggregated to a single value, default),
        or a dataframe with multiple rows, if the aggregation algorithm uses a different aggregation approach.
    """,
        "intermediate_results": """
    gs_list_
        The raw output of the gait sequence detection algorithm.
        This is a DataFrame with the start and end of each detected gait sequence.
    raw_ic_list_
        The raw output of the IC detection and the laterality classification.
        This is a DataFrame with the detected ICs and the corresponding L/R label.
    raw_turn_list_
        The raw output of the turn detection algorithm.
        This is a DataFrame with the detected turns (start, end, angle, ...).
    raw_per_sec_parameters_
        A concatenated dataframe with all calculated per-second parameters.
        The index represents the sample of the center of the second the parameter value belongs to.
    raw_per_stride_parameters_
        A concatenated dataframe with all calculated per-stride parameters and the general stride information (start,
        end, laterality).
    """,
        "debug_results": """
    gait_sequence_detection_
        The instance of the gait sequence detection algorithm that was run with all of its results.
    gs_iterator_
        The instance of the GS iterator that was run with all of its results.
        This contains the raw results for each GS, as well as the information about the constrained gs.
        These raw results (inputs and outputs per GS) can be used to test run individual algorithms exactly like they
        were run within the pipeline.
    stride_selection_
        The instance of the stride selection algorithm that was run with all of its results.
    wba_
        The instance of the WBA algorithm that was run with all of its results.
    dmo_aggregation_
        The instance of the DMO aggregation algorithm that was run with all of its results.
    """,
        "step_by_step": """
    The Mobilise-D pipeline consists of the following steps:

    1. Gait sequences are detected using the provided gait sequence detection algorithm.
    2. Within each gait sequence, initial contacts are detected using the provided IC detection algorithm.
       A "refined" version of the gait sequence is created, starting and ending at the first and last detected IC.
    3. Cadence, stride length and walking speed are calculated for each "refined" gait sequence.
       The output of these algorithms is provided per second.
    4. Using the L/R label for each IC calculated by the laterality classification algorithm, strides are defined.
    5. The per-second parameters are interpolated to per-stride parameters.
    6. The stride selection algorithm is used to filter out strides that don't fulfill certain criteria.
    7. The WBA algorithm is used to assemble the strides into walking bouts.
       This is done independent of the original gait sequences.
    8. Aggregated parameters for each WB are calculated.
    9. If DMO thresholds are provided, these WB-level parameters are filtered based on physiological valid thresholds.
    10. The DMO aggregation algorithm is used to aggregate the WB-level parameters to either a set of values
        per-recording or any other granularity (i.e. one value per hour), depending on the aggregation algorithm.

    For a step-by-step example of how these steps are executed, check out :ref:`mobilised_pipeline_step_by_step`.
    """,
    }
)

BaseGaitDatasetT = TypeVar("BaseGaitDatasetT", bound=BaseGaitDataset)


@mobilsed_pipeline_docfiller
class BaseMobilisedPipeline(Pipeline[BaseGaitDatasetT], Generic[BaseGaitDatasetT]):
    """Basic Pipeline structure of the Mobilise-D pipeline.

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
    recommended_cohorts: Optional[tuple[str, ...]]

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

    # Final Results
    per_stride_parameters_: pd.DataFrame
    per_wb_parameters_: pd.DataFrame
    per_wb_parameter_mask_: Optional[pd.DataFrame]
    aggregated_parameters_: Optional[pd.DataFrame]

    class PredefinedParameters:
        regular_walking: Final = MappingProxyType(
            {
                "gait_sequence_detection": GsdIluz(),
                "initial_contact_detection": IcdIonescu(),
                "laterality_classification": LrcUllrich(**LrcUllrich.PredefinedParameters.msproject_all),
                "cadence_calculation": CadFromIcDetector(IcdShinImproved(), silence_ic_warning=True),
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
                "cadence_calculation": CadFromIcDetector(IcdHKLeeImproved(), silence_ic_warning=True),
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

    @mobilsed_pipeline_docfiller
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

        if self.recommended_cohorts and participant_cohort not in self.recommended_cohorts:
            warnings.warn(
                f"The provided datapoint has data of a participant with the cohort {participant_cohort} is not part of "
                "the recommended "
                f"cohorts for this pipeline {type(self).__name__}.\n"
                f"Recommended cohorts are {self.recommended_cohorts}",
                stacklevel=1,
            )

        self.datapoint = datapoint

        imu_data = datapoint.data_ss
        sampling_rate_hz = datapoint.sampling_rate_hz

        self.gait_sequence_detection_ = self.gait_sequence_detection.clone().detect(
            imu_data, sampling_rate_hz=sampling_rate_hz
        )
        self.gs_list_ = self.gait_sequence_detection_.gs_list_
        self.gs_iterator_ = self._run_per_gs(self.gs_list_, imu_data, sampling_rate_hz, participant_metadata)

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
            self.stride_selection_.filtered_stride_list_, sampling_rate_hz=sampling_rate_hz
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
        if lr_ic_list.empty:
            # We still call the function to get the correct index
            # We need to do that in a separate step, as the groupby is not working with an empty dataframe
            stride_list = strides_list_from_ic_lr_list(lr_ic_list)

        else:
            stride_list = lr_ic_list.groupby("gs_id", group_keys=False).apply(strides_list_from_ic_lr_list)
        stride_list = stride_list.assign(stride_duration_s=lambda df_: (df_.end - df_.start) / sampling_rate_hz)

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
                # TODO: Add "avg_" prefix to the columns
                .mean(),
            ],
            axis=1,
        )


@mobilsed_pipeline_docfiller
class MobilisedPipelineHealthy(BaseMobilisedPipeline[BaseGaitDatasetT], Generic[BaseGaitDatasetT]):
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

    @set_defaults(**{k: cf(v) for k, v in BaseMobilisedPipeline.PredefinedParameters.regular_walking.items()})
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


@mobilsed_pipeline_docfiller
class MobilisedPipelineImpaired(BaseMobilisedPipeline[BaseGaitDatasetT], Generic[BaseGaitDatasetT]):
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

    @set_defaults(**{k: cf(v) for k, v in BaseMobilisedPipeline.PredefinedParameters.impaired_walking.items()})
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


@mobilsed_pipeline_docfiller
class MobilisedMetaPipeline(Pipeline[BaseGaitDatasetT], Generic[BaseGaitDatasetT]):
    """Metapipeline that can use a specific pipeline depending on the cohort of the participant.

    This uses the ``RECOMMENDED_COHORTS`` attribute of the pipelines to determine which pipeline to use.
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
    pipeline_
        The pipeline that was used for the provided data with all its results.
    pipeline_name_
        The name of the pipeline that was used.

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

    @mobilsed_pipeline_docfiller
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
            if p1[1].recommended_cohorts and p2[1].recommended_cohorts:
                union = set(p1[1].recommended_cohorts) & set(p2[1].recommended_cohorts)
                if union:
                    raise ValueError(
                        f"The provided pipelines with the names {p1[0]} and {p2[0]} have an overlap in the recommended "
                        f"cohorts: {union}"
                    )

        cohort = datapoint.participant_metadata["cohort"]
        for name, pipeline in self.pipelines:
            if pipeline.recommended_cohorts and cohort in pipeline.recommended_cohorts:
                self.pipeline_ = pipeline.clone().run(datapoint)
                self.pipeline_name_ = name
                return self
        raise ValueError(
            f"Could not determine the correct pipeline for the cohort {cohort}. "
            "Check the ``RECOMMENDED_COHORTS`` attribute of the pipelines."
        )
