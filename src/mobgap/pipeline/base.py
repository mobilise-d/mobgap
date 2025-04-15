"""Base typing interface for Mobilised Pipelines."""

from typing import Generic, Optional, TypeVar

import pandas as pd
from tpcp import Pipeline

from mobgap._docutils import make_filldoc
from mobgap._utils_internal.misc import MeasureTimeResults, timer_doc_filler
from mobgap.data.base import BaseGaitDataset

mobilised_pipeline_docfiller = make_filldoc(
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
    | timer_doc_filler._dict,
)

BaseGaitDatasetT = TypeVar("BaseGaitDatasetT", bound=BaseGaitDataset)


class BaseMobilisedPipeline(Pipeline[BaseGaitDatasetT], Generic[BaseGaitDatasetT]):
    """Base typing interface for Mobilised Pipelines.

    This only defines the main attributes and methods, we expect the pipeline to have.
    For the actual implementation of the pipeline, see :class:`~mobgab.pipeline.GenericMobilisedPipeline`.


    Attributes
    ----------
    %(primary_results)s
    %(perf_)s

    See Also
    --------
    mobgap.pipeline.GenericMobilisedPipeline : The generic pipeline without predefined algorithms.
    mobgap.pipeline.MobilisedPipelineHealthy : A predefined pipeline for healthy/mildly impaired walking.
    mobgap.pipeline.MobilisedPipelineImpaired : A predefined pipeline for impaired walking.

    """

    per_stride_parameters_: pd.DataFrame
    per_wb_parameters_: pd.DataFrame
    per_wb_parameter_mask_: Optional[pd.DataFrame]
    aggregated_parameters_: Optional[pd.DataFrame]

    perf_: MeasureTimeResults

    def get_recommended_cohorts(self) -> Optional[tuple[str, ...]]:
        """Get the recommended cohorts for this pipeline.

        Returns
        -------
        recommended_cohorts
            The recommended cohorts for this pipeline or None
        """
        raise NotImplementedError


__all__ = ["BaseGaitDatasetT", "BaseMobilisedPipeline", "mobilised_pipeline_docfiller"]
