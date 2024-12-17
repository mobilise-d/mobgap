import warnings

import numpy as np
import pandas as pd
from tpcp.validate import Scorer, no_agg

from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.initial_contacts.pipeline import IcdEmulationPipeline


def icd_per_datapoint_score(pipeline: IcdEmulationPipeline, datapoint: BaseGaitDatasetWithReference) -> dict:
    """Evaluate the performance of an ICD algorithm on a single datapoint.

    .. warning:: This function is not meant to be called directly, but as a scoring function in a
       :class:`tpcp.validate.Scorer`.
       If you are writing custom scoring functions, you can use this function as a template or wrap it in a new
       function.

    This function is used to evaluate the performance of an ICD algorithm on a single datapoint.
    It calculates the performance metrics based on the detected initial contacts and the reference initial contacts.

    The following performance metrics are calculated:

    - all outputs of :func:`~mobgap.initial_contacts.evaluation.calculate_matched_icd_performance_metrics`
      (will be averaged over all datapoints)
    - ``matches``: The matched initial contacts calculated by
      :func:`~mobgap.initial_contacts.evaluation.categorize_ic_list` (return as ``no_agg``)
    - ``detected``: The detected initial contacts (return as ``no_agg``)
    - ``reference``: The reference initial contacts (return as ``no_agg``)
    - ``sampling_rate_hz``: The sampling rate of the data (return as ``no_agg``)

    Parameters
    ----------
    pipeline
        An instance of ICD emulation pipeline that wraps the algorithm that should be evaluated.
    datapoint
        The datapoint to be evaluated.

    Returns
    -------
    dict
        A dictionary containing the performance metrics.
        Note, that some results are wrapped in a ``no_agg`` object or other aggregators.
        The results of this function are not expected to be parsed manually, but rather the function is expected to be
        used in the context of the :func:`~tpcp.validate.validate`/:func:`~tpcp.validate.cross_validate` functions or
        similar as scorer.
        This functions will aggregate the results and provide a summary of the performance metrics.

    """
    from mobgap.initial_contacts.evaluation import (
        calculate_matched_icd_error,
        calculate_matched_icd_performance_metrics,
        categorize_ic_list,
    )
    from mobgap.utils.conversions import as_samples
    from mobgap.utils.df_operations import create_multi_groupby

    with warnings.catch_warnings():
        # We know that these errors might happen, and they are usually not relevant for the evaluation
        warnings.filterwarnings("ignore", message="Zero division", category=UserWarning)
        warnings.filterwarnings("ignore", message="multiple ICs", category=UserWarning)

        # Run the algorithm on the datapoint
        pipeline.safe_run(datapoint)
        detected_ic_list = pipeline.ic_list_
        reference_ic_list = datapoint.reference_parameters_.ic_list
        sampling_rate_hz = datapoint.sampling_rate_hz

        # tolerance around the reference ic
        tolerance_s = 0.5
        tolerance_samples = as_samples(tolerance_s, sampling_rate_hz)

        # match types
        matches_per_wb = create_multi_groupby(detected_ic_list, reference_ic_list, groupby="wb_id").apply(
            lambda df1, df2: categorize_ic_list(
                ic_list_detected=df1,
                ic_list_reference=df2,
                tolerance_samples=tolerance_samples,
                multiindex_warning=False,
            )
        )
        # check if matches_per_wb has the required columns
        if matches_per_wb.shape[1] == 1:
            # then it is an empty dataframe without required columns
            matches_per_wb = pd.DataFrame(
                {
                    "ic_id_detected": [],
                    "ic_id_reference": [],
                    "match_type": [],
                    "wb_id": [],
                }
            ).set_index(["wb_id"])

        # calculate run time
        try:
            runtime_s = getattr(pipeline.per_wb_algo_[0], "perf_", {}).get("runtime_s", np.nan)
        except KeyError:
            runtime_s = 0

        # Calculate the performance metrics
        performance_metrics = {
            **calculate_matched_icd_performance_metrics(
                matches_per_wb,
            ),
            **calculate_matched_icd_error(
                detected_ic_list,
                reference_ic_list,
                matches_per_wb,
                sampling_rate_hz,
            ),
            "matches": no_agg(matches_per_wb),
            "detected": no_agg(detected_ic_list),
            "reference": no_agg(reference_ic_list),
            "sampling_rate_hz": no_agg(sampling_rate_hz),
            "runtime_s": runtime_s,
        }

    return performance_metrics


def icd_final_agg(
    agg_results: dict[str, float],
    single_results: dict[str, list],
    pipeline: IcdEmulationPipeline,  # noqa: ARG001
    dataset: BaseGaitDatasetWithReference,
) -> tuple[dict[str, any], dict[str, list[any]]]:
    """Aggregate the performance metrics of an ICD algorithm over multiple datapoints.

    .. warning:: This function is not meant to be called directly, but as ``final_aggregator`` in a
       :class:`tpcp.validate.Scorer`.
       If you are writing custom scoring functions, you can use this function as a template or wrap it in a new
       function.

    This function aggregates the performance metrics as follows:

    - All raw outputs (``detected``, ``reference``, ``sampling_rate_hz``) are concatenated to a single
      dataframe, to make it easier to work with and are returned as part of the single results.
    - We recalculate all performance metrics from
      :func:`~mobgap.initial_contacts.evaluation.calculate_matched_icd_performance_metrics` on the combined data.
      The results are prefixed with ``combined__``.
      Compared to the per-datapoint results (which are calculated, as errors per recording -> average over all
      recordings), these metrics are calculated as combining all ICDs from all recordings and then calculating the
      performance metrics.
      Effectively, this means, that in the `per_datapoint` version, each recording is weighted equally, while in the
      `combined` version, each IC is weighted equally.

    Parameters
    ----------
    agg_results
        The aggregated results from all datapoints (see :class:`~tpcp.validate.Scorer`).
    single_results
        The per-datapoint results (see :class:`~tpcp.validate.Scorer`).
    pipeline
        The pipeline that was passed to the scorer.
        This is ignored in this function, but might be useful in custom final aggregators.
    dataset
        The dataset that was passed to the scorer.

    Returns
    -------
    final_agg_results
        The final aggregated results.
    final_single_results
        The per-datapoint results, that are not aggregated.

    """
    from mobgap.initial_contacts.evaluation import (
        calculate_matched_icd_performance_metrics,
    )

    data_labels = [d.group_label for d in dataset]
    data_label_names = data_labels[0]._fields
    # We combine each to a combined dataframe
    matches = single_results.pop("matches")
    matches = pd.concat(matches, keys=data_labels, names=[*data_label_names, *matches[0].index.names])
    detected = single_results.pop("detected")
    detected = pd.concat(detected, keys=data_labels, names=[*data_label_names, *detected[0].index.names])
    reference = single_results.pop("reference")
    reference = pd.concat(reference, keys=data_labels, names=[*data_label_names, *reference[0].index.names])

    aggregated_single_results = {
        "raw__detected": detected,
        "raw__reference": reference,
    }

    sampling_rate_hz = single_results.pop("sampling_rate_hz")
    if set(sampling_rate_hz) != {sampling_rate_hz[0]}:
        raise ValueError(
            "Sampling rate is not the same for all datapoints in the dataset. "
            "This not supported by this scorer. "
            "Provide a custom scorer that can handle this case."
        )

    combined_matched = {f"combined__{k}": v for k, v in calculate_matched_icd_performance_metrics(matches).items()}

    # Note, that we pass the "aggregated_single_results" out via the single results and not the aggregated results
    # The reason is that the aggregated results are expected to be a single value per metric, while the single results
    # can be anything.
    return {**agg_results, **combined_matched}, {**single_results, **aggregated_single_results}


#: :data:: icd_score
#: Scorer class instance for ICD algorithms.
icd_score = Scorer(icd_per_datapoint_score, final_aggregator=icd_final_agg)
icd_score.__doc__ = """Scorer for ICD algorithms.

This is a pre-configured :class:`~tpcp.validate.Scorer` object using the :func:`icd_per_datapoint_score` function as
per-datapoint scorer and the :func:`icd_final_agg` function as final aggregator.
For more information about Scorer, head to the tpcp documentation (:class:`~tpcp.validate.Scorer`).
For usage information in the context of mobgap, have a look at the :ref:`evaluation example <icd_evaluation>` for ICD.

The following metrics are calculated:

Raw metrics (part of the single results):

- ``single__raw__detected``: The detected initial contacts as a single dataframe with the datapoint labels as index.
- ``single__raw__reference``: The reference initial contacts as a single dataframe with the datapoint labels as index.

Metrics per datapoint (single results):
*These values are all provided as a list of values, one per datapoint.*

- All outputs of :func:`~mobgap.initial_contacts.evaluation.calculate_matched_icd_performance_metrics` averaged per
  datapoint. These are stored as ``single__{metric_name}``
- ``single__runtime_s``: The runtime of the algorithm in seconds.

Aggregated metrics (aggregated results):

- All single outputs averaged over all datapoints. These are stored as ``agg__{metric_name}``.
- All metrics from :func:`~mobgap.initial_contacts.evaluation.calculate_matched_icd_performance_metrics` recalculated
  on all detected ICs across all datapoints. These are stored as ``combined__{metric_name}``.
  Compared to the per-datapoint results (which are calculated, as errors per recording -> average over all
  recordings), these metrics are calculated as combining all ICDs from all recordings and then calculating the
  performance metrics.
  Effectively, this means, that in the `per_datapoint` version, each recording is weighted equally, while in the
  `combined` version, each IC is weighted equally.

"""
