import warnings

import pandas as pd
from tpcp.validate import Scorer, no_agg

from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.gait_sequences.pipeline import GsdEmulationPipeline


def gsd_per_datapoint_score(pipeline: GsdEmulationPipeline, datapoint: BaseGaitDatasetWithReference) -> dict:
    """Evaluate the performance of a GSD algorithm on a single datapoint.

    This function is used to evaluate the performance of a GSD algorithm on a single datapoint.
    It calculates the performance metrics based on the detected gait sequences and the reference gait sequences.

    Parameters
    ----------
    pipeline
        An instance of GSD emulation pipeline that wraps the algorithm that should be evaluated.
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
    from mobgap.gait_sequences.evaluation import (
        calculate_matched_gsd_performance_metrics,
        calculate_unmatched_gsd_performance_metrics,
        categorize_intervals_per_sample,
    )

    with warnings.catch_warnings():
        # We know that these errors might happen, and they are usually not relevant for the evaluation
        warnings.filterwarnings("ignore", message="Zero division", category=UserWarning)
        warnings.filterwarnings("ignore", message="multiple ICs", category=UserWarning)

        # Run the algorithm on the datapoint
        detected_gs_list = pipeline.safe_run(datapoint).gs_list_
        reference_gs_list = datapoint.reference_parameters_.wb_list[["start", "end"]]
        n_overall_samples = len(datapoint.data_ss)
        sampling_rate_hz = datapoint.sampling_rate_hz

        matches = categorize_intervals_per_sample(
            gsd_list_detected=detected_gs_list,
            gsd_list_reference=reference_gs_list,
            n_overall_samples=n_overall_samples,
        )

        # Calculate the performance metrics
        performance_metrics = {
            **calculate_unmatched_gsd_performance_metrics(
                gsd_list_detected=detected_gs_list,
                gsd_list_reference=reference_gs_list,
                sampling_rate_hz=sampling_rate_hz,
            ),
            **calculate_matched_gsd_performance_metrics(matches),
            "matches": no_agg(matches),
            "detected": no_agg(detected_gs_list),
            "reference": no_agg(reference_gs_list),
            "sampling_rate_hz": no_agg(sampling_rate_hz),
        }

    return performance_metrics


def gsd_final_agg(
    agg_results: dict[str, float],
    single_results: dict[str, list],
    _: GsdEmulationPipeline,
    dataset: BaseGaitDatasetWithReference,
):
    from mobgap.gait_sequences.evaluation import (
        calculate_matched_gsd_performance_metrics,
        calculate_unmatched_gsd_performance_metrics,
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

    combined_unmatched = {
        f"combined__{k}": v
        for k, v in calculate_unmatched_gsd_performance_metrics(
            gsd_list_detected=detected,
            gsd_list_reference=reference,
            sampling_rate_hz=sampling_rate_hz[0],
        ).items()
    }
    combined_matched = {f"combined__{k}": v for k, v in calculate_matched_gsd_performance_metrics(matches).items()}

    return {**aggregated_single_results, **agg_results, **combined_unmatched, **combined_matched}, single_results


gsd_score = Scorer(gsd_per_datapoint_score, final_aggregator=gsd_final_agg)
