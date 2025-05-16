"""Evaluation and scoring helpers for the laterality pipeline."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tpcp.validate import Scorer, no_agg

from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.laterality.pipeline import LrcEmulationPipeline


def lrc_per_datapoint_score(pipeline: LrcEmulationPipeline, datapoint: BaseGaitDatasetWithReference) -> dict:
    """Calculate the accuracy of the LRC pipeline for a single datapoint.

    .. warning:: This function is not meant to be called directly, but as a scoring function in a
       :class:`tpcp.validate.Scorer`.
       If you are writing custom scoring functions, you can use this function as a template or wrap it in a new
       function.

    This function is used to evaluate the performance of an LRC algorithm.
    It calculates the performance by comparing the assigned left right label for each IC in the data.
    We assume, that the algorithm has assigned either "left" or "right" to each IC.
    Two metrics are calculated:

    - "accuracy": The accuracy of the binary classification of the left right labels.
    - "accuracy_pairwise": This accuracy is calculated, by comparing the left right labels of consecutive ICs. This
        means, we check in the reference and the predictions, if the left right label of the current IC is the same as
        the left right label of the previous IC.
        We then calculate the accuracy of these True False labels.
        This is a better metric to understand, if steps and strides would be correctly defined based on the provided
        L/R labels.
        Algorithms that completely switch left and right labels, but still get the order of the labels correct, would
        still get a high accuracy.

    Parameters
    ----------
    pipeline
        An instance of LRC emulation pipeline that wraps the algorithm that should be evaluated.
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
    predicted_lr_labels = pipeline.safe_run(datapoint).ic_lr_list_

    ref_labels = datapoint.reference_parameters_.ic_list["lr_label"]

    combined = predicted_lr_labels.assign(reference=ref_labels).rename(
        columns={
            "lr_label": "predicted",
        }
    )[["predicted", "reference"]]

    # We convert to numpy array explicitly to make some of the later operations faster!
    tmp = combined.to_numpy()
    combined_for_comparison = np.zeros(shape=tmp.shape)
    combined_for_comparison[tmp == "left"] = 1

    # We also calculate the "pairwise" accuracy. Instead of comparing the left right label independently, we compare
    # pares of consecutive labels, if they have the same value.
    # So, LRRLR would be converted to 1011.
    pairwise_predictions = np.abs(np.diff(combined_for_comparison, axis=0))

    return {
        "accuracy": accuracy_score(combined_for_comparison[:, 1], combined_for_comparison[:, 0]),
        "accuracy_pairwise": accuracy_score(pairwise_predictions[:, 1], pairwise_predictions[:, 0]),
        "predictions": no_agg(combined),
    }


def lrc_final_agg(
    agg_results: dict[str, float],
    single_results: dict[str, list],
    pipeline: LrcEmulationPipeline,  # noqa: ARG001
    dataset: BaseGaitDatasetWithReference,
) -> tuple[dict[str, any], dict[str, list[any]]]:
    """Aggregate the results of the LRC pipeline.

    .. warning:: This function is not meant to be called directly, but as ``final_aggregator`` in a
       :class:`tpcp.validate.Scorer`.
       If you are writing custom scoring functions, you can use this function as a template or wrap it in a new
       function.

    This function only applies a single aggregation step beyond the default aggregation.
    It combines the raw `predictions` into a single data frame across all datapoints to make them easier to parse.

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
    data_labels = [d.group_label for d in dataset]
    data_label_names = data_labels[0]._fields

    raw_predictions = single_results.pop("predictions")
    raw_predictions = pd.concat(
        raw_predictions, keys=data_labels, names=[*data_label_names, *raw_predictions[0].index.names]
    )

    return agg_results, {**single_results, "raw__predictions": raw_predictions}


#: :data:: lrc_score
#: Scorer class instance for LRC algorithms.
lrc_score = Scorer(lrc_per_datapoint_score, final_aggregator=lrc_final_agg)
lrc_score.__doc__ = """Scorer for LRC algorithms.

This is a pre-configured :class:`~tpcp.validate.Scorer` object using the :func:`lrc_per_datapoint_score` function as
per-datapoint scorer and the :func:`lrc_final_agg` function as final aggregator.
For more information about Scorer, head to the tpcp documentation (:class:`~tpcp.validate.Scorer`).
For usage information in the context of mobgap, have a look at the :ref:`evaluation example <lrc_evaluation>` for LRC.
"""

__all__ = ["lrc_final_agg", "lrc_per_datapoint_score", "lrc_score"]
