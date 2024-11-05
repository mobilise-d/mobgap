"""Evaluation and scoring helpers for the laterality pipeline."""

from sklearn.metrics import accuracy_score
from tpcp.validate import Scorer, no_agg

from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.laterality.pipeline import LrcEmulationPipeline


def laterality_evaluation_scorer(pipeline: LrcEmulationPipeline, datapoint: BaseGaitDatasetWithReference):
    """Score the pipeline on a single datapoint.

    This runs ``algo`` on the provided datapoint and returns the accuracy and the raw classified labels.

    This method should be used in combination with the scoring/validation methods available in ``tpcp.optimize``

    Parameters
    ----------
    datapoint
        A single datapoint of a Gait Dataset with reference information.

    Returns
    -------
    metrics
        A dictionary with relevant performance metrics

    """
    predicted_lr_labels = pipeline.safe_run(datapoint).ic_lr_list_

    ref_labels = datapoint.reference_parameters_.ic_list["lr_label"]

    combined = predicted_lr_labels.assign(ref_lr_label=ref_labels)

    return {
        "accuracy": accuracy_score(ref_labels, predicted_lr_labels["lr_label"]),
        "raw_results": no_agg(combined),
    }

__all__ = ["laterality_evaluation_scorer"]
