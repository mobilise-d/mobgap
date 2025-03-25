"""Evaluation and scoring helpers for the laterality pipeline."""

from sklearn.metrics import accuracy_score
from tpcp.validate import Scorer, no_agg

from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.laterality.pipeline import LrcEmulationPipeline


def per_datapoint_scorer(pipeline: LrcEmulationPipeline, datapoint: BaseGaitDatasetWithReference):
    predicted_lr_labels = pipeline.safe_run(datapoint).ic_lr_list_

    ref_labels = datapoint.reference_parameters_.ic_list["lr_label"]

    combined = predicted_lr_labels.assign(ref_lr_label=ref_labels)

    return {
        "accuracy": accuracy_score(ref_labels, predicted_lr_labels["lr_label"]),
        "raw__results": no_agg(combined),
    }


lrc_score = Scorer(per_datapoint_scorer)


__all__ = ["lrc_score"]
