"""Evaluation and scoring helpers for the reorientation emulation pipeline."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from tpcp.validate import Scorer, no_agg

from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.re_orientation.pipeline import (
    REORIENTATION_LABELS,
    ReorientationEmulationPipeline,
)


def _confusion_matrix_as_df(predictions: pd.DataFrame) -> pd.DataFrame:
    known_labels = list(REORIENTATION_LABELS)
    extra_labels = sorted(
        set(predictions["label"]).union(predictions["prediction"]) - set(known_labels)
    )
    labels = [*known_labels, *extra_labels]

    return pd.DataFrame(
        confusion_matrix(
            predictions["label"],
            predictions["prediction"],
            labels=labels,
        ),
        index=pd.Index(labels, name="label"),
        columns=pd.Index(labels, name="prediction"),
    )


def reorientation_per_datapoint_score(
    pipeline: ReorientationEmulationPipeline,
    datapoint: BaseGaitDatasetWithReference,
) -> dict[str, Any]:
    """Calculate multiclass orientation-class accuracy for one datapoint.

    .. warning:: This function is not meant to be called directly, but as a scoring
       function in a :class:`tpcp.validate.Scorer`.

    The wrapped pipeline simulates all supported rough sensor rotations on every
    reference walking bout and returns one prediction for each simulated class. This
    scorer treats the result as a multiclass classification task.
    """
    predictions = pipeline.safe_run(datapoint).predictions_

    return {
        "accuracy": accuracy_score(predictions["label"], predictions["prediction"])
        if len(predictions) > 0
        else np.nan,
        "predictions": no_agg(predictions),
    }


def reorientation_final_agg(
    agg_results: dict[str, float],
    single_results: dict[str, list],
    pipeline: ReorientationEmulationPipeline,  # noqa: ARG001
    dataset: BaseGaitDatasetWithReference,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate the results of the reorientation emulation pipeline.

    The final aggregation combines the raw predictions across all datapoints,
    recalculates multiclass accuracy over every simulated walking-bout orientation,
    and exposes the combined confusion matrix as a raw result.
    """
    data_labels = [d.group_label for d in dataset]
    data_label_names = data_labels[0]._fields

    raw_predictions_list = single_results.pop("predictions")
    raw_predictions = pd.concat(
        raw_predictions_list,
        keys=data_labels,
        names=[*data_label_names, *raw_predictions_list[0].index.names],
    )

    if len(raw_predictions) > 0:
        combined_accuracy = accuracy_score(
            raw_predictions["label"], raw_predictions["prediction"]
        )
        confusion_matrix_df = _confusion_matrix_as_df(raw_predictions)
    else:
        combined_accuracy = np.nan
        confusion_matrix_df = pd.DataFrame(
            0,
            index=pd.Index(REORIENTATION_LABELS, name="label"),
            columns=pd.Index(REORIENTATION_LABELS, name="prediction"),
        )

    return (
        {**agg_results, "combined__accuracy": combined_accuracy},
        {
            **single_results,
            "raw__predictions": raw_predictions,
            "raw__confusion_matrix": confusion_matrix_df,
        },
    )


reorientation_score = Scorer(
    reorientation_per_datapoint_score,
    final_aggregator=reorientation_final_agg,
)
reorientation_score.__doc__ = """Scorer for reorientation algorithms.

This is a pre-configured :class:`~tpcp.validate.Scorer` object using the
:func:`reorientation_per_datapoint_score` function as per-datapoint scorer and
the :func:`reorientation_final_agg` function as final aggregator.
"""


__all__ = [
    "reorientation_final_agg",
    "reorientation_per_datapoint_score",
    "reorientation_score",
]
