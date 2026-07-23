"""Evaluation and scoring helpers for the reorientation emulation pipeline."""

from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from sklearn.metrics import accuracy_score, confusion_matrix
from tpcp import DatasetWrapperMixin
from tpcp.validate import Scorer, no_agg

from mobgap._gaitmap.utils.rotations import flip_dataset
from mobgap.data.base import (
    IMU_DATA_DTYPE,
    BaseGaitDatasetWithReference,
    ParticipantMetadata,
    RecordingMetadata,
    ReferenceData,
)
from mobgap.re_orientation.pipeline import (
    REORIENTATION_LABELS,
    REORIENTATION_ROTATIONS,
    ReorientationEmulationPipeline,
)
from mobgap.utils.conversions import to_body_frame, to_sensor_frame
from mobgap.utils.dtypes import get_frame_definition

OrientationSpec = Optional[Union[Mapping[str, Rotation], Sequence[str]]]


class MisorientedDataset(
    DatasetWrapperMixin[BaseGaitDatasetWithReference],
    BaseGaitDatasetWithReference,
):
    """Wrap a dataset and simulate mounting orientations per recording.

    The wrapped dataset index is expanded by an additional ``orientation`` column.
    Its existing grouping is preserved and extended with that column.
    Data access delegates to the matching row of the wrapped dataset, converts each
    signal to body frame to apply the selected rough mounting rotation, and returns
    it in the same frame as the wrapped dataset.

    The rotation is intentionally performed outside of the wrapped dataset loading
    cache. This keeps expensive raw data loading reusable while each simulated
    orientation remains a cheap access-time transformation.

    Parameters
    ----------
    wrapped_dataset
        Dataset to wrap.
    orientations
        Either a mapping from orientation labels to rotations, or a sequence of
        labels from
        :data:`mobgap.re_orientation.pipeline.REORIENTATION_ROTATIONS`.
    orientation_col
        Name of the added index column.
    groupby_cols
        Columns to group by. See :class:`tpcp.Dataset`.
    subset_index
        Selected subset of the expanded index. See :class:`tpcp.Dataset`.
    """

    wrapped_dataset: BaseGaitDatasetWithReference
    orientations: OrientationSpec
    orientation_col: str

    def __init__(
        self,
        wrapped_dataset: BaseGaitDatasetWithReference,
        orientations: OrientationSpec = None,
        *,
        orientation_col: str = "orientation",
        groupby_cols: Optional[Union[list[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ) -> None:
        self.wrapped_dataset = wrapped_dataset
        self.orientations = orientations
        self.orientation_col = orientation_col
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def orientation_map(self) -> dict[str, Rotation]:
        """Return the configured orientation labels and rotations."""
        if self.orientations is None:
            return dict(REORIENTATION_ROTATIONS)
        if isinstance(self.orientations, Mapping):
            return dict(self.orientations)
        return {label: REORIENTATION_ROTATIONS[label] for label in self.orientations}

    @property
    def orientation_label(self) -> str:
        """Return simulated orientation label for one dataset subset."""
        self.assert_is_single_group("orientation_label")
        return getattr(self.group_label, self.orientation_col)

    @property
    def orientation_rotation(self) -> Rotation:
        """Return simulated orientation rotation for one dataset subset."""
        return self.orientation_map[self.orientation_label]

    @property
    def _wrapper_groupby_cols(self) -> tuple[str, ...]:
        return (self.orientation_col,)

    def _create_wrapped_index(self, source_index: pd.DataFrame) -> pd.DataFrame:
        """Expand the wrapped dataset index by configured orientation labels."""
        if self.orientation_col in source_index.columns:
            raise ValueError(f"Wrapped dataset already contains an `{self.orientation_col}` column.")

        orientation_labels = list(self.orientation_map)
        expanded_index = source_index.loc[source_index.index.repeat(len(orientation_labels))].reset_index(drop=True)
        expanded_index[self.orientation_col] = orientation_labels * len(source_index)
        return expanded_index

    def _rotate_imu_data(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = get_frame_definition(data, ["sensor", "body"])
        body_frame_data = to_body_frame(data) if frame == "sensor" else data
        rotated = flip_dataset(body_frame_data, self.orientation_rotation)
        # Consumers should still see the same frame as the wrapped dataset. The
        # full pipeline converts sensor-frame data to body frame itself.
        return to_sensor_frame(rotated) if frame == "sensor" else rotated

    @property
    def data(self) -> IMU_DATA_DTYPE:
        """Return all sensor data with simulated orientation applied."""
        return {
            sensor: self._rotate_imu_data(sensor_data) for sensor, sensor_data in self.wrapped_datapoint.data.items()
        }

    @property
    def data_ss(self) -> pd.DataFrame:
        """Return single-sensor data with simulated orientation applied."""
        return self._rotate_imu_data(self.wrapped_datapoint.data_ss)

    @property
    def sampling_rate_hz(self) -> float:
        """Return sampling rate from the wrapped datapoint."""
        return self.wrapped_datapoint.sampling_rate_hz

    @property
    def participant_metadata(self) -> ParticipantMetadata:
        """Return participant metadata from the wrapped datapoint."""
        return self.wrapped_datapoint.participant_metadata

    @property
    def recording_metadata(self) -> RecordingMetadata:
        """Return recording metadata from the wrapped datapoint."""
        return self.wrapped_datapoint.recording_metadata

    @property
    def reference_parameters_(self) -> ReferenceData:
        """Return reference parameters from the wrapped datapoint."""
        return self.wrapped_datapoint.reference_parameters_

    @property
    def reference_parameters_relative_to_wb_(self) -> ReferenceData:
        """Return WB-relative reference parameters from the wrapped datapoint."""
        return self.wrapped_datapoint.reference_parameters_relative_to_wb_

    @property
    def reference_sampling_rate_hz_(self) -> float:
        """Return reference sampling rate from the wrapped datapoint."""
        return self.wrapped_datapoint.reference_sampling_rate_hz_


def _confusion_matrix_as_df(predictions: pd.DataFrame) -> pd.DataFrame:
    known_labels = list(REORIENTATION_LABELS)
    extra_labels = sorted(set(predictions["label"]).union(predictions["prediction"]) - set(known_labels))
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
        "accuracy": accuracy_score(predictions["label"], predictions["prediction"]) if len(predictions) > 0 else np.nan,
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
        combined_accuracy = accuracy_score(raw_predictions["label"], raw_predictions["prediction"])
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
    "MisorientedDataset",
    "reorientation_final_agg",
    "reorientation_per_datapoint_score",
    "reorientation_score",
]
