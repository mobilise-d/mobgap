"""Dataset helpers for full-pipeline orientation robustness validation."""

from collections.abc import Mapping, Sequence
from copy import copy
from typing import Any, Optional, Union

import pandas as pd
from mobgap.data.base import (
    IMU_DATA_DTYPE,
    BaseGaitDatasetWithReference,
    ParticipantMetadata,
    RecordingMetadata,
    ReferenceData,
)
from mobgap.re_orientation.pipeline import REORIENTATION_ROTATIONS
from mobgap.utils.conversions import to_body_frame, to_sensor_frame
from mobgap.utils.dtypes import get_frame_definition
from mobgap.utils.rotations import flip_dataset
from scipy.spatial.transform import Rotation

OrientationSpec = Optional[Union[Mapping[str, Rotation], Sequence[str]]]


class MisorientedDataset(BaseGaitDatasetWithReference):
    """Wrap a dataset and simulate mounting orientations per recording.

    The wrapped dataset's index is expanded by an additional ``orientation``
    column. Data access delegates to the matching row of the wrapped dataset and
    applies the selected rough mounting rotation to the full recording on
    demand.
    The returned data keeps the same frame as the wrapped dataset, so TVS
    ``data_ss`` remains in sensor frame and the full pipeline can perform its
    normal sensor-to-body-frame conversion internally.

    The rotation is intentionally performed outside of the wrapped dataset's
    loading cache, so the expensive raw data loading can still be reused while
    each simulated orientation remains a cheap view-time transformation.

    Parameters
    ----------
    base_dataset
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

    base_dataset: BaseGaitDatasetWithReference
    orientations: OrientationSpec
    orientation_col: str

    def __init__(
        self,
        base_dataset: BaseGaitDatasetWithReference,
        orientations: OrientationSpec = None,
        *,
        orientation_col: str = "orientation",
        groupby_cols: Optional[Union[list[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ) -> None:
        self.base_dataset = base_dataset
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
        return {
            label: REORIENTATION_ROTATIONS[label] for label in self.orientations
        }

    @property
    def orientation_label(self) -> str:
        """Return simulated orientation label for one dataset subset."""
        self.assert_is_single(None, "orientation_label")
        return getattr(self.group_label, self.orientation_col)

    @property
    def orientation_rotation(self) -> Rotation:
        """Return simulated orientation rotation for one dataset subset."""
        return self.orientation_map[self.orientation_label]

    def create_index(self) -> pd.DataFrame:
        """Expand the wrapped dataset index by configured orientation labels."""
        base_index = self.base_dataset.index
        if self.orientation_col in base_index.columns:
            raise ValueError(
                "Wrapped dataset already contains an "
                f"`{self.orientation_col}` column."
            )

        orientation_labels = list(self.orientation_map)
        expanded_index = base_index.loc[
            base_index.index.repeat(len(orientation_labels))
        ].reset_index(drop=True)
        expanded_index[self.orientation_col] = orientation_labels * len(
            base_index
        )
        return expanded_index

    @property
    def _base_datapoint(self) -> BaseGaitDatasetWithReference:
        self.assert_is_single(None, "_base_datapoint")
        base_index = self.index.drop(columns=self.orientation_col)
        return self.base_dataset.get_subset(index=base_index)

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
            sensor: self._rotate_imu_data(sensor_data)
            for sensor, sensor_data in self._base_datapoint.data.items()
        }

    @property
    def data_ss(self) -> pd.DataFrame:
        """Return single-sensor data with simulated orientation applied."""
        return self._rotate_imu_data(self._base_datapoint.data_ss)

    @property
    def sampling_rate_hz(self) -> float:
        return self._base_datapoint.sampling_rate_hz

    @property
    def participant_metadata(self) -> ParticipantMetadata:
        return self._base_datapoint.participant_metadata

    @property
    def recording_metadata(self) -> RecordingMetadata:
        return self._base_datapoint.recording_metadata

    @property
    def reference_parameters_(self) -> ReferenceData:
        return self._base_datapoint.reference_parameters_

    @property
    def reference_parameters_relative_to_wb_(self) -> ReferenceData:
        return self._base_datapoint.reference_parameters_relative_to_wb_

    @property
    def reference_sampling_rate_hz_(self) -> float:
        return self._base_datapoint.reference_sampling_rate_hz_

    @classmethod
    def __clone_param__(cls, param_name: str, value: Any) -> Any:
        if param_name == "base_dataset":
            return value
        if param_name == "orientations":
            return copy(value)
        return super().__clone_param__(param_name, value)


__all__ = ["MisorientedDataset"]
