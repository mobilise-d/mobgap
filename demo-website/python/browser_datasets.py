"""File-backed datasets for the browser's uploaded recordings."""

from pathlib import Path
from typing import Any

import pandas as pd
from mobgap.data import GenericMobilisedDataset, load_mobilised_participant_metadata_file


class UploadedMatlabDataset(GenericMobilisedDataset):
    """Use the standard MATLAB loader with upload-specific metadata locations.

    Upload filenames and folders carry no cohort information. A companion path
    is supplied only after the UI adapter has established an unambiguous pairing.
    Manual participant metadata takes precedence over file metadata.
    """

    def __init__(
        self,
        path: Path,
        *,
        metadata_path: Path | None = None,
        participant_metadata_override: dict[str, Any] | None = None,
        measurement_condition: str = "laboratory",
        groupby_cols: list[str] | str | None = None,
        subset_index: pd.DataFrame | None = None,
    ) -> None:
        self.path = path
        self.metadata_path = metadata_path
        self.participant_metadata_override = participant_metadata_override
        super().__init__(
            path,
            test_level_names=(),
            measurement_condition=measurement_condition,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
        )

    @property
    def _paths_list(self) -> list[Path]:
        return [self.path]

    @property
    def selected_meta_data_file(self) -> Path:
        if self.metadata_path is None:
            raise FileNotFoundError("No participant metadata file was paired with this upload.")
        return self.metadata_path

    @property
    def _test_level_names(self) -> tuple[str, ...]:
        # The standard dataset loader discovers the actual trial hierarchy.
        tests = self._cached_data_load_no_checks(self.path)[0]
        if not tests:
            raise ValueError("The MAT file contains no Mobilise-D recordings.")
        depth = len(next(iter(tests)))
        return {2: ("time_measure", "recording"), 3: ("time_measure", "test", "trial")}.get(
            depth, tuple(f"level_{i}" for i in range(depth))
        )

    @property
    def participant_metadata(self) -> dict[str, Any]:
        self.assert_is_single(None, "participant_metadata")
        if self.participant_metadata_override is not None:
            return self.participant_metadata_override
        if self.metadata_path is None:
            return {}
        # Reuse the public metadata loader while allowing optional/missing heights.
        # No cohort is inferred from browser upload paths.
        metadata = load_mobilised_participant_metadata_file(self.selected_meta_data_file)
        selected = metadata.get(self.index.iloc[0, 0], {})
        return {
            target: selected[source] / 100
            for source, target in (("SensorHeight", "sensor_height_m"), ("Height", "height_m"))
            if isinstance(selected.get(source), (int, float))
        }
