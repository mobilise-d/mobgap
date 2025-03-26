import warnings
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Literal, Optional, Union

import joblib
import pandas as pd

from mobgap.data._mobilised_matlab_loader import (
    BaseGenericMobilisedDataset,
    GenericMobilisedDataset,
    ReferenceData,
    matlab_dataset_docfiller,
)
from mobgap.data.base import ParticipantMetadata, RecordingMetadata


@lru_cache(maxsize=1)
def _load_participant_metadata(base_path: Path) -> pd.DataFrame:
    return (
        pd.read_excel(base_path / "Clinical Info.xlsx", header=0)
        .rename(
            columns={
                "ID": "participant_id",
                "Population_t1": "cohort",
            }
        )
        .assign(cohort=lambda df_: df_.cohort.replace({"CTRL": "HA"}))
        .set_index("participant_id")
    )


@matlab_dataset_docfiller
class MsProjectDataset(BaseGenericMobilisedDataset):
    """Dataset class for the MS Project dataset.

    This dataset was used in Mobilise-D as part of the pre-technical validation and is used to train the
    :class:`~mobgap.laterality.LrcUllrich` algorithm.
    The dataset was originally recorded by the University of Sheffield as part of the MSProject project.
    (TODO: Add citation).
    The data was then standardized into the Mobilise-D matlab format as part of the Mobilise-D project.

    .. warning:: This dataset does not have all relevant metadata available to run the full Mobilise-D pipeline.
       We allow setting a default constant value for the missing metadata in the init.
       Alternatively you can overwrite the respective dataset property.


    Parameters
    ----------
    base_path
        The path to the folder containing the data.
    default_participant_height_m
        The default height of the participants in meters.
        This is used, as the dataset does not contain the participant height.
        From the participant height, the sensor height is derived as 0.53 * height.
        If None is provided for this parameter, downstream algorithms will likely fail.
    %(file_loader_args)s
    %(dataset_memory_args)s
    %(general_dataset_args)s

    Attributes
    ----------
    %(dataset_data_attrs)s

    """

    base_path: Union[Path, str]
    default_participant_height_m: Optional[float]

    reference_system: Optional[Literal["SU_LowerShanks"]]
    raw_data_sensor: Literal["SU"]

    _not_expected_per_ref_system: ClassVar = [("SU_LowerShanks", ["turn_parameters"])]
    _test_level_names: ClassVar = GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"]

    def __init__(
        self,
        base_path: Union[Path, str],
        *,
        default_participant_height_m: Optional[float] = None,
        raw_data_sensor: Literal["SU"] = "SU",
        reference_system: Optional[Literal["SU_LowerShanks"]] = None,
        reference_para_level: Literal["wb", "lwb"] = "wb",
        sensor_positions: Sequence[str] = ("LowerBack",),
        single_sensor_position: str = "LowerBack",
        sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]] = ("acc", "gyr"),
        missing_sensor_error_type: Literal["raise", "warn", "ignore", "skip"] = "skip",
        missing_reference_error_type: Literal["raise", "warn", "ignore", "skip"] = "ignore",
        memory: joblib.Memory = joblib.Memory(None),
        groupby_cols: Optional[Union[list[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ) -> None:
        self.base_path = base_path
        self.default_participant_height_m = default_participant_height_m
        super().__init__(
            raw_data_sensor=raw_data_sensor,
            reference_para_level=reference_para_level,
            sensor_positions=sensor_positions,
            single_sensor_position=single_sensor_position,
            sensor_types=sensor_types,
            missing_sensor_error_type=missing_sensor_error_type,
            missing_reference_error_type=missing_reference_error_type,
            memory=memory,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
        )
        # Note that needs to be below to have the correct types and avoid overwriting the reference_system by the base
        # class default
        self.reference_system = reference_system

    @property
    def _paths_list(self) -> list[Path]:
        all_participants = sorted(Path(self.base_path).rglob("data.mat"))
        return all_participants

    def _relpath_to_precomputed_test_list(self) -> str:
        return "test_list.json"

    @property
    def _metadata_level_names(self) -> Optional[tuple[str, ...]]:
        return "cohort", "participant_id"

    def _get_file_index_metadata(self, path: Path) -> tuple[str, ...]:
        p_id = path.parent.parent.name
        return _load_participant_metadata(Path(self.base_path)).loc[path.parent.parent.name].cohort, p_id

    @property
    def reference_parameters_relative_to_wb_(self) -> ReferenceData:
        with warnings.catch_warnings():
            # We know the stride parameters are in seconds
            warnings.filterwarnings("ignore", message="Assuming stride start and end values are provided in seconds")
            return super().reference_parameters_relative_to_wb_

    @property
    def reference_parameters_(self) -> ReferenceData:
        with warnings.catch_warnings():
            # We know the stride parameters are in seconds
            warnings.filterwarnings("ignore", message="Assuming stride start and end values are provided in seconds")
            return super().reference_parameters_

    @property
    def participant_metadata(self) -> ParticipantMetadata:
        self.assert_is_single(None, "participant_metadata")
        p_id = self.group_label.participant_id
        p_meta = _load_participant_metadata(Path(self.base_path)).loc[p_id]
        return ParticipantMetadata(
            cohort=p_meta["cohort"],
            height_m=self.default_participant_height_m or None,
            sensor_height_m=self.default_participant_height_m * 0.53 if self.default_participant_height_m else None,
        )

    @property
    def recording_metadata(self) -> RecordingMetadata:
        return {"measurement_condition": "laboratory"}
