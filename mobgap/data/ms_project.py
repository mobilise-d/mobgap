from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Literal, Optional, Union

import joblib
import pandas as pd

from mobgap.data._mobilised_matlab_loader import BaseGenericMobilisedDataset, GenericMobilisedDataset


class MsProjectDataset(BaseGenericMobilisedDataset):
    _not_expected_per_ref_system: ClassVar = [("SU_LowerShanks", ["turn_parameters"])]
    _test_level_names: ClassVar = GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"]

    def __init__(
        self,
        base_path: Union[Path, str],
        *,
        raw_data_sensor: Literal["SU", "INDIP", "INDIP2"] = "SU",
        reference_system: Optional[Literal["INDIP", "Stereophoto"]] = None,
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
        super().__init__(
            raw_data_sensor=raw_data_sensor,
            reference_system=reference_system,
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

    @property
    def _paths_list(self) -> list[Path]:
        all_participants = sorted(Path(self.base_path).rglob("data.mat"))
        return all_participants

    @property
    def _metadata_level_names(self) -> Optional[tuple[str, ...]]:
        return "participant_id", "condition"

    def _get_file_index_metadata(self, path: Path) -> tuple[str, ...]:
        return path.parents[1].name, path.parents[0].name