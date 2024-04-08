from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional, Union

import joblib
import pandas as pd

from mobgap.data import BaseGenericMobilisedDataset, GenericMobilisedDataset, matlab_dataset_docfiller


# TODO:
#  - [ ] Add version for Free Living
#  - [ ] Add loader for clinical data
#  - [ ] Think about how to represent "data quality" in the dataset
#  - [ ] Check which tests should actually be used in the Clinical and the Free Living dataset for validation and only
#        include those by default.

@matlab_dataset_docfiller
class TVSLabDataset(BaseGenericMobilisedDataset):
    """A dataset containing all Lab Data recorded within the Mobilise-D technical validation study.

    .. warning:: The dataset is not yet available. The data will be made available end of June 2024. Then you need to
        download the data from Zenodo and provide the path to the data folder.

    Parameters
    ----------
    base_path
        The path to the folder containing the data.
    %(file_loader_args)s
    %(dataset_memory_args)s
    %(general_dataset_args)s

    Attributes
    ----------
    %(dataset_data_attrs)s

    See Also
    --------
    %(dataset_see_also)s

    """

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

    _MEASUREMENT_CONDITION = "Laboratory"

    def _relpath_to_precomputed_test_list(self) -> str:
        return "test_list.json"

    @property
    def _paths_list(self) -> list[Path]:
        return sorted(Path(self.base_path).glob(f"**/{self._MEASUREMENT_CONDITION}/data.mat"))

    @property
    def _test_level_names(self) -> tuple[str, ...]:
        if self._MEASUREMENT_CONDITION == "Laboratory":
            return GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"]
        return GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_free"]

    @property
    def _metadata_level_names(self) -> Optional[tuple[str, ...]]:
        return "cohort", "participant_id"

    def _get_file_index_metadata(self, path: Path) -> tuple[str, ...]:
        return path.parents[2].name, path.parents[1].name


ds = TVSLabDataset(
    "/home/arne/Downloads/TVS_DATA_ALL/", missing_reference_error_type="ignore", reference_system="INDIP"
)

# ds._create_precomputed_test_list()

print(ds.index)
