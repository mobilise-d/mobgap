from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Union

import joblib
import pandas as pd

from mobgap.data import BaseGenericMobilisedDataset, GenericMobilisedDataset, matlab_dataset_docfiller

# TODO:
#  - [ ] Add version for Free Living
#  - [ ] Think about how to represent "data quality" in the dataset
#  - [ ] Check which tests should actually be used in the Clinical and the Free Living dataset for validation and only
#        include those by default.
# - [ ] Center=id will not work after changing the patient codes


@lru_cache(maxsize=1)
def _load_participant_information(path: Path) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    clinical_info = (
        pd.read_excel(
            path,
            sheet_name="Participant Characteristics",
            engine="openpyxl",
            header=[0, 1],
            na_values=["N/A", "N.A.", "N.A"],
        ).iloc[:-2]  # We need to remove the last two rows, as the Excel file contains "summary" rows at the end
    )
    cols = clinical_info.columns.to_list()
    # We delay the setting of the index, as we need to set the correct dtypes first.
    # For some reason that is not possible in the loading step.
    clinical_info = (
        clinical_info.astype({cols[0]: int})
        .astype({cols[0]: str})
        .set_index(cols[:2])
        .rename_axis(["participant_id", "cohort"])
        .swaplevel()
        .sort_index()
    )
    data_quality = pd.read_excel(
        path, sheet_name="Data Quality Summary", engine="openpyxl", header=[0, 1], index_col=[0, 1]
    )

    # Set better dtypes

    clinical_info = clinical_info.rename(
        columns=lambda c: c.lower().replace("(", "").replace(")", "").replace(" ", "_").replace("-", "_")
    ).rename(columns={"updrsiii": "updrs3", "6mwt_distancewalked": "6mwt_distance_walked", "h&y": "h_and_y"})
    clinical_info_cols = clinical_info.columns.to_list()
    clinical_info_categories = {}
    for c in clinical_info_cols:
        clinical_info_categories.setdefault(c[0], []).append(c[1])

    # Drop the first level of the columns
    clinical_info = clinical_info.droplevel(0, axis=1)
    original_col_order = clinical_info.columns.to_list()
    clinical_info = (
        clinical_info.assign(
            handedness=lambda df_: df_.handedness.str.strip(" '").replace({"R": "right", "L": "left"}),
            sex=lambda df_: df_.sex.str.lower(),
            fall_history=lambda df_: df_.fall_history.str.lower().replace({"yes": True, "no": False}),
        )[original_col_order]
        .assign()
        .astype(
            {
                "age": "Int8",
                "bmi": "float32",
                "sex": pd.CategoricalDtype(["female", "male"]),
                "handedness": pd.CategoricalDtype(["left", "right"]),
                "fall_history": "boolean",
            }
        )
    )

    return clinical_info, clinical_info_categories, data_quality


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

    @property
    def unique_center_id(self) -> str:
        return (
            self.index[["participant_id", "cohort"]]
            .drop_duplicates()
            .assign(center_id=lambda df_: df_.participant_id.str[0])
            .set_index(["participant_id", "cohort"])
            .center_id
        )

    @property
    def participant_information(self) -> pd.DataFrame:
        info = _load_participant_information(Path(self.base_path) / "participant_information.xlsx")[0]
        selected_values = info.loc[info.index.get_level_values("participant_id").isin(self.index["participant_id"])]
        return selected_values

    def _get_info_categories(self) -> dict[str, list[str]]:
        return _load_participant_information(Path(self.base_path) / "participant_information.xlsx")[1]

    @property
    def demographic_information(self) -> pd.DataFrame:
        return self.participant_information[self._get_info_categories()["demographics"]]

    @property
    def general_clinical_information(self) -> pd.DataFrame:
        return self.participant_information[self._get_info_categories()["general_clinical_characteristics"]]

    @property
    def cohort_clinical_information(self) -> pd.DataFrame:
        return self.participant_information[self._get_info_categories()["cohort_specific_clinical_characteristics"]]

    @property
    def walking_aid_use_information(self) -> pd.DataFrame:
        return self.participant_information[self._get_info_categories()["walking_aid_use"]]


ds = TVSLabDataset(
    "/home/arne/Downloads/TVS_DATA_ALL/", missing_reference_error_type="ignore", reference_system="INDIP"
)
# ds._create_precomputed_test_list()


print(ds.unique_center_id)
