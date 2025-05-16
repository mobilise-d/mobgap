from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Final, Literal, Optional, Union

import joblib
import pandas as pd

from mobgap._docutils import make_filldoc
from mobgap.data import BaseGenericMobilisedDataset, GenericMobilisedDataset, matlab_dataset_docfiller

# TODO:
#  - [ ] Check which tests should actually be used in the Clinical and the Free Living dataset for validation and only
#        include those by default.

tvs_dataset_filler = make_filldoc(
    {
        **matlab_dataset_docfiller._dict,
        "participant_information": """
    participant_information
        All demographic and clinical information of the participants.
        Note, that this information is loaded from the ``participant_information.xlsx`` file in the data folder and
        might be slightly different from the information available via the ``participant_metadata`` attribute.
        The information there is loaded from the ``infoForAlgo.mat`` files and only contains the minimal set of
        information relevant for the algorithms.
    demographic_information
        Subset of the participant information containing only the demographic information.
    general_clinical_information
        Subset of the participant information containing only the general clinical information.
    cohort_clinical_information
        Subset of the participant information containing only the cohort specific clinical information.
    walking_aid_use_information
        Subset of the participant information containing only the walking aid use information.
    unique_center_id
        An unique identifier that indicates in which clinical center the data was recorded.
        Note, that these are "obfuscated" center ids. I.e. there are just numbers from 1 to 5.
    data_quality
        The data quality of the SU and the reference data per recording.
        This is a simple quality score (0-3) + additional comments that is provided for each recording.
        The numbers can be interpreted as follows:

        - 0: Recording discarded completely (these recordings are likely not included in the dataset in the first place)
        - 1: Recording has issues, but included in the dataset. Individual tests or trials might be missing, or might
          have degraded quality.
        - 2: Recording had some issues, but they could be fixed. Actual data should be good (INDIP only)
        - 3: Recording is good
    """,
        "dataset_warning": """
    .. note:: The data to be used with this dataset class is available on
       `https://zenodo.org/records/13899386 <Zenodo>`_.
       Download all files and extract them to a folder on your local machine.
       Make sure you don't change the folder structure (each zip file should be extracted to a separate folder).
       Then use the path to the folder containing the data as the ``base_path`` argument.
    """,
    }
)


@lru_cache(maxsize=1)
def _load_participant_information(path: Path) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    clinical_info = pd.read_excel(
        path,
        sheet_name="Participant Characteristics",
        engine="openpyxl",
        header=[0, 1],
        na_values=["N/A", "N.A.", "N.A"],
    )
    # The file contains summary rows at the end that we have to skip.
    # However, depending on how the file was saved, they are either 1 or 2 rows (see #197).
    # Instead of skipping by number we skip all rows that don't contain a participant id.
    # This should work for all cases.
    cols = clinical_info.columns.to_list()
    clinical_info = clinical_info[clinical_info[cols[0]].notna()]
    # We delay the setting of the index, as we need to set the correct dtypes first.
    # For some reason that is not possible in the loading step.
    clinical_info = (
        clinical_info.astype({cols[0]: int})
        .astype({cols[1]: str})
        .set_index(cols[:2])
        .rename_axis(["participant_id", "cohort"])
        .swaplevel()
        .sort_index()
    )

    # Set better dtypes

    clinical_info = clinical_info.rename(
        columns=lambda c: c.lower()
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "p")
        # Note we rename the height columns here, but the actually conversion to m is done later
    ).rename(
        columns={
            "updrsiii": "updrs3",
            "6mwt_distancewalked": "6mwt_distance_walked",
            "h&y": "h_and_y",
            "dominant_hand_l_or_r___wrist_sensor_worn_on_non_dominant_hand": "handedness",
            "height_cm": "height_m",
            "sensor_height_cm": "sensor_height_m",
            "lab_based_assessment": "walking_aid_used_laboratory",
            "real_world_outdoors": "walking_aid_used_free_living",
            "real_world_indoors": "walking_aid_used_free_living_indoors",
        }
    )
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
            height_m=lambda df_: df_.height_m / 100,
            sensor_height_m=lambda df_: df_.sensor_height_m / 100,
        )[original_col_order]
        .assign()
        .reset_index()
        .infer_objects()
        .astype(
            {
                "age": "Int8",
                "sex": pd.CategoricalDtype(["female", "male"]),
                "handedness": pd.CategoricalDtype(["left", "right"]),
                "fall_history": "boolean",
                "cohort": "string",
                "participant_id": "string",
            }
        )
        .set_index(["cohort", "participant_id"])
    )

    data_quality = (
        pd.read_excel(path, sheet_name="Data Quality Summary", engine="openpyxl", header=[0, 1], index_col=[0, 1])
        # We are only interested in the first 7 columns
        .iloc[:, :7]
        .rename_axis(["participant_id", "cohort"])
        .swaplevel()
    )
    index_with_dtypes = pd.MultiIndex.from_frame(
        data_quality.index.to_frame().astype({"participant_id": "string", "cohort": "string"})
    )
    data_quality = (
        data_quality.set_index(index_with_dtypes)
        .rename(
            columns={
                "Lower back sensor": "SU",
                "Lower back sensor2": "SU",
                "INDIP2": "INDIP",
                "Stereophotogrammetric": "Sterophoto",
                "Comments2": "Comments",
            },
            level=1,
        )
        .rename(
            columns={"Lab-based Assessment": "Laboratory", "Unstructured 2.5 hour Assessment": "Free-living"}, level=0
        )
        .sort_index()
    )

    return clinical_info, clinical_info_categories, data_quality


class BaseTVSDataset(BaseGenericMobilisedDataset):
    """Base class for the TVS datasets."""

    _not_expected_per_ref_system: ClassVar = [("INDIP", ["turn_parameters"]), ("Stereophoto", ["turn_parameters"])]

    _MEASUREMENT_CONDITION: str

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

    def _relpath_to_precomputed_test_list(self) -> str:
        return "test_list.json"

    def _get_measurement_condition(self) -> str:
        return self._MEASUREMENT_CONDITION.lower().replace("-", "_")

    @property
    def _paths_list(self) -> list[Path]:
        files = sorted(Path(self.base_path).glob(f"**/{self._MEASUREMENT_CONDITION}/data.mat"))
        if not files:
            raise ValueError(
                f"No files found in {self.base_path.resolve()}. Are you sure you provided the correct path to the data?"
            )
        return files

    @property
    def _test_level_names(self) -> tuple[str, ...]:
        if self._MEASUREMENT_CONDITION == "Laboratory":
            return GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"]
        return GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_2.5h"]

    @property
    def _metadata_level_names(self) -> Optional[tuple[str, ...]]:
        return "cohort", "participant_id"

    def _get_file_index_metadata(self, path: Path) -> tuple[str, ...]:
        return path.parents[2].name, path.parents[1].name

    @property
    def unique_center_id(self) -> pd.Series:
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
        selected_values = info.reindex(
            pd.MultiIndex.from_frame(self.index[["cohort", "participant_id"]].drop_duplicates())
        )
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

    @property
    def data_quality(self) -> pd.DataFrame:
        info = _load_participant_information(Path(self.base_path) / "participant_information.xlsx")[2]
        selected_values = info.reindex(
            pd.MultiIndex.from_frame(self.index[["cohort", "participant_id"]].drop_duplicates())
        )
        return selected_values[self._MEASUREMENT_CONDITION]


@tvs_dataset_filler
class TVSLabDataset(BaseTVSDataset):
    """A dataset containing all Lab Data recorded within the Mobilise-D technical validation study.

    %(dataset_warning)s

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
    %(participant_information)s

    """

    _MEASUREMENT_CONDITION = "Laboratory"

    TEST_NAMES_SHORT: Final = {
        "Test4": "TimedUpGo",
        "Test5": "WalkComfortable",
        "Test6": "WalkSlow",
        "Test7": "WalkFast",
        "Test8": "L",
        "Test9": "Surface",
        "Test10": "Halway",
        "Test11": "DailyActivities",
    }

    TEST_NAMES_PRETTY: Final = {
        "Test4": "TUG",
        "Test5": "Straight Walk Comfortable",
        "Test6": "Straight Walk Slow",
        "Test7": "Straight Walk Fast",
        "Test8": "L-Test",
        "Test9": "Surface Test",
        "Test10": "Hallway Test",
        "Test11": "Simulated Daily Activities",
    }

    def create_index(self) -> pd.DataFrame:
        return (
            super()
            .create_index()
            .assign(
                test_name=lambda df: df.test.replace(self.TEST_NAMES_SHORT),
                test_name_pretty=lambda df: df.test.replace(self.TEST_NAMES_PRETTY),
            )
        )


@tvs_dataset_filler
class TVSFreeLivingDataset(BaseTVSDataset):
    """A dataset containing all Free-Living (2.5 hour) Data recorded within the Mobilise-D technical validation study.

    %(dataset_warning)s

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
    %(participant_information)s

    """

    _MEASUREMENT_CONDITION = "Free-living"

    def create_index(self) -> pd.DataFrame:
        return (
            super()
            .create_index()
            .assign(
                recording_name=lambda df: df.recording.replace({"Recording4": "2.5h"}),
                recording_name_pretty=lambda df: df.recording.replace({"Recording4": "Free Living 2.5h"}),
            )
        )
