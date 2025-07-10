import warnings
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Literal, Optional, Union

import pandas as pd
from joblib import Memory, Parallel, delayed
from tpcp import Dataset
from tpcp.caching import hybrid_cache
from tqdm.auto import tqdm

from mobgap.data._mobilsed_weartime_loader import load_weartime_from_daily_mcroberts_report

SITE_CODES: type = Literal[
    "CAU",
    "CHUM",
    "ICL",
    "KUL1",
    "KUL2",
    "NTNU",
    "PFLG",
    "RBMF",
    "TASMC",
    "TFG",
    "UKER",
    "UNEW",
    "UNN",
    "USFD",
    "USR",
    "UZH1",
    "UZH2",
    "UZH3",
    "ISG1",
    "ISG2",
    "ISG3",
    "ISG4",
]

TIME_ZONES: dict[SITE_CODES, str] = {
    "CAU": "Europe/Berlin",
    "CHUM": "Europe/Paris",
    "ICL": "Europe/London",
    "KUL1": "Europe/Brussels",
    "KUL2": "Europe/Brussels",
    "NTNU": "Europe/Oslo",
    "PFLG": "Europe/Berlin",
    "RBMF": "Europe/Berlin",
    "TASMC": "Asia/Tel_Aviv",
    "TFG": "Europe/Athens",
    "UKER": "Europe/Berlin",
    "UNEW": "Europe/London",
    "UNN": "Europe/London",
    "USFD": "Europe/London",
    "USR": "Europe/Rome",
    "UZH1": "Europe/Zurich",
    "UZH2": "Europe/Zurich",
    "UZH3": "Europe/Zurich",
    "ISG1": "Europe/Madrid",
    "ISG2": "Europe/Madrid",
    "ISG3": "Europe/Madrid",
    "ISG4": "Europe/Madrid",
}


def _create_index(
    dmo_path: Path, site_pid_map_path: Path, timezones: dict[SITE_CODES, str], memory: Memory, visit_type: str
) -> pd.DataFrame:
    cache = hybrid_cache(memory, 1)
    site_data = cache(_load_site_pid_map)(site_pid_map_path=site_pid_map_path, timezones=timezones)
    dmo_data, _ = cache(_load_dmo_data)(dmo_path=dmo_path, timezone_per_subject=site_data, visit_type=visit_type)

    return (
        dmo_data.index.to_frame()[["visit_type", "participant_id", "measurement_date"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .astype({"participant_id": "string", "measurement_date": "string", "visit_type": "string"})[
            ["visit_type", "participant_id", "measurement_date"]
        ]
    )


def _load_site_pid_map(site_pid_map_path: Path, timezones: dict[SITE_CODES, str]) -> pd.DataFrame:
    timezones_df = pd.DataFrame.from_dict(timezones, orient="index", columns=["timezone"])

    site_data = (
        pd.read_csv(site_pid_map_path)[["Local.Participant", "Participant.Site"]]
        .rename(columns={"Local.Participant": "participant_id", "Participant.Site": "site"})
        .join(timezones_df, on="site")
        .astype({"participant_id": "string", "site": "string", "timezone": "string"})
        .set_index("participant_id")
    )
    return site_data


def _map_visit_names(names: pd.Series) -> pd.Series:
    # We cut all the keys to the same length to make the comparison easier.

    cut_names = names.str[:14]
    start_with = {
        "Baseline Visit": "T1",
        "Follow-up (T2)": "T2",
        "Follow-up (T3)": "T3",
        "Follow-up (T4)": "T4",
        "Follow-up (T5)": "T5",
        "Follow-up (T6)": "T6",
        "Unscheduled Vi": "UV",
    }
    return cut_names.map(start_with, None)


@lru_cache(maxsize=1)
def _load_pid_mid_map(compliance_report: Path) -> pd.DataFrame:
    return (
        pd.read_excel(compliance_report, sheet_name="compliance")[["participants", "results_id", "visit"]]
        .drop_duplicates()
        .rename(columns={"participants": "participant_id", "visit": "visit_type", "results_id": "measurement_id"})
        .assign(visit_type=lambda df_: _map_visit_names(df_["visit_type"]))
        .astype({"participant_id": "string", "visit_type": "string", "measurement_id": "int64"})
        .set_index(["participant_id", "visit_type"])
    )


def _load_dmo_data(
    dmo_path: Path, timezone_per_subject: pd.DataFrame, visit_type: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    warnings.warn(
        "Initial data loading. This might take a while! But, don't worry, we cache the loaded results.\n\n"
        "If you are seeing this message multiple times, you might want to consider using a joblib memory by "
        "passing ``memory=Memory('some/cache/path)`` to the dataset constructor to cache the index creation "
        "between script executions.",
        stacklevel=1,
    )

    dmos = [
        "duration",
        "initialcontact_event_number",
        "turn_number_so",
        "averagecadence",
        "averagestridespeed",
        "averagestridelength",
        "averagestrideduration",
    ]
    dmos_flag = [f"{dmo}_flag" for dmo in dmos]

    dmo_rename_dict = {
        "duration": "duration_s",
        "initialcontact_event_number": "n_raw_initial_contacts",
        "turn_number_so": "n_turns",
        "averagecadence": "cadence_spm",
        "averagestridespeed": "walking_speed_mps",
        "averagestridelength": "stride_length_m",
        "averagestrideduration": "stride_duration_s",
    }

    dmo_data = (
        pd.read_csv(dmo_path)[
            [
                "participantid",
                "wb_id",
                "visit_date",
                "wbday",
                *dmos,
                *dmos_flag,
                "visit_number",
            ]
        ]
        .rename(columns={"participantid": "participant_id", "visit_number": "visit_type"})
        .astype(
            {
                "wb_id": "string",
                "participant_id": "string",
            }
        )
        .assign(visit_date_utc=lambda df_: pd.to_datetime(df_["visit_date"], unit="s", utc=True))
        .drop("visit_date", axis=1)
        .merge(timezone_per_subject, left_on="participant_id", right_index=True, how="left")
        .assign(
            measurement_date=lambda df_: df_.groupby("timezone")["visit_date_utc"].transform(
                lambda x: x.dt.tz_convert(x.name).dt.strftime("%Y-%m-%d")
            )
        )
        .astype(
            {
                "measurement_date": "string",
                "site": "string",
                "timezone": "string",
            }
        )
        .set_index(["visit_type", "participant_id", "measurement_date", "wb_id"])
        .sort_index()
    )

    if dmo_data["timezone"].isna().any():
        raise ValueError(
            "For one or more visits either no valid site or no valid timezone for the respective site could be "
            "identified."
        )

    if len(visit_types := dmo_data.index.get_level_values("visit_type").unique().to_list()) != 1:
        warnings.warn(
            "The visit_type is not unique. "
            f"This should not happen! The data contains the following types: {visit_types}. "
            f"It should only contain data from {visit_type}. "
            "We will drop all data that does not match the expected visit_type. "
            "Please check your data and make sure that you are using the correct data file.",
            stacklevel=2,
        )
        dmo_data = dmo_data.loc[[visit_type]]

    dmo_flag_data = dmo_data[dmos_flag].rename(columns=lambda c: c.replace("_flag", "")).rename(columns=dmo_rename_dict)
    dmo_data = dmo_data.drop(dmos_flag, axis=1).rename(columns=dmo_rename_dict)

    return dmo_data, dmo_flag_data


class MobilisedCvsDmoDataset(Dataset):
    """A dataset representing **calculated** DMO data of the clinical validation study.

    .. warning::
        This dataset will not provide the raw data of the clinical validation study, but rather the official export of
        the calculated DMO data as provided by WP3.

    .. warning:: We assume that the dmo data file has the structure `WP6-{visit_type}-...`

    Parameters
    ----------
    dmo_path
        The path to the calculated DMO data export.
        This should be the path to the approx. 1 Gb csv file with all the DMOs.
        Note, that we only support DMO files including the data of a single visit (e.g. T1, T2, ...) at a time.
    site_pid_map_path
        The path to the file that contains the mapping between all the participants and their site.
        This is required to calculate the correct timezone for each participant.
    weartime_reports_base_path
        The base path to the folder that contains the wear-time compliance reports.
        These exports are provided by McRoberts and contain the wear-time per minute data.
    pre_compute_daily_weartime
        If True, the daily weartime will be pre-computed and a new file will be stored within the
        ``weartime_reports_base_path``.
        This computation will be performed for all participants when you first attempt to load any weartime related
        features.
        This might take multiple minutes, but will speed up the loading of the data in the future.
        If you are only planning to access a small subset of the data, you might want to set this to False.

        .. warning:: Remember to delete the pre-computed weartime file, if you obtain a new version of the weartime
           reports.


    """

    dmo_path: Union[str, Path]
    site_pid_map_path: Union[str, Path]
    weartime_reports_base_path: Union[str, Path]
    pre_compute_daily_weartime: bool
    site_pid_map_path: Union[str, Path]
    memory: Memory

    TIME_ZONES: ClassVar[dict[SITE_CODES, str]] = TIME_ZONES
    WEARTIME_REPORT_CACHE_FILE_NAME: ClassVar[str] = "daily_weartime_pre_computed.csv"

    def __init__(
        self,
        dmo_path: Union[str, Path],
        site_pid_map_path: Union[str, Path],
        *,
        weartime_reports_base_path: Optional[Union[str, Path]] = None,
        pre_compute_daily_weartime: bool = True,
        memory: Memory = Memory(None),
        groupby_cols: Optional[Union[list[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ) -> None:
        self.dmo_path = dmo_path
        self.site_pid_map_path = site_pid_map_path
        self.weartime_reports_base_path = weartime_reports_base_path
        self.pre_compute_daily_weartime = pre_compute_daily_weartime
        self.memory = memory
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def visit_type(self) -> str:
        """The visit type (T1 - TN) of the dataset.

        Each dataset instance can only load data from a single visit type.
        This is determined by the visit type in the filename of the dmo export file.

        """
        return str(Path(self.dmo_path).name).split("-")[1].upper()

    def _get_participant_site_metadata(self) -> pd.DataFrame:
        return hybrid_cache(self.memory, 1)(_load_site_pid_map)(
            site_pid_map_path=Path(self.site_pid_map_path),
            timezones=self.TIME_ZONES,
        )

    @property
    def _compliance_file_path(self) -> Path:
        try:
            return next(Path(self.weartime_reports_base_path).glob("CVS-wear-compliance-*.xlsx"))
        except StopIteration as e:
            raise FileNotFoundError(
                "Could not find the wear-time compliance report. "
                "Please make sure that the file is named `CVS-wear-complicance-*.xlsx` and located in the following "
                f"folder {self.weartime_reports_base_path}."
            ) from e

    def _get_pid_mid_map(self) -> pd.DataFrame:
        return _load_pid_mid_map(self._compliance_file_path)

    @property
    def weartime_daily(self) -> pd.DataFrame:
        """The daily weartime per participant.

        This is calculated from the minute to minute weartime reports provided by McRoberts.
        This is optional, and you might not have access to the weartime reports.
        """
        if self.weartime_reports_base_path is None:
            raise ValueError(
                "The `weartime_reports_base_path` must be provided to load any weartime related features. "
            )
        if self.pre_compute_daily_weartime is True:
            wear_time = self._get_precomputed_daily_weartime()
        else:
            # In this case we manually compute the daily weartime only for the participants that are still in the index
            # and not for all participants.
            try:
                relevant_mid = (
                    self._get_pid_mid_map()
                    .xs(self.visit_type, level="visit_type")
                    .loc[self.index["participant_id"].unique()]
                )
            except KeyError as e:
                raise KeyError(
                    "It looks like you are trying to access the weartime for a participant that does not have any "
                    "weartime data. "
                    f"Check the compliance report {self._compliance_file_path} to see which participants have data "
                    "available."
                ) from e
            files = []
            non_available = []
            for mid in relevant_mid.itertuples():
                try:
                    files.append(next(Path(self.weartime_reports_base_path).glob(f"*_{mid.measurement_id}_minute.csv")))
                except StopIteration:
                    non_available.append(mid.measurement_id)
                warnings.warn(
                    f"Could not find the wear-time data for the following measurement ids {non_available}. "
                    f"This corresponds to ({len(non_available)} out of {len(relevant_mid)}) participants.",
                    stacklevel=1,
                )
            wear_time = self._calculate_daily_weartime(files)

        # Ensure correct dtypes:
        wear_time = wear_time.astype(
            {
                "visit_type": "string",
                "participant_id": "string",
                "measurement_date": "string",
                "measurement_id": "string",
            }
        )

        # We merge the weartime report with the dataset index to get NaNs for all participants that don't have any
        # weartime data, but are still in the dataset.
        wear_time = self.index.merge(
            wear_time, on=["visit_type", "participant_id", "measurement_date"], how="left"
        ).drop(["measurement_id"], axis=1)

        wear_time = wear_time.set_index(["visit_type", "participant_id", "measurement_date"]).round(3)

        if (wim := wear_time.index.to_frame().duplicated(keep="last")).any():
            warnings.warn(
                "The weartime report contains duplicate indices. "
                "This indicates that multiple weartime reports exist for the same participant. "
                "This should not happen! "
                "We will drop all duplicate indices for now (keeping the last one). "
                "You should investigate this further!",
                stacklevel=2,
            )
            wear_time = wear_time[~wim]

        return wear_time

    def _get_precomputed_daily_weartime(self) -> pd.DataFrame:
        try:
            return pd.read_csv(Path(self.weartime_reports_base_path) / self.WEARTIME_REPORT_CACHE_FILE_NAME)
        except FileNotFoundError:
            pass

        warnings.warn(
            "Could not find the pre-computed daily weartime file. Computing it now. This might take a while!",
            stacklevel=2,
        )

        files = list(Path(self.weartime_reports_base_path).glob("*_minute.csv"))
        results = self._calculate_daily_weartime(files)

        file_path = Path(self.weartime_reports_base_path) / self.WEARTIME_REPORT_CACHE_FILE_NAME
        warnings.warn(
            f"Finished computing the daily weartime. Now saving the results to disk at:{file_path}.", stacklevel=2
        )
        results.to_csv(file_path, index=False)
        return results

    def _calculate_daily_weartime(self, filelist: list[Path]) -> pd.DataFrame:
        if len(filelist) == 0:
            raise FileNotFoundError(
                "Could not find any wear-time reports. "
                "Please make sure that the files are named `*_minute.csv` and located in the following "
                f"folder: {self.weartime_reports_base_path}."
            )

        # Finally we will merge the results with the pid_mid_map to get the correct participant id.
        # We already query that up here to fail early in case the file does not exist.
        pid_mid_map = self._get_pid_mid_map().reset_index()

        def process_single_file(path: Path) -> tuple[str, pd.DataFrame]:
            p_id = path.name.split("_")[-2]
            result = load_weartime_from_daily_mcroberts_report(path)
            return p_id, result

        # Note: We use multiprocessing here, as this can speed up the computation significantly.
        #       However, we are still primarily bound by the IO speed of the hard drive, so to many processes will
        #       not help.
        results = list(
            tqdm(
                Parallel(n_jobs=3, return_as="generator")(delayed(process_single_file)(f) for f in filelist),
                total=len(filelist),
                desc="Loading daily weartime reports for participants",
            )
        )
        results = (
            pd.concat(dict(results), names=["measurement_id", "measurement_date"])
            .reset_index()
            .astype({"measurement_date": "string", "measurement_id": "int64"})
        )

        results = results.merge(pid_mid_map, on=["measurement_id"])
        return results

    def _get_dmo_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return hybrid_cache(self.memory, 1)(_load_dmo_data)(
            dmo_path=Path(self.dmo_path),
            timezone_per_subject=self._get_participant_site_metadata(),
            visit_type=self.visit_type,
        )

    def _extract_relevant_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # TODO: replace this with the official tpcp check once released
        #       (https://github.com/mad-lab-fau/tpcp/issues/104)
        is_full_dataset = self.index_is_unchanged
        # Short circuit, if we have the full dataset, as this is a typical usecase and likely much faster.
        if not is_full_dataset:
            # That was the fastest version I found so far, but this is still slow as hell, if you have a large number of
            # participants still in the dataset....
            query_index = pd.MultiIndex.from_frame(self.index)
            data = data.reset_index("wb_id").loc[query_index]
        return data.reset_index().set_index(["visit_type", "participant_id", "measurement_date", "wb_id"])

    @property
    def data(self) -> pd.DataFrame:
        """The DMO data per WB.

        This will provide a df with all DMOs, where each row corresponds to a single WB.
        The df will include the data of all participants and days currently selected in the index of the dataset.
        """
        dmo_data, _ = self._get_dmo_data()
        return self._extract_relevant_data(dmo_data)

    @property
    def data_mask(self) -> pd.DataFrame:
        """The DMO data mask per WB.

        A "true"/"false" flag for each individual DMO.
        A "false" indicates that the specific DMO value might potentially be incorrect.
        These flags are determined using some expert defined thresholds for likely valid ranges of DMOs.

        The shaoe of the flags-df is identical to the shape of the DMO data-df, so that they can be directly overlayed.
        """
        _, dmo_flag_data = self._get_dmo_data()
        return self._extract_relevant_data(dmo_flag_data)

    @property
    def measurement_site(self) -> SITE_CODES:
        """The measurement site of the dataset.

        This can only be accessed if the dataset only contains data from a single participant.
        """
        self.assert_is_single(["visit_type", "participant_id"], "measurement_site")
        p_id = self.group_labels[0].participant_id

        return self._get_participant_site_metadata().loc[p_id, "site"]

    @property
    def timezone(self) -> str:
        """The timezone of the measurement site.

        This can only be accessed if the dataset only contains data from a single participant.
        """
        self.assert_is_single(["visit_type", "participant_id"], "timezone")
        p_id = self.group_labels[0].participant_id

        return self._get_participant_site_metadata().loc[p_id, "timezone"]

    def create_index(self) -> pd.DataFrame:
        return hybrid_cache(self.memory, 1)(_create_index)(
            Path(self.dmo_path),
            Path(self.site_pid_map_path),
            self.TIME_ZONES,
            memory=self.memory,
            visit_type=self.visit_type,
        )
