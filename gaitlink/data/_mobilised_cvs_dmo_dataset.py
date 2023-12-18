import binascii
import warnings
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Generic, Literal, Optional, TypeAlias, TypeVar, Union

import pandas as pd
from joblib import Memory, Parallel, delayed
from tpcp import Dataset
from tpcp._hash import custom_hash
from tqdm.auto import tqdm

from gaitlink.data._mobilsed_weartime_loader import load_weartime_from_daily_mcroberts_report
from gaitlink.data._utils import staggered_cache

SITE_CODES: TypeAlias = Literal[
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


# TODO: Replace with tpcp version once released

T = TypeVar("T")


class UniversalHashableWrapper(Generic[T]):
    def __init__(self, obj: T) -> None:
        self.obj = obj

    def __hash__(self):
        """Hash the object using the pickle based approach."""
        return int(binascii.hexlify(custom_hash(self.obj).encode("utf-8")), 16)

    def __eq__(self, other):
        """Compare the object using their hash."""
        return custom_hash(self.obj) == custom_hash(other.obj)


def _create_index(
    dmo_path: Path, site_pid_map_path: Path, timezones: UniversalHashableWrapper[dict[SITE_CODES, str]], memory: Memory
) -> pd.DataFrame:
    site_data = staggered_cache(_load_site_pid_map, memory, 1)(site_pid_map_path=site_pid_map_path, timezones=timezones)
    dmo_data = staggered_cache(_load_dmo_data, memory, 1)(
        dmo_path=dmo_path, timezone_per_subject=UniversalHashableWrapper(site_data)
    )

    visit_type = dmo_path.name.split("-")[1].upper()

    return (
        dmo_data.index.to_frame()[["participant_id", "measurement_date"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .assign(visit_type=visit_type)
        .astype({"participant_id": "string", "measurement_date": "string"})[
            ["visit_type", "participant_id", "measurement_date"]
        ]
    )


def _load_site_pid_map(
    site_pid_map_path: Path, timezones: UniversalHashableWrapper[dict[SITE_CODES, str]]
) -> pd.DataFrame:
    timezones_df = pd.DataFrame.from_dict(timezones.obj, orient="index", columns=["timezone"])

    site_data = (
        pd.read_csv(site_pid_map_path)[["Local.Participant", "Participant.Site"]]
        .rename(columns={"Local.Participant": "participant_id", "Participant.Site": "site"})
        .join(timezones_df, on="site")
        .set_index("participant_id")
    )
    return site_data


@lru_cache(maxsize=1)
def _load_pid_mid_map(compliance_report: Path) -> pd.DataFrame:
    return (
        pd.read_excel(compliance_report, sheet_name="compliance")[["participants", "results_id", "visit"]]
        .drop_duplicates()
        .rename(columns={"participants": "participant_id", "visit": "visit_type", "results_id": "measurement_id"})
        .assign(
            visit_type=lambda df_: df_["visit_type"].replace(
                {
                    "Baseline Visit – daily mobility": "T1",
                    "Follow-up (T2) – daily mobility": "T2",
                    "Follow-up (T3) – daily mobility": "T3",
                    "Follow-up (T4) – daily mobility": "T4",
                    "Follow-up (T5) – daily mobility": "T5",
                    "Follow-up (T6) – daily mobility": "T6",
                    "Unscheduled Visit (--) – daily mobility": "UV",
                }
            )
        )
        .astype({"participant_id": "string", "visit_type": "string", "measurement_id": int})
        .set_index(["participant_id", "visit_type"])
    )


def _load_dmo_data(dmo_path: Path, timezone_per_subject: UniversalHashableWrapper[pd.DataFrame]) -> pd.DataFrame:
    warnings.warn(
        "Initial data loading. This might take a while! But, don't worry, we cache the loaded results.\n\n"
        "If you are seeing this message multiple times, you might want to consider using a joblib memory by "
        "passing ``memory=Memory('some/cache/path)`` to the dataset constructor to cache the index creation"
        "between script executions.",
        stacklevel=3,
    )
    # TODO: Extract flag-data separately
    timezone_per_subject = timezone_per_subject.obj

    dmo_data = (
        pd.read_csv(dmo_path)[
            [
                "participantid",
                "wb_id",
                "visit_date",
                "duration",
                "initialcontact_event_number",
                "averagecadence",
                "averagestridespeed",
                "averagestridelength",
                "averagestrideduration",
                "turn_number_so",
                "wbday",
            ]
        ]
        .rename(columns={"participantid": "participant_id"})
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
                "wb_id": "string",
                "site": "string",
                "timezone": "string",
                "participant_id": "string",
            }
        )
        .set_index(["participant_id", "measurement_date", "wb_id"])
        .sort_index()
    )
    return dmo_data


class MobilisedCvsDmoDataset(Dataset):
    """A dataset representing **calculated** DMO data of the clinical validation study.

    .. warning::
        This dataset will not provide the raw data of the clinical validation study, but rather the official export of
        the calculated DMO data as provided by WP3.

    .. warning:: We assume that the dmo data file has the structure `WP6-{visit_type}-...`

    Parameters
    ----------
    dmo_export_path
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

    dmo_export_path: Union[str, Path]
    site_pid_map_path: Union[str, Path]
    weartime_reports_base_path: Union[str, Path]
    pre_compute_daily_weartime: bool
    site_pid_map_path: Union[str, Path]
    memory: Memory

    TIME_ZONES: ClassVar[dict[SITE_CODES, str]] = TIME_ZONES
    WEARTIME_REPORT_CACHE_FILE_NAME: ClassVar[str] = "daily_weartime.csv"

    def __init__(
        self,
        dmo_export_path: Union[str, Path],
        site_pid_map_path: Union[str, Path],
        *,
        weartime_reports_base_path: Optional[Union[str, Path]] = None,
        pre_compute_daily_weartime: bool = True,
        memory: Memory = Memory(None),
        groupby_cols: Optional[Union[list[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ) -> None:
        self.dmo_export_path = dmo_export_path
        self.site_pid_map_path = site_pid_map_path
        self.weartime_reports_base_path = weartime_reports_base_path
        self.pre_compute_daily_weartime = pre_compute_daily_weartime
        self.memory = memory
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def visit_type(self):
        return self.dmo_export_path.split("-")[1]

    def _get_participant_site_metadata(self) -> pd.DataFrame:
        return staggered_cache(_load_site_pid_map, self.memory, 1)(
            site_pid_map_path=Path(self.site_pid_map_path),
            timezones=UniversalHashableWrapper(self.TIME_ZONES),
        )

    def _get_pid_mid_map(self) -> pd.DataFrame:
        if self.weartime_reports_base_path is None:
            raise ValueError(
                "The `weartime_reports_base_path` must be provided to load any weartime related features. "
            )

        try:
            pid_mid_map_path = next(Path(self.weartime_reports_base_path).glob("CVS-wear-compliance-*.xlsx"))
        except StopIteration as e:
            raise FileNotFoundError(
                "Could not find the wear-time compliance report. "
                "Please make sure that the file is named `CVS-wear-complicance-*.xlsx` and located in the following "
                f"folder {self.weartime_reports_base_path}."
            ) from e
        return _load_pid_mid_map(pid_mid_map_path)

    def _get_daily_weartime(self) -> pd.DataFrame:
        if self.pre_compute_daily_weartime is True:
            wear_time = self._get_precomputed_daily_weartime()
        else:
            # In this case we manually compute the daily weartime only for the participants that are still in the index
            # and not for all participants.
            try:
                relevant_mid = (
                    self._get_pid_mid_map()
                    .xs(self.visit_type.upper(), level="visit_type")
                    .loc[self.index["participant_id"].unique()]
                )
            except KeyError as e:
                compliance_file = next(Path(self.weartime_reports_base_path).glob("CVS-wear-compliance-*.xlsx"))
                raise KeyError(
                    "It looks like you are trying to access the weartime for a participant that does not have any "
                    "weartime data. "
                    f"Check the compliance report {compliance_file} to see which participants have data available. "
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
                    f"This corresponds to ({len(non_available)} out of {len(relevant_mid)}) participants. "
                )
            wear_time = self._calculate_daily_weartime(files)
        # We merge the weartime report with the dataset index to get NaNs for all participants that don't have any
        # weartime data, but are still in the dataset.
        return (
            self.index.merge(wear_time, on=["participant_id", "visit_date"], how="left")
            .set_index(["visit_type", "participant_id", "visit_date"])
            .drop(["measurement_id", "visit_type"], axis=1)
        )

    def _get_precomputed_daily_weartime(self) -> pd.DataFrame:
        if self.weartime_reports_base_path is None:
            raise ValueError(
                "The `weartime_reports_base_path` must be provided to load any weartime related features. "
            )
        try:
            return pd.read_csv(Path(self.weartime_reports_base_path) / self.WEARTIME_REPORT_CACHE_FILE_NAME)
        except FileNotFoundError:
            pass

        warnings.warn(
            "Could not find the pre-computed daily weartime file. Computing it now. " "This might take a while!"
        )

        files = list(Path(self.weartime_reports_base_path).glob("*_minute.csv"))
        results = self._calculate_daily_weartime(files)

        file_path = Path(self.weartime_reports_base_path) / self.WEARTIME_REPORT_CACHE_FILE_NAME
        warnings.warn("Finished computing the daily weartime. Now saving the results to disk at:" f"{file_path}.")
        results.to_csv(file_path)
        return results

    def _calculate_daily_weartime(self, filelist: list[Path]) -> pd.DataFrame:
        def process_single_file(path):
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
            .astype({"visit_date": "string", "measurement_id": int})
        )

        # Finally we will merge the results with the pid_mid_map to get the correct participant id.
        pid_mid_map = self._get_pid_mid_map().reset_index()
        results = results.merge(pid_mid_map, on=["measurement_id"]).set_index(["participant_id", "measurement_date"])
        return results

    def _get_dmo_data(self):
        return staggered_cache(_load_dmo_data, self.memory, 1)(
            dmo_path=Path(self.dmo_export_path),
            timezone_per_subject=UniversalHashableWrapper(self._get_participant_site_metadata()),
        )

    @property
    def data(self):
        dmo_data = self._get_dmo_data()
        # TODO: replace this with the official tpcp check once released
        #       (https://github.com/mad-lab-fau/tpcp/issues/104)
        is_full_dataset = len(
            dmo_data.index.to_frame()[["participant_id", "measurement_date"]].drop_duplicates()
        ) == len(self.index)
        if is_full_dataset:
            # Short circuit, if we have the full dataset, as this is a typical usecase and likely much faster.
            return dmo_data
        # That was the fastest version I found so far, but this is still slow as hell, if you have a large number of
        # participants still in the dataset....
        query_index = pd.MultiIndex.from_frame(self.index.drop("visit_type", axis=1))
        return dmo_data.reset_index("wb_id").loc[query_index].set_index("wb_id", append=True)

    @property
    def measurement_site(self) -> SITE_CODES:
        self.assert_is_single(None, "measurement_site")
        p_id, _ = self.group_label

        return self._get_participant_site_metadata().loc[p_id, "site"]

    @property
    def timezone(self) -> str:
        self.assert_is_single(None, "timezone")
        p_id, _ = self.group_label

        return self._get_participant_site_metadata().loc[p_id, "timezone"]

    def create_index(self) -> pd.DataFrame:
        return self.memory.cache(_create_index)(
            Path(self.dmo_export_path),
            Path(self.site_pid_map_path),
            UniversalHashableWrapper(self.TIME_ZONES),
            memory=self.memory,
        )
