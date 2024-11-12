"""Loader for the results of the algorithm validation."""

import warnings
from pathlib import Path
from typing import Final, Literal, Optional, Union

import pandas as pd
import pooch


class ValidationResultLoader:
    """Load the revalidation results either by downloading them or from a local folder."""

    VALIDATION_REPO_DATA = "https://raw.githubusercontent.com/mobilise-d/mobgap_validation/{version}/"

    CONDITION_INDEX_COLS: Final[dict[Literal["free_living", "laboratory"], list[str]]] = {
        "free_living": [
            "cohort",
            "participant_id",
            "time_measure",
            "recording",
            "recording_name",
            "recording_name_pretty",
        ],
        "laboratory": [
            "cohort",
            "participant_id",
            "time_measure",
            "test",
            "trial",
            "test_name",
            "test_name_pretty",
        ],
    }

    def __init__(self, sub_folder: str, *, result_path: Optional[Path] = None, version: str = "main") -> None:
        self.sub_folder = sub_folder
        self.result_path = result_path
        self.version = version

        if self.result_path is not None and version != "main":
            warnings.warn(
                "For local loading, we always use the version available in the local folder. "
                "This means the `version` parameter is ignored.",
                stacklevel=1,
            )

        if self.result_path is None:
            self.brian = pooch.create(
                # Use the default cache folder for the operating system
                path=pooch.os_cache("mobgap"),
                # The remote data is on Github
                base_url=f"{self.VALIDATION_REPO_DATA.format(version=version)}/results",
                registry=None,
                # The name of an environment variable that *can* overwrite the path
                env="MOBGAP_DATA_DIR",
            )

    @property
    def _base_path(self) -> Union[Path, str]:
        if self.result_path is not None:
            return self.result_path / self.sub_folder
        return self.sub_folder

    def load_single_results(self, algo_name: str, condition: Literal["free_living", "laboratory"]) -> pd.DataFrame:
        """Load the results for a specific condition."""
        if self.result_path is not None:
            return pd.read_csv(
                self._base_path / condition / algo_name / "single_results.csv",
            ).set_index(self.CONDITION_INDEX_COLS[condition])
        if not self.brian.registry:
            registry = pooch.retrieve(
                f"{self.VALIDATION_REPO_DATA.format(version=self.version)}/results_file_registry.txt",
                known_hash=None,
            )
            self.brian.load_registry(registry)
        return pd.read_csv(
            self.brian.fetch(f"{self.sub_folder}/{condition}/{algo_name}/single_results.csv"),
        ).set_index(self.CONDITION_INDEX_COLS[condition])
