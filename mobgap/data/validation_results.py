"""Loader for the results of the algorithm validation."""

import sys
import tempfile
import warnings
from pathlib import Path
from typing import Final, Literal, Optional, Union

import pandas as pd
import pooch
import requests

from mobgap import PACKAGE_ROOT


def download_file_to_temp(url: str, filename: Optional[str] = None, headers: Optional[dict] = None) -> Path:
    """
    Download a file from URL with custom headers and store it in a temporary directory.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    filename : str, optional
        Custom filename to use. If None, extracts from URL.
    headers : dict, optional
        Custom HTTP headers to include in the request.

    Returns
    -------
    pathlib.Path
        Path object to the downloaded file.

    Raises
    ------
    ValueError
        If URL is invalid.
    requests.exceptions.HTTPError
        If HTTP request returns an unsuccessful status code.
    requests.exceptions.RequestException
        For other request-related errors.
    IOError
        If there's an error writing the file.

    Examples
    --------
    >>> file_path = download_file_to_temp("https://example.com/file.pdf")
    >>> file_path
    PosixPath('/tmp/file.pdf')

    >>> custom_headers = {"User-Agent": "Mozilla/5.0"}
    >>> file_path = download_file_to_temp(
    ...     "https://example.com/data.csv",
    ...     filename="mydata.csv",
    ...     headers=custom_headers,
    ... )
    """
    if not url or not isinstance(url, str):
        raise ValueError("Invalid URL provided")

    try:
        # Send GET request with custom headers
        response = requests.get(url, headers=headers, stream=True, timeout=3)

        # Check if the request was successful
        response.raise_for_status()

        # Create a temporary directory as a Path object
        temp_dir = Path(tempfile.gettempdir())

        # Use the provided filename or extract from URL
        if filename is None:
            filename = url.split("/")[-1]
            if not filename:  # If URL ends with /, use a default name
                filename = "downloaded_file"

        # Create the full file path
        file_path = temp_dir / filename

        # Write the content to a file
        with file_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    file.write(chunk)
    except requests.exceptions.HTTPError:
        # Re-raise HTTP errors (like 404, 500) as is
        raise
    except requests.exceptions.RequestException:
        # Re-raise network-related errors
        raise
    except OSError as e:
        # Re-raise file writing errors
        raise OSError(f"Error writing to temporary file: {e!s}") from e
    else:
        return file_path


def find_best_version() -> str:
    """Find the best version of the library to use for loading the validation results.

    This is used to determine the version of the library that is currently used.
    If the library is installed as a dependency, we use the version of the library.
    If the library is cloned from git, we use the current branch or tag.

    Returns
    -------
        The version of the library.
    """
    try:
        import git

        repo = git.Repo(PACKAGE_ROOT.parent)
    except:  # noqa: E722
        repo = None

    if repo is not None:
        if repo.head.is_detached:
            raise RuntimeError(
                "The current version of the library is a detached HEAD. "
                "Please specify a version to load the validation results."
            )
        try:
            tags = [tag.name for tag in repo.tags if tag.commit == repo.head.commit]
            if tags:
                return tags[0]  # Return the first tag if multiple exist
        except:  # noqa: E722
            pass

        return repo.active_branch.name
    # If the library is installed as a dependency, we use the version of the library
    try:
        import pkg_resources

        return pkg_resources.get_distribution("mobgap").version
    except Exception as e:
        raise RuntimeError(
            "We could neither find the version of the library in the git repository nor in the installed package. "
            "Specify the version manually."
        ) from e


def check_url_exists(url: str, headers: Optional[dict] = None) -> bool:
    """Check if a URL exists by sending a HEAD request."""
    try:
        response = requests.head(url, timeout=2, headers=headers)
    except requests.RequestException:
        return False
    else:
        return response.status_code == 200


class ValidationResultLoader:
    """Load the revalidation results either by downloading them or from a local folder.

    This is a helper to load the validation results of the algorithms.
    They are stored in a dedicated git repository (https://github.com/mobilise-d/mobgap_validation).
    This repository follows the same versioning as the software library.
    So this means, if you want the validation results for the algorithms of a specific version, you can just find a
    tag with the same version in the validation repository and use it here.
    Note, that the validation results might be identical, if no changes were made to the algorithms between two
    versions.
    In this case, multiple tags might point to the same results.

    Alternatively, you can load the data from a local folder.
    This might be helpful during development.
    For more on that see the Notes section.

    Parameters
    ----------
    sub_folder
        The sub folder within the repository you want to load the results from.
        The sub folder usually represents a set of experiments or a group of algorithm that was validated.
        For example, `full_pipeline` is the subfolder for the full pipeline results calulated on the TVS dataset.
    local_result_path
        An optional local path to a folder where the results are stored.
        If specified, the results are loaded from this folder instead of downloading them.
        The value always takes precedence over the `version` parameter.
    version
        A valid git specifier (tag, branch, commit hash) to load the results from.
        If not specified, the result best fitting to your current version of the library is used.
        For more information see the Notes section.
    fallback_version
        The result version that is used, if the automatically determined version (version == None) is not available.

    Notes
    -----
    We determine the version and with that the location where the results are loaded from in the following order:

    1. If `local_result_path` is specified, we use this path and ignore the `version` parameter.
    2. If `local_result_path` is not specified, and `version` is specified, we use the `version` parameter.
       Results are then loaded as one request per file to the github api using the specified version as part of the URL.
    3. If `local_result_path` is not specified and `version` is not specified, we use the following logic to determine
       the version:
       - If the library has been installed as a dependency (vs. cloned from git for development), we use the version of
         the library as tag.
       - If the library has been cloned from git for development, we use the current we go in order of
         priority tag > branch.
         In case of a detached HEAD, we throw an error forcing you to specifying an explicit version that you want to
         load.
    4. When a version is automatically determined, we do a test request to the github api to see if the version is
       available.
       If not, we use the `fallback_version` parameter to load the results from.
       If this also fails, you will se the error message

    """

    VALIDATION_REPO_DATA = "https://raw.githubusercontent.com/mobilise-d/mobgap_validation/{version}"
    HTTP_HEADERS: Final = {
        # Looks like github rate limits and throttles common user agents to completely unusable rates...
        # Curl useragent seems to work for now.
        "User-Agent": "curl/8.12.1"
    }

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

    def __init__(
        self,
        sub_folder: str,
        *,
        local_result_path: Optional[Union[str, Path]] = None,
        version: Optional[str] = None,
        fallback_version: Optional[str] = "main",
    ) -> None:
        self.sub_folder = sub_folder
        self.local_result_path = local_result_path
        self.version = version
        self.fallback_version = fallback_version

        if self.local_result_path is not None and version is not None:
            warnings.warn(
                "For local loading, we always use the version available in the local folder. "
                "This means the `version` parameter is ignored.",
                stacklevel=1,
            )

        if self.local_result_path is None:
            if version is None:
                version = find_best_version()
                # Test request to see if the version is available
                if not check_url_exists(
                    f"{self.VALIDATION_REPO_DATA.format(version=version)}/results_file_registry.txt",
                    headers=self.HTTP_HEADERS,
                ):
                    if not self.fallback_version:
                        raise RuntimeError(
                            "No version was specified, the automatically determined version is not available and no "
                            "fallback version was specified."
                        )
                    warnings.warn(
                        f"The automatically determined version {version} is not available. "
                        f"Using the fallback version {self.fallback_version} instead.",
                        stacklevel=1,
                    )
                    version = self.fallback_version
            self._resolved_version = version
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
        if self.local_result_path is not None:
            return Path(self.local_result_path) / "results" / self.sub_folder
        return self.sub_folder

    @property
    def _downloader(self) -> pooch.HTTPDownloader:
        return pooch.HTTPDownloader(headers=self.HTTP_HEADERS, progressbar=True)

    def load_single_csv_file(
        self, algo_name: str, condition: Literal["free_living", "laboratory"], file_name: str
    ) -> pd.DataFrame:
        """Load any of the results files by name."""
        if self.local_result_path is not None:
            return pd.read_csv(
                self._base_path / condition / algo_name / file_name,
            ).set_index(self.CONDITION_INDEX_COLS[condition])
        if not self.brian.registry:
            assert self._resolved_version is not None
            print(f"Downloading registry file for version {self._resolved_version}", file=sys.stderr)
            # We download the registry direclty and not with pooch resolve, because otherwise poocch cache might
            # screw us, as it will not download a new version of the file if it is already in the cache.
            registry = download_file_to_temp(
                f"{self.VALIDATION_REPO_DATA.format(version=self._resolved_version)}/results_file_registry.txt",
                filename="results_file_registry.txt",
                headers=self.HTTP_HEADERS,
            )
            self.brian.load_registry(registry)
        return pd.read_csv(
            self.brian.fetch(f"{self.sub_folder}/{condition}/{algo_name}/{file_name}", downloader=self._downloader),
        ).set_index(self.CONDITION_INDEX_COLS[condition])

    def load_single_results(self, algo_name: str, condition: Literal["free_living", "laboratory"]) -> pd.DataFrame:
        """Load the results for a specific condition."""
        return self.load_single_csv_file(algo_name, condition, "single_results.csv")

    def load_agg_results(self, algo_name: str, condition: Literal["free_living", "laboratory"]) -> pd.DataFrame:
        """Load the aggregated results for a specific condition."""
        return self.load_single_csv_file(algo_name, condition, "aggregated_results.csv")
