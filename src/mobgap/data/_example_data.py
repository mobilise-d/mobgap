import fnmatch
from importlib.resources import files
from pathlib import Path
from typing import Optional

from mobgap import PROJECT_ROOT, __version__
from mobgap.data._mobilised_matlab_loader import (
    BaseGenericMobilisedDataset,
    GenericMobilisedDataset,
    matlab_dataset_docfiller,
)

LOCAL_EXAMPLE_PATH = PROJECT_ROOT / "example_data/"

BRIAN = None

if not (LOCAL_EXAMPLE_PATH / "README.md").is_file():
    import pooch

    GITHUB_FOLDER_PATH = "https://raw.githubusercontent.com/mobilise-d/mobgap/{version}/example_data/"

    BRIAN = pooch.create(
        # Use the default cache folder for the operating system
        path=pooch.os_cache("mobgap"),
        # The remote data is on Github
        base_url=GITHUB_FOLDER_PATH,
        version=f"v{__version__}",
        version_dev="main",
        registry=None,
        # The name of an environment variable that *can* overwrite the path
        env="MOBGAP_DATA_DIR",
    )

    # Get registry file from package_data
    # The registry file can be recreated by running the task `poe update_example_data`
    registry_file = files("mobgap") / "data/_example_data_registry.txt"
    # Load this registry file
    BRIAN.load_registry(registry_file)


def _pooch_get_folder(folder_path: Path) -> Path:
    """Get the path to the example data folder.

    If the data is not available locally, it will be downloaded from the remote repository.
    For this we use pooch to download all files that start with the folder name.
    """
    if BRIAN is None:
        return folder_path

    rel_folder_path = folder_path.relative_to(LOCAL_EXAMPLE_PATH)

    matching_files = []
    for f in BRIAN.registry:
        try:
            Path(f).relative_to(rel_folder_path)
        except ValueError:
            continue
        matching_files.append(Path(BRIAN.fetch(f, progressbar=True)))

    print(matching_files)
    return BRIAN.abspath / rel_folder_path


def _pooch_glob(base_path: Path, pattern: str) -> list[Path]:
    """Get the path to the example data file.

    If the data is not available locally, it will be downloaded from the remote repository.
    For this we use pooch to download the file.
    """
    if BRIAN is None:
        return list(base_path.rglob(pattern))

    rel_base_path = base_path.relative_to(LOCAL_EXAMPLE_PATH)

    matching_files = []
    for f in BRIAN.registry:
        try:
            rel_f = Path(f).relative_to(rel_base_path)
        except ValueError:
            continue

        if fnmatch.fnmatch(str(rel_f), pattern):
            matching_files.append(Path(BRIAN.fetch(f, progressbar=True)))

    return matching_files


def get_example_cvs_dmo_data_path() -> Path:
    """Get the path to the example CVS DMO data.

    Returns
    -------
    The path to the example CVS DMO data.

    See Also
    --------
    MobilisedCvsDmoDataset

    """
    return _pooch_get_folder(LOCAL_EXAMPLE_PATH / "dmo_data/example_cvs_data")


def get_all_lab_example_data_paths() -> dict[tuple[str, str], Path]:
    """Get the paths to all lab example data.

    Returns
    -------
    A dictionary mapping the cohort and participant id to the path to the example data.

    See Also
    --------
    get_lab_example_data_path

    """
    # We also fetch the infoForAlgo.mat files in case they need to be downloaded
    _ = _pooch_glob(LOCAL_EXAMPLE_PATH / "data/lab", "**/infoForAlgo.mat")
    _ = _pooch_glob(LOCAL_EXAMPLE_PATH / "data/lab", "**/test_list.json")
    potential_paths = _pooch_glob(LOCAL_EXAMPLE_PATH / "data/lab", "**/data.mat")
    return {(path.parents[1].name, path.parents[0].name): path.parent for path in potential_paths}


def get_example_csv_data_path() -> Path:
    """Get the path to the example CSV data.

    Returns
    -------
    The path to the example CSV data.

    See Also
    --------
    MobilisedCsvDataset

    """
    return _pooch_get_folder(LOCAL_EXAMPLE_PATH / "data_csv")


@matlab_dataset_docfiller
class LabExampleDataset(BaseGenericMobilisedDataset):
    """A dataset containing all lab example data provided with mobgap.

    Parameters
    ----------
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

    @property
    def _paths_list(self) -> list[Path]:
        return [p / "data.mat" for p in sorted(get_all_lab_example_data_paths().values())]

    @property
    def _test_level_names(self) -> tuple[str, ...]:
        return GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"]

    def _get_measurement_condition(self) -> str:
        return "laboratory"

    @property
    def _metadata_level_names(self) -> Optional[tuple[str, ...]]:
        return "cohort", "participant_id"

    def _relpath_to_precomputed_test_list(self) -> str:
        return "test_list.json"

    def _get_file_index_metadata(self, path: Path) -> tuple[str, ...]:
        return path.parents[1].name, path.parents[0].name
