from pathlib import Path
from typing import Optional

from mobgap import PACKAGE_ROOT
from mobgap.data._mobilised_matlab_loader import (
    BaseGenericMobilisedDataset,
    GenericMobilisedDataset,
    matlab_dataset_docfiller,
)

LOCAL_EXAMPLE_PATH = PACKAGE_ROOT.parent / "example_data/"


def _is_manually_installed() -> bool:
    return (LOCAL_EXAMPLE_PATH / "README.md").is_file()


def get_example_cvs_dmo_data_path() -> Path:
    """Get the path to the example CVS DMO data.

    Returns
    -------
    The path to the example CVS DMO data.

    See Also
    --------
    MobilisedCvsDmoDataset

    """
    if not _is_manually_installed():
        # This is a redundant check for now, as we can not download the example data yet automatically
        # This means that the example data is only available, if the person cloned the repository
        raise FileNotFoundError(
            "It looks like the example data folder does not exist. "
            "This can happen if you installed mobgap via a build package and not the raw git-repo. "
            "At the moment, we only support accessing the example data if you cloned the repo manually. "
        )

    return LOCAL_EXAMPLE_PATH / "dmo_data" / "example_cvs_data"


def get_all_lab_example_data_paths() -> dict[tuple[str, str], Path]:
    """Get the paths to all lab example data.

    Returns
    -------
    A dictionary mapping the cohort and participant id to the path to the example data.

    See Also
    --------
    get_lab_example_data_path

    """
    if not _is_manually_installed():
        # This is a redundant check for now, as we can not download the example data yet automatically
        # This means that the example data is only available, if the person cloned the repository
        raise FileNotFoundError(
            "It looks like the example data folder does not exist. "
            "This can happen if you installed mobgap via a build package and not the raw git-repo. "
            "At the moment, we only support accessing the example data if you cloned the repo manually. "
        )

    potential_paths = (LOCAL_EXAMPLE_PATH / "data/lab").rglob("data.mat")
    return {(path.parents[1].name, path.parents[0].name): path.parent for path in potential_paths}


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

    @property
    def _metadata_level_names(self) -> Optional[tuple[str, ...]]:
        return "cohort", "participant_id"

    def _get_file_index_metadata(self, path: Path) -> tuple[str, ...]:
        return path.parents[1].name, path.parents[0].name
