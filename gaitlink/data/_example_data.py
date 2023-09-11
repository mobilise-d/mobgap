from pathlib import Path

from gaitlink import PACKAGE_ROOT

LOCAL_EXAMPLE_PATH = PACKAGE_ROOT.parent / "example_data/"


def _is_manually_installed() -> bool:
    return (LOCAL_EXAMPLE_PATH / "README.md").is_file()


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
            "This can happen if you installed gaitlink via a build package and not the raw git-repo. "
            "At the moment, we only support accessing the example data if you cloned the repo manually. "
        )

    potential_paths = (LOCAL_EXAMPLE_PATH / "data/lab").rglob("data.mat")
    return {(path.parents[1].name, path.parents[0].name): path.parent for path in potential_paths}
