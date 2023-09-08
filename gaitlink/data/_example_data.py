from pathlib import Path
from typing import Literal

from gaitlink import PACKAGE_ROOT

LOCAL_EXAMPLE_PATH = PACKAGE_ROOT.parent / "example_data/"


def _is_manually_installed() -> bool:
    return (LOCAL_EXAMPLE_PATH / "README.md").is_file()


def get_lab_example_data_path(cohort: Literal["HA", "PD", "MS", "COPD", "PFF", "CHF"], participant_id: str) -> Path:
    """Get the path to the example data for a given cohort and participant.

    Parameters
    ----------
    cohort
        The cohort of the participant.
    participant_id
        The id of the participant as string.

    Returns
    -------
    The path to the example data for the given cohort and participant.
    Within this folder the `data.mat` file and the `infoForAlgo.mat` file can be found.

    See Also
    --------
    load_mobilised_matlab_format

    """
    if not _is_manually_installed():
        # This is a redundant check for now, as we can not download the example data yet automatically
        # This means that the example data is only available, if the person cloned the repository
        raise FileNotFoundError(
            "It looks like the example data folder does not exist. "
            "This can happen if you installed gaitlink via a build package and not the raw git-repo. "
            "At the moment, we only support accessing the example data if you cloned the repo manually. "
        )

    potential_path = LOCAL_EXAMPLE_PATH / "data" / "lab" / cohort / participant_id
    if potential_path.is_dir():
        return potential_path
    raise FileNotFoundError(
        f"Could not find example data for {cohort}/{participant_id}. "
        f"Double check that example-data folder ({LOCAL_EXAMPLE_PATH}) if the expected chohort and "
        "participant id exist."
    )
