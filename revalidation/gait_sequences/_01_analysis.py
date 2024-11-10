"""
Performance of the gait sequences algorithm on the TVS dataset
==============================================================

The following provides an analysis and comparison of the GSD performance on the TVS dataset (lab and free-living).
We look into the actual performance of the algorithms compared to the reference data and compare these results with
the performance of the original matlab algorithm.

We focus on the `single_results` (aka the performance per trail) and will aggregate it over multiple levels.

"""

from pathlib import Path

import pandas as pd


def load_single_results(
    algo_name: str, condition: str, base_path: Path, index_cols: list[str]
) -> pd.DataFrame:
    """Load the results for a specific condition."""
    return pd.read_csv(
        base_path / condition / algo_name / "single_results.csv",
    ).set_index(index_cols)


# %%
algorithms = {
    "GsdIonescu": ("GsdIonescu", "new"),
    "GsdAdaptiveIonescu": ("GsdAdaptiveIonescu", "new"),
    "GsdIluz": ("GsdIluz", "new"),
    "GsdIluz_orig_peak": ("GsdIluz", "new_orig_peak"),
}
# We only load the matlab algorithms that were also reimplemented
algorithms.update(
    {
        "matlab_EPFL_V1-improved_th": ("GsdIonescu", "orig"),
        "matlab_EPFL_V2-original": ("GsdAdaptiveIonescu", "orig"),
        "matlab_TA_Iluz-original": ("GsdIluz", "orig"),
    }
)

# %%
# The code below loads the data and prepares it for the analysis.
# By default, the data will be downloaded from an online repository (and cached locally).
# If you want to use a local copy of the data, you can set the `MOBGAP_VALIDATION_DATA_PATH` environment variable.
# and the MOBGAP_VALIDATION_USE_LOCA_DATA to `1`.
#
# The file download will print a couple log information, which can usually be ignored.
# You can also change the `version` parameter to load a different version of the data.
from mobgap.data.validation_results import ValidationResultLoader
from mobgap.utils.misc import get_env_var

local_data_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH"))
    if get_env_var("MOBGAP_VALIDATION_USE_LOCAL_DATA", 0)
    else None
)
loader = ValidationResultLoader(
    "gsd", result_path=local_data_path, version="main"
)


free_living_index_cols = [
    "cohort",
    "participant_id",
    "time_measure",
    "recording",
    "recording_name",
    "recording_name_pretty",
]

results = {
    v: loader.load_single_results(k, "free_living")
    for k, v in algorithms.items()
}
results = pd.concat(results, names=["algo", "version", *free_living_index_cols])
results_long = results.reset_index().assign(
    algo_with_version=lambda df: df["algo"] + " (" + df["version"] + ")",
    _combined="combined",
)


# %%
# All results across all cohorts.
# Note, that the `new_orig_peak` version is a variant of the new GsdIluz algorithm for which we tried to emulate the
# original peak detection algorithm as closely as possible.
# The regular `new` version uses a slightly modified peak detection algorithm.
import matplotlib.pyplot as plt
import seaborn as sns

hue_order = ["orig", "new", "new_orig_peak"]

sns.boxplot(
    data=results_long,
    x="algo",
    y="f1_score",
    hue="version",
    hue_order=hue_order,
)
plt.show()

# %%
# While this provides a good overview, it does not fully reflect how these algorithms perform on the different cohorts.
sns.boxplot(
    data=results_long, x="cohort", y="f1_score", hue="algo_with_version"
)
plt.show()

# %%
# Overview over all cohorts is good, but this is not how the GSD algorithms are used in our main pipeline.
# Here, the HA, CHF, and COPD cohort use the `GsdIluz` algorithm, while the `GsdIonescu` algorithm is used for the
# MS, PD, PFF cohorts.
# Let's look at the performance of these algorithms on the respective cohorts.
from mobgap.pipeline import MobilisedPipelineHealthy, MobilisedPipelineImpaired

low_impairment_algo = "GsdIluz"
low_impairment_cohorts = MobilisedPipelineHealthy().recommended_cohorts

low_impairment_results = results_long[
    results_long["cohort"].isin(low_impairment_cohorts)
].query("algo == @low_impairment_algo")

sns.boxplot(
    data=low_impairment_results,
    x="cohort",
    y="f1_score",
    hue="version",
    hue_order=hue_order,
)
sns.boxplot(
    data=low_impairment_results,
    x="_combined",
    y="f1_score",
    hue="version",
    hue_order=hue_order,
    legend=False,
)
plt.title(f"Low Impairment Cohorts ({low_impairment_algo})")
plt.show()

# %%
high_impairment_algo = "GsdIonescu"
high_impairment_cohorts = MobilisedPipelineImpaired().recommended_cohorts

high_impairment_results = results_long[
    results_long["cohort"].isin(high_impairment_cohorts)
].query("algo == @high_impairment_algo")

hue_order = ["orig", "new"]
sns.boxplot(
    data=high_impairment_results,
    x="cohort",
    y="f1_score",
    hue="version",
    hue_order=hue_order,
)
sns.boxplot(
    data=high_impairment_results,
    x="_combined",
    y="f1_score",
    hue="version",
    hue_order=hue_order,
    legend=False,
)
plt.title(f"High Impairment Cohorts ({high_impairment_algo})")
plt.show()
