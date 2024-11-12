"""
.. warning:: On this page you will find preliminary results for a standardized revalidation of the pipeline and all
  of its algorithm.
  The current state, **TECHNICAL EXPERIMENTATION**.
  Don't use these results or make any assumptions based on them.
  We will update this page incrementatlly and provide further information, as soon as the state of any of the validation
  steps changes.

.. _gsd_val_results

Performance of the gait sequences algorithm on the TVS dataset
==============================================================

The following provides an analysis and comparison of the GSD performance on the TVS dataset (lab and free-living).
We look into the actual performance of the algorithms compared to the reference data and compare these results with
the performance of the original matlab algorithm.

.. note:: If you are interested in how these results are calculated, head over to the
   :ref:`processing page <gsd_val_results>`.

We focus on the `single_results` (aka the performance per trail) and will aggregate it over multiple levels.

"""

# %%
# Below are the list of algorithms that we will compare.
# Note, that we use the prefix "new" to refer to the reimplemented python algorithms and "orig" to refer to the
# original matlab algorithms.
# In case of the GsdIluz algorithm, we also have two reimplemented versions.
# The version `new` uses a slightly modified peak detection algorithm, while the version `new_orig_peak` tries to
# emulate the original peak detection algorithm as closely as possible.
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
from pathlib import Path

import pandas as pd
from mobgap.data.validation_results import ValidationResultLoader
from mobgap.utils.misc import get_env_var

local_data_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "results"
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
cohort_order = ["HA", "CHF", "COPD", "MS", "PD", "PFF"]

# %%
# Free-Living Comparison
# ----------------------
# We focus the comparison on the free-living data, as this is the most relevant considering our final use-case.
# In the free-living data, there is one 2.5 hour recording per participant.
# This means, each datapoint in the plots below and in the summary statistics represents one participant.
#
# For each participant, performance metrics were calculated by classifying each sample in the recording as either
# TP, FP, TN, or FN.
# Based on these values recall (sensitivity), precision (positive predictive value), F1 score, accuracy, specificity
# and many other metrics were calculated.
# On top of that the duration of overall detected gait per participant was calculated.
# From this we calculate the mean and confidence interval for both systems, the bias and limits of agreement (LoA)
# between the algorithm output and the reference data, the absolute error and the ICC.
#
# All results across all cohorts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Note, that the `new_orig_peak` version is a variant of the new ``GsdIluz`` algorithm for which we tried to emulate the
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
from functools import partial

from mobgap.pipeline.evaluation import CustomErrorAggregations as A
from mobgap.utils.df_operations import CustomOperation, apply_aggregations

custom_aggs = [
    CustomOperation(
        identifier=None,
        function=A.n_datapoints,
        column_name=[("n_participants", "all")],
    ),
    ("recall", ["mean", A.conf_intervals]),
    ("precision", ["mean", A.conf_intervals]),
    ("f1_score", ["mean", A.conf_intervals]),
    ("accuracy", ["mean", A.conf_intervals]),
    ("specificity", ["mean", A.conf_intervals]),
    ("reference_gs_duration_s", ["mean", A.conf_intervals]),
    ("detected_gs_duration_s", ["mean", A.conf_intervals]),
    ("gs_duration_error_s", ["mean", A.loa]),
    ("gs_absolute_duration_error_s", ["mean", A.conf_intervals]),
    CustomOperation(
        identifier=None,
        function=partial(
            A.icc,
            reference_col_name="reference_gs_duration_s",
            detected_col_name="detected_gs_duration_s",
            icc_type="icc2",
        ),
        column_name=[("icc", "gs_duration_s"), ("icc_ci", "gs_duration_s")],
    ),
]

perf_metrics_all = results.pipe(apply_aggregations, custom_aggs)
perf_metrics_all

# %%
# Per Cohort
# ~~~~~~~~~~
# While this provides a good overview, it does not fully reflect how these algorithms perform on the different cohorts.
sns.boxplot(
    data=results_long, x="cohort", y="f1_score", hue="algo_with_version"
)
plt.show()

perf_metrics_per_cohort = (
    results.groupby(["cohort", "algo", "version"])
    .apply(apply_aggregations, custom_aggs)
    .swaplevel(axis=1)
    .loc[cohort_order]
)
perf_metrics_per_cohort

# %%
# Per relevant cohort
# ~~~~~~~~~~~~~~~~~~~
# Overview over all cohorts is good, but this is not how the GSD algorithms are used in our main pipeline.
# Here, the HA, CHF, and COPD cohort use the ``GsdIluz` algorithm, while the ``GsdIonescu`` algorithm is used for the
# MS, PD, PFF cohorts.
# Let's look at the performance of these algorithms on the respective cohorts.
from mobgap.pipeline import MobilisedPipelineHealthy, MobilisedPipelineImpaired

low_impairment_algo = "GsdIluz"
low_impairment_cohorts = list(MobilisedPipelineHealthy().recommended_cohorts)

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
perf_metrics_per_cohort.loc[
    pd.IndexSlice[low_impairment_cohorts, low_impairment_algo], :
].reset_index("algo", drop=True)

# %%
high_impairment_algo = "GsdIonescu"
high_impairment_cohorts = list(MobilisedPipelineImpaired().recommended_cohorts)

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

# %%
perf_metrics_per_cohort.loc[
    pd.IndexSlice[high_impairment_cohorts, high_impairment_algo], :
].reset_index("algo", drop=True)
