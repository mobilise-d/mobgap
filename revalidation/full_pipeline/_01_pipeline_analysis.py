"""
.. _pipeline_val_results:

Performance of the Mobilise-D algorithm pipeline on the TVS dataset
==============================================================

.. warning:: On this page you will find preliminary results for a standardized revalidation of the pipeline and all
  of its algorithm.
  The current state, **TECHNICAL EXPERIMENTATION**.
  Don't use these results or make any assumptions based on them.
  We will update this page incrementally and provide further information, as soon as the state of any of the validation
  steps changes.

The following provides an analysis and comparison of the Mobilise-D algorithm pipeline on the TVS dataset
for the estimation of cadence, stride length, and walking speed (lab and free-living).
We look into the actual performance of the algorithms compared to the reference data.
Note, that at the time of writing, comparison with the original Matlab results is not possible, as these algorithms
were not run on the same version of the TVS dataset.

.. note:: If you are interested in how these results are calculated, head over to the
    :ref:`processing page <pipeline_val_gen>`.

"""

# %%
# Below are the list of algorithms that we will compare.
# Note, that we use the prefix "new" to refer to the reimplemented python algorithms.
# For the zjils algorithm, we compare both potential threshold values that were determined as part of the pre-validation
# analysis on the MsProject dataset.

algorithms = {
    "Official_MobiliseD_Pipeline": ("Official_MobiliseD_Pipeline", "new"),
}

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


def format_loaded_results(
    values: dict[tuple[str, str], pd.DataFrame],
    index_cols: list[str],
    convert_rel_error: bool = False,
) -> pd.DataFrame:
    formatted = (
        pd.concat(values, names=["algo", "version", *index_cols])
        .reset_index()
        .assign(
            algo_with_version=lambda df: df["algo"]
            + " ("
            + df["version"]
            + ")",
            _combined="combined",
        )
    )
    if not convert_rel_error:
        return formatted
    rel_cols = [c for c in formatted.columns if "rel_error" in c]
    formatted[rel_cols] = formatted[rel_cols] * 100
    return formatted


local_data_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "results"
    if int(get_env_var("MOBGAP_VALIDATION_USE_LOCAL_DATA", 0))
    else None
)
loader = ValidationResultLoader(
    "full_pipeline", result_path=local_data_path, version="main"
)


free_living_index_cols = [
    "cohort",
    "participant_id",
    "time_measure",
    "recording",
    "recording_name",
    "recording_name_pretty",
]

free_living_results = format_loaded_results(
    {
        v: loader.load_single_results(k, "free_living")
        for k, v in algorithms.items()
    },
    free_living_index_cols,
    convert_rel_error=True,
)

cohort_order = ["HA", "CHF", "COPD", "MS", "PD", "PFF"]

# %%
# Performance metrics
# -------------------
# Below you can find the setup for all performance metrics that we will calculate.
# We only use the `single__` results for the comparison.
# These results are calculated by first calculating the average stride length per WB.
# Then calculating the error metrics for each WB.
# Then we take the average over all WBs of a participant to get the `wb__` results.
from functools import partial

from mobgap.pipeline.evaluation import CustomErrorAggregations as A
from mobgap.utils.df_operations import (
    CustomOperation,
    apply_aggregations,
    apply_transformations,
)
from mobgap.utils.tables import FormatTransformer as F
from mobgap.utils.tables import RevalidationInfo, revalidation_table_styles

custom_aggs = [
    CustomOperation(
        identifier=None,
        function=A.n_datapoints,
        column_name=[("n_datapoints", "all")],
    ),
    CustomOperation(
        identifier=None,
        function=lambda df_: df_["wb__detected"].isna().sum(),
        column_name=[("n_nan_detected", "all")],
    ),
    ("wb__detected", ["mean", A.conf_intervals]),
    ("wb__reference", ["mean", A.conf_intervals]),
    ("wb__error", ["mean", A.loa]),
    ("wb__abs_error", ["mean", A.conf_intervals]),
    ("wb__rel_error", ["mean", A.conf_intervals]),
    ("wb__abs_rel_error", ["mean", A.conf_intervals]),
    CustomOperation(
        identifier=None,
        function=partial(
            A.icc,
            reference_col_name="wb__reference",
            detected_col_name="wb__detected",
            icc_type="icc2",
            # For the lab data, some trials have no results for the old algorithms.
            nan_policy="omit",
        ),
        column_name=[("icc", "wb_level"), ("icc_ci", "wb_level")],
    ),
]

format_transforms = [
    CustomOperation(
        identifier=None,
        function=lambda df_: df_[("n_datapoints", "all")].astype(int),
        column_name="n_datapoints",
    ),
    CustomOperation(
        identifier=None,
        function=lambda df_: df_[("n_nan_detected", "all")].astype(int),
        column_name="n_nan_detected",
    ),
    *(
        CustomOperation(
            identifier=None,
            function=partial(
                F.value_with_range,
                value_col=("mean", c),
                range_col=("conf_intervals", c),
            ),
            column_name=c,
        )
        for c in [
            "wb__reference",
            "wb__detected",
            "wb__abs_error",
            "wb__rel_error",
            "wb__abs_rel_error",
        ]
    ),
    CustomOperation(
        identifier=None,
        function=partial(
            F.value_with_range,
            value_col=("mean", "wb__error"),
            range_col=("loa", "wb__error"),
        ),
        column_name="wb__error",
    ),
    CustomOperation(
        identifier=None,
        function=partial(
            F.value_with_range,
            value_col=("icc", "wb_level"),
            range_col=("icc_ci", "wb_level"),
        ),
        column_name="icc",
    ),
]


final_names = {
    "n_datapoints": "# participants",
    "wb__detected": "WD mean and CI [m]",
    "wb__reference": "INDIP mean and CI [m]",
    "wb__error": "Bias and LoA [m]",
    "wb__abs_error": "Abs. Error [m]",
    "wb__rel_error": "Rel. Error [%]",
    "wb__abs_rel_error": "Abs. Rel. Error [%]",
    "icc": "ICC",
    "n_nan_detected": "# Failed WBs",
}

validation_thresholds = {
    "Abs. Error [m]": RevalidationInfo(threshold=None, higher_is_better=False),
    "Abs. Rel. Error [%]": RevalidationInfo(
        threshold=20, higher_is_better=False
    ),
    "ICC": RevalidationInfo(threshold=0.7, higher_is_better=True),
    "# Failed WBs": RevalidationInfo(threshold=None, higher_is_better=False),
}


def format_tables(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(apply_transformations, format_transforms)
        .rename(columns=final_names)
        .loc[:, list(final_names.values())]
    )


