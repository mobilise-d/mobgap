r"""
.. _gsd_performance_index:

GSD Evaluation - Performance index
==============

This example shows how to calculate the performance index based on running gait detection on multiple data points.
"""

# %%
import pandas as pd
from mobgap.data import LabExampleDataset
from mobgap.gsd import GsdIluz


# %%
# Running a full evaluation pipeline
# ----------------------------------
# Instead of manually evaluating and investigating the performance of a GSD algorithm on a single piece of data, we
# often want to run a full evaluation on an entire dataset.
# This can be done using the :class:`~mobgap.gsd.evaluation.GsdEvaluationPipeline` class and some ``tpcp`` functions.
#
# But let's start with selecting some data.
# We want to use all the simulated real-world walking data from the INDIP reference system (Test11).
simulated_real_world_walking = LabExampleDataset(reference_system="INDIP").get_subset(
    test="Test11"
)

simulated_real_world_walking
# %%
# Now we can use the :class:`~mobgap.gsd.evaluation.GsdEvaluationPipeline` class to directly run a Gsd algorithm on
# a datapoint.
# The pipeline takes care of extracting the required data.
from mobgap.gsd.pipeline import GsdEmulationPipeline

pipeline = GsdEmulationPipeline(GsdIluz())

pipeline.safe_run(simulated_real_world_walking[0]).gs_list_

# %%
# Note, that this did just "run" the pipeline on a single datapoint.
# If we want to run it on all datapoints and evaluate the performance of the algorithm, we can use the
# :func:`~tpcp.validate.validate` function.
#
# It uses the build in ``score`` method of the pipeline to calculate the performance of the algorithm on each datapoint
# and then takes the mean of the results.
# All mean and individual results are returned in huge dictionary that can be easily converted to a pandas DataFrame.
from tpcp.validate import validate

evaluation_results = pd.DataFrame(validate(pipeline, simulated_real_world_walking))

evaluation_results_dict = evaluation_results.drop(
    ["single_reference", "single_detected"], axis=1
).T[0]


# %%
# Print the list of available scoring metrics
evaluation_results_dict

# %%
# Calculate performance index
# ------------------------------
# Bonci et al (2020) (https://www.mdpi.com/1424-8220/20/22/6509) suggest a methodology to determine a performance
# index that combines multiple metrics into a single value.

import numpy as np
from typing import Literal


def normalize(
    x, criterion: Literal["benefit", "cost"] = "benefit", normalization: bool = False
):
    """
    Normalize a given array of values based on Bonci et al.

    Parameters:
    - x (array-like): The input array to be normalized.
    - criterion (str, optional): The type of normalization to be applied. Valid options are "cost" and "benefit" (default).
    - normalization (bool, optional): Whether to perform normalization. If True, the values will be normalized between
                                      the minimum and maximum values in the array. If False, the values will be
                                      normalized between 0 and 1 (e.g., relevant for metrics such as precision, recall, ...). Default is False.

    Returns:
    array-like: The normalized array.

    """
    x = np.array(x)

    if normalization:
        max_val = max(x)
        min_val = min(x)
    else:
        max_val = 1
        min_val = 0

    if criterion == "cost":
        return (max_val - x) / (max_val - min_val)
    elif criterion == "benefit":
        return (x - min_val) / (max_val - min_val)
    else:
        return x


# E.g.:
normalize(
    evaluation_results_dict["single_num_gs_absolute_relative_error_log"],
    criterion="cost",
    normalization=True,
)

# %%
evaluation_results_dict["single_num_gs_absolute_relative_error_log"]
# %%
# Define metrics that are used to calculate the performance index
# For each metric, the underlying score, criterion (cost/benefit), aggregation (e.g., mean, std, ...), and weight needs to be defined
weighting_factor_micoamigo = {
    "recall_mean": {
        "metric": "single_recall",
        "criterion": "benefit",
        "normalization": False,
        "aggregation": lambda x: np.mean(x),
        "weight": 0.117,
    },
    "specificity_mean": {
        "metric": "single_specificity",
        "criterion": "benefit",
        "normalization": False,
        "aggregation": lambda x: np.mean(x),
        "weight": 0.178,
    },
    "precision_mean": {
        "metric": "single_precision",
        "criterion": "benefit",
        "normalization": False,
        "aggregation": lambda x: np.mean(x),
        "weight": 0.105,
    },
    "accuracy_mean": {
        "metric": "single_accuracy",
        "criterion": "benefit",
        "normalization": False,
        "aggregation": lambda x: np.mean(x),
        "weight": 0.160,
    },
    "gs_absolute_relative_duration_error_mean": {
        "metric": "single_gs_absolute_relative_duration_error_log",
        "criterion": "cost",
        "normalization": True,
        "aggregation": lambda x: np.mean(x),
        "weight": 0.122,
    },
    "gs_absolute_relative_duration_error_std": {
        "metric": "single_gs_absolute_relative_duration_error_log",
        "criterion": "cost",
        "normalization": True,
        "aggregation": lambda x: np.std(x),
        "weight": 0.122,
    },
    # "icc_mean": {
    #     "metric": ["single_detected_gs_duration_s", "single_reference_gs_duration_s"],
    #     "normalization": None,
    #     "aggregation": lambda x: np.intraclass_corr(x),
    #     "weight": 0.196,
    # },
}

# Calculate performance index
performance_index = sum(
    weighting_factor_micoamigo[key]["aggregation"](
        normalize(
            evaluation_results_dict[weighting_factor_micoamigo[key]["metric"]],
            criterion=weighting_factor_micoamigo[key]["criterion"],
            normalization=weighting_factor_micoamigo[key]["normalization"],
        )
    )
    * weighting_factor_micoamigo[key]["weight"]
    for key in weighting_factor_micoamigo
)

performance_index


# These are the weights used in Kluge et al.
# weighting_factor_kluge = {
#     "recall": 0.117,
#     "specificity": 0.151,
#     "precision": 0.089,
#     "accuracy": 0.135,
#     "gs_relative_duration_error": 0.104,
#     # "Bias_rel_std": 0.104,
#     # "ICC_mean": 0.167,
#     # "Bias_nr_gs_mean": 0.15,
# }
