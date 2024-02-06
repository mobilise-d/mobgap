"""
GSD Iluz
========

This example shows how to use the GSD Iluz algorithm and some examples on how the results compare to the original
matlab implementation.

We start by defining some helpers for plotting and loading the data.
You can skip them for now and jump directly to "Performance on a single lab trial", if you just want to see how to
apply the algorithm.
"""
# %%
# Plotting Helper
# ---------------
# We define a helper function to plot the results of the algorithm.
# Just ignore this function for now.
import matplotlib.pyplot as plt


def plot_gsd_outputs(data, **kwargs):
    fig, ax = plt.subplots()

    ax.plot(data["acc_x"].to_numpy(), label="acc_x")

    color_cycle = iter(plt.rcParams["axes.prop_cycle"])

    y_max = 1.1
    plot_props = [
        {"data": v, "label": k, "alpha": 0.2, "ymax": (y_max := y_max - 0.1), "color": next(color_cycle)["color"]}
        for k, v in kwargs.items()
    ]

    for props in plot_props:
        for gsd in props.pop("data").itertuples(index=False):
            ax.axvspan(gsd.start, gsd.end, label=props.pop("label", None), **props)

    ax.legend()
    return fig, ax


# %%
# Loading some example data
# -------------------------
# .. note :: More infos about data loading can be found in the :ref:`data loading example <data_loading_example>`.
#
# We load example data from the lab dataset together with the INDIP reference system.
# We will use the INDIP "WB" output as ground truth.
# Note, that the "WB" (Walking Bout) output is further processed than a normal "Gait Sequence".
# This means we expect Gait Sequences to contain some false positives compared to the "WB" output.
# However, a good gait sequence detection algorithm should have high sensitivity (i.e. contain all the "WBs"
# of the reference system).
import json

import pandas as pd

from gaitlink import PACKAGE_ROOT
from gaitlink.data import LabExampleDataset

lab_example_data = LabExampleDataset(reference_system="INDIP")


def load_matlab_output(datapoint):
    p = datapoint.group_label
    with (
        PACKAGE_ROOT.parent
        / f"example_data/original_results/gsd_iluz/lab/{p.cohort}/{p.participant_id}/GSDA_Output.json"
    ).open() as f:
        original_results = json.load(f)["GSDA_Output"][p.time_measure][p.test][p.trial]["SU"]["LowerBack"]["GSD"]

    if not isinstance(original_results, list):
        original_results = [original_results]
    return (
        pd.DataFrame.from_records(original_results).rename(
            {"GaitSequence_Start": "start", "GaitSequence_End": "end"}, axis=1
        )[["start", "end"]]
        * datapoint.sampling_rate_hz
    )


# %%
# Performance on a single lab trial
# ---------------------------------
# Below we apply the algorithm to a lab trail, where we only expect a single gait sequence.
from gaitlink.gsd import GsdIluz

short_trial = lab_example_data.get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2")
short_trial_matlab_output = load_matlab_output(short_trial)
short_trial_reference_parameters = short_trial.reference_parameters_.wb_list

short_trial_output = GsdIluz().detect(short_trial.data["LowerBack"], sampling_rate_hz=short_trial.sampling_rate_hz)

print("Reference Parameters:\n\n", short_trial_reference_parameters)
print("\nMatlab Output:\n\n", short_trial_matlab_output)
print("\nPython Output:\n\n", short_trial_output.gs_list_)
# %%
# When we plot the output, we can see that the python version is a little more sensitive than the matlab version.
# It includes a section of the signal before the region classified as WB by the reference system.
# Both algorithm implementations produce a gait sequence that extends beyond the end of the reference system.

fig, ax = plot_gsd_outputs(
    short_trial.data["LowerBack"],
    reference=short_trial_reference_parameters,
    matlab=short_trial_matlab_output,
    python=short_trial_output.gs_list_,
)
fig.show()

# %%
# Performance on a longer lab trial
# ---------------------------------
# Below we apply the algorithm to a lab trail that contains activities of daily living.
# This is a more challenging scenario, as we expect multiple gait sequences.
long_trial = lab_example_data.get_subset(cohort="MS", participant_id="001", test="Test11", trial="Trial1")
long_trial_matlab_output = load_matlab_output(long_trial)
long_trial_reference_parameters = long_trial.reference_parameters_.wb_list

long_trial_output = GsdIluz().detect(long_trial.data["LowerBack"], sampling_rate_hz=long_trial.sampling_rate_hz)

print("Reference Parameters:\n\n", long_trial_reference_parameters)
print("\nMatlab Output:\n\n", long_trial_matlab_output)
print("\nPython Output:\n\n", long_trial_output.gs_list_)

# %%
# When we plot the output, we can see again that the python version is more sensitive.
# It detects longer gait sequences and even one entire gait sequence that is not detected by the matlab version.

fig, _ = plot_gsd_outputs(
    long_trial.data["LowerBack"],
    reference=long_trial_reference_parameters,
    matlab=long_trial_matlab_output,
    python=long_trial_output.gs_list_,
)
fig.show()

# %%
# Changing the parameters
# -----------------------
# The Python version aims to expose all relevant parameters of the algorithm.
# The `GsdlIluz` algorithm has a lot of parameters that can be modified.
# Finding a combination of parameters that works well for all scenarios is difficult.
# Below we show, just how to modify them in general.
#
# We modify one of the basic parameters, the window length.
# This can effect all parts of the output.
# In this case, we can see that all GSDs are slightly longer and that we now detect a gait sequence that was not
# detected before.

long_trial_output_modified = GsdIluz(window_length_s=5, window_overlap=0.8).detect(
    long_trial.data["LowerBack"], sampling_rate_hz=long_trial.sampling_rate_hz
)

print("Reference Parameters:\n\n", long_trial_reference_parameters)
print("\nPython Output:\n\n", long_trial_output.gs_list_)
print("\nPython Output Modified:\n\n", long_trial_output_modified.gs_list_)

fig, _ = plot_gsd_outputs(
    long_trial.data["LowerBack"],
    reference=long_trial_reference_parameters,
    python=long_trial_output.gs_list_,
    python_modified=long_trial_output_modified.gs_list_,
)
fig.show()

# %%
# Validation of algorithm output against a reference
# --------------------------------------------------
# Let's quantify how the Python output compares to the reference labels.
# To do this, we use the `categorize_intervals` function to compare detected gait sequences to reference labels
# sample by sample.

from gaitlink.gsd.validation import categorize_intervals

categorized_intervals = categorize_intervals(long_trial_output.gs_list_, long_trial_reference_parameters)

# %%
# The function returns a DataFrame containing `start` and `end`  index of the resulting matched intervals together with
# a `match_type` column that contains the type of match for each interval, i.e. `tp` for true positive, `fp` for false
# positive, and `fn` for false negative.
# These intervals can not be interpreted as gait sequences, but are rather subsequences of the detected gait sequences
# categorizing correctly detected samples (`tp`), falsely detected samples (`fp`), and samples
# from the reference gsd list that were not detected (`fn`).
# Note that the true negative intervals are not explicitly returned, but can be inferred from the other intervals
# (if the total length of the underlying recording is known), as everything between them is considered as true negative.

print("Matched Intervals:\n\n", categorized_intervals)

# %%
# Based on the tp, fp, and fn intervals, common performance metrics such as F1 score, precision,
# and recall can be calculated.
# For this purpose, the :func:`~gaitlink.utils.evaluation.precision_recall_f1_score` function can be used.
# It returns a dictionary containing the metrics for the specified categorized intervals DataFrame.
# Furthermore, we provide similar functions for other metrics such as accuracy, specificity,
# and negative predictive value.

from gaitlink.utils.evaluation import precision_recall_f1_score

prec_rec_f1_dict = precision_recall_f1_score(categorized_intervals)

print("Performance Metrics:\n\n", prec_rec_f1_dict)

# %%
# To calculate not only a specific performance metric but the whole range of possible metrics that were utilized for
# gait sequence detection in Mobilise-D, we can use the
# :func:`~gaitlink.gsd.validation.calculate_gsd_performance_metrics` function.
# It returns a DataFrame containing all metrics for the specified detected and reference gait sequences. To retrieve
# the whole range of metrics, the length and the sampling frequency of the recording are required.
# This is used to infer the number of true negative samples and derived metrics (e.g., accuracy and specificity), and
# to calculate the duration errors (in seconds), respectively.

from gaitlink.gsd.validation import calculate_gsd_performance_metrics

metrics_all = calculate_gsd_performance_metrics(
    long_trial_output.gsd_list_,
    long_trial_reference_parameters,
    sampling_rate_hz=long_trial.sampling_rate_hz,
    n_samples=long_trial.data["LowerBack"].shape[0],
)

print("Performance Metrics:\n\n", metrics_all)

# %%
# Another useful function for validation is :func:`~gaitlink.gsd.validation.find_matches_with_min_overlap`. It returns all intervals from the Python
# output that overlap with the reference gait sequences by at least a given amount.
# We can see that with an overlap threshold of 0.7 (70%), three of the five detected gait sequences are considered as
# matches with the reference gait sequences.
# The remaining ones either contain too many false positive and/or false negative samples.

from gaitlink.gsd.validation import find_matches_with_min_overlap

matches = find_matches_with_min_overlap(
    long_trial_output.gs_list_, long_trial_reference_parameters, overlap_threshold=0.7
)

print("Matches:\n\n", matches)
