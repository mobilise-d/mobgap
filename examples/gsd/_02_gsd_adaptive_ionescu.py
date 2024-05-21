r"""
.. _gsd_pi:

GSD Paraschiv-Ionescu
========

This example shows how to use the GSD Paraschiv-Ionescu algorithm and some examples on how the results compare to the original
matlab implementation.

We start by defining some helpers for plotting and loading the data.
You can skip them for now and jump directly to "Performance on a single lab trial", if you just want to see how to
apply the algorithm.
"""

# Plotting Helper
# ---------------
# We define a helper function to plot the results of the algorithm.
# Just ignore this function for now.
import json

import matplotlib.pyplot as plt
import pandas as pd

from mobgap import PACKAGE_ROOT


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


from mobgap.data import LabExampleDataset

lab_example_data = LabExampleDataset(reference_system="INDIP")


def load_matlab_output(datapoint):
    p = datapoint.group_label
    with (
        PACKAGE_ROOT.parent
        / f"example_data/original_results/gsd_adaptive_ionescu/lab/{p.cohort}/{p.participant_id}/GSDB_Output.json"
    ).open() as f:
        original_results = json.load(f)["GSDB_Output"][p.time_measure][p.test][p.trial]["SU"]["LowerBack"]["GSD"]

    if not isinstance(original_results, list):
        original_results = [original_results]
    return (
        (
            pd.DataFrame.from_records(original_results).rename({"Start": "start", "End": "end"}, axis=1)[
                ["start", "end"]
            ]
            * datapoint.sampling_rate_hz
        )
        .round()
        .astype("int64")
    )


# %%
# Performance on a single lab trial
# ---------------------------------
# Below we apply the algorithm to a lab trail, where we only expect a single gait sequence.
from mobgap.gsd import GsdAdaptiveIonescu

short_trial = lab_example_data.get_subset(cohort="MS", participant_id="001", test="Test5", trial="Trial2")
short_trial_matlab_output = load_matlab_output(short_trial)
short_trial_reference_parameters = short_trial.reference_parameters_.wb_list

short_trial_output = GsdAdaptiveIonescu().detect(short_trial.data_ss, sampling_rate_hz=short_trial.sampling_rate_hz)

print("Reference Parameters:\n\n", short_trial_reference_parameters)
print("\nMatlab Output:\n\n", short_trial_matlab_output)
print("\nPython Output:\n\n", short_trial_output.gs_list_)
# %%
# When we plot the output, we can see that the python version is a little more sensitive than the matlab version.
# It includes a section of the signal before the region classified as WB by the reference system.
# Both algorithm implementations produce a gait sequence that extends beyond the end of the reference system.

fig, ax = plot_gsd_outputs(
    short_trial.data_ss,
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

long_trial_output = GsdAdaptiveIonescu().detect(long_trial.data_ss, sampling_rate_hz=long_trial.sampling_rate_hz)

print("Reference Parameters:\n\n", long_trial_reference_parameters)
print("\nMatlab Output:\n\n", long_trial_matlab_output)
print("\nPython Output:\n\n", long_trial_output.gs_list_)

# %%
# When we plot the output, we can see again that the python version is more sensitive.
# It detects longer gait sequences and even one entire gait sequence that is not detected by the matlab version.

fig, _ = plot_gsd_outputs(
    long_trial.data_ss,
    reference=long_trial_reference_parameters,
    matlab=long_trial_matlab_output,
    python=long_trial_output.gs_list_,
)
fig.show()

# %%
# Evaluation of the algorithm against a reference
# --------------------------------------------------
# To quantify how the Python output compares to the reference labels, we are providing a range of evaluation functions.
# See the :ref:`example on GSD evaluation <gsd_evaluation>` for more details.
