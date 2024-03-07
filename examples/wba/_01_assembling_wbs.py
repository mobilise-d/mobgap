"""

Assembling WBs
==============

An important part of the Mobilise-D pipeline is that we output WBs based on a fixed set of definitions.
These definitions are build based on a consensus process within the Mobilise-D consortium [1]_ and additional normative
data from the literature to define realistic ranges for stride and WB level parameters.

To support the process of creating WBs based on these and further custom rules, we provide a framework of dynamic rule
objects that can be adjusted and combined and then used to create WBs from a list of strides.
This works in 3 steps:

1. First, we filter the list of strides based on the stride-level rules.
2. Second, we iterate through the remaining list of strides and add them to a WB, until we reach a "Termination
   Criterion".
3. The WBs found in this way are then finally filtered based on WB-level rules.

.. [1] Kluge F, Del Din S, Cereatti A, Gaßner H, Hansen C, Helbostad JL, et al. (2021) Consensus based framework for
      digital mobility monitoring. PLoS ONE 16(8): e0256541. https://doi.org/10.1371/journal.pone.0256541
"""
# TODO: Update the example with real stride values and rules

# %%
# Step 1: Stride Selection
# ------------------------
# Stride selection is a relatively simple process.
# Each stride is defined by its start and end time and a set of parameters.
# Stride-level rules are basically just threshold rules for these parameters or the duration of the stride.
#
# Let's start by creating a list of strides with dummy values.
# We specifically create list of "left" and "right" strides and combine them at the end.
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def window(start, end, **parameter):
    parameter = parameter or {}

    parameter = {**parameter, "duration": end - start}

    return dict(id=str(uuid.uuid4())[:6], start=start, end=end, **parameter)


def naive_stride_list(start, stop, duration, foot=None, **paras):
    """A window list full of identical strides."""
    x = np.arange(start, stop + duration, duration)
    start_end = zip(x[:-1], x[1:])

    return pd.DataFrame.from_records(
        [window(start=s, end=e, foot=foot, duration=duration, **paras) for i, (s, e) in enumerate(start_end)]
    ).set_index("id")


stride_list = [
    naive_stride_list(0, 5000, 100, foot="left"),
    naive_stride_list(50, 5050, 100, foot="right"),
    naive_stride_list(5000, 6020, 60, foot="left"),
    naive_stride_list(5050, 6070, 60, foot="right"),
    naive_stride_list(6020, 8000, 90, foot="left"),
    naive_stride_list(6070, 8050, 90, foot="right"),
]

stride_list = pd.concat(stride_list).sort_values("start")
# We add some additional parameters, we can use to filter later on.
large_sl_ids = [10, 11, 12, 13, 14, 18, 19, 20, 21, 56, 90, 91, 121, 122, 176]
stride_list["stride_length"] = 1
stride_list.loc[stride_list.index[large_sl_ids], "stride_length"] = 2
stride_list

# %%
# We can filter the stride list above based on a set of rules.
# We start with some simple rules to demonstrate the process.
# All rules must be subclasses of :class:`~gaitlink.wba.BaseIntervalCriteria`.
# At the moment we have implemented :class:`~gaitlink.wba.IntervalParameterCriteria` and
# :class:`~gaitlink.wba.IntervalDurationCriteria`.
# But you could implement custom subclasses, if you see a need for it.
#
# Here we just implement a simple rule that filters strides based on their stride length for now.
# With the thresholds, all strides, we marked as "large" above should be filtered out.
from gaitlink.wba import IntervalParameterCriteria

rules = [("sl_thres", IntervalParameterCriteria("stride_length", lower_threshold=0.5, upper_threshold=1.5))]

# %%
# We can now use these rules to filter the stride list.
from gaitlink.wba import StrideSelection

ss = StrideSelection(rules)
ss.filter(stride_list)

filtered_stride_list = ss.filtered_stride_list_
filtered_stride_list

# %%
filtered_stride_list["stride_length"].unique()

# %%
# As we can see all the strides above the threshold were filtered out.
#
# Let's make this a little more complicated and add a second rule that filters strides based on their duration.
# This could either be done by a second rule targeting the parameter `duration` or by using the
# :class:`~gaitlink.wba.IntervalDurationCriteria`, which recalculates the duration of the stride based on the
# `start` and `end` columns.
# We use the latter here.
#
# The thresholds we use here should filter out all the strides from above with a duration of 60
from gaitlink.wba import IntervalDurationCriteria

rules.append(("dur_thres", IntervalDurationCriteria(lower_threshold=80, upper_threshold=120)))

ss = StrideSelection(rules)
ss.filter(stride_list)

filtered_stride_list = ss.filtered_stride_list_
filtered_stride_list

# %%
filtered_stride_list["duration"].unique()

# %%
# We can also explicitly inspect which strides are filtered out.
ss.excluded_stride_list_

# %%
# And even see which rule filtered them out.
# Note, that only the first rule that filtered the stride is shown.
ss.excluded_stride_list_.merge(ss.exclusion_reasons_, left_index=True, right_index=True)

# %%
# Step 2,3: WB Assembly
# ---------------------
# Now that we have a list of strides that we consider valid, we want to group them into WBs.
# For this we define a set of rules that define when a WB should be terminated and if a preliminary WB fulfills
# the criteria to be a valid WB.
# These rules are subclasses of :class:`~gaitlink.wba.BaseWbCriteria` and each rule can act as both a termination
# criterion and an inclusion criterion.
# Have a look at the documentation of the specific rules for more details.
# For more details on how the rules are applied, have a look at the documentation of the
# :class:`~gaitlink.wba.WbAssembly`.
#
# For this example, we use two rules:
#
# 1. :class:`~gaitlink.wba.MaxBreakCriteria`: This rule terminates a WB if the time between two strides is larger
#    than a given threshold.
#    It acts as a termination criterion.
# 2. :class:`~gaitlink.wba.NStridesCriteria`: This rule excludes preliminary WBs that have less than a given number of
#    strides.
#    It acts as an inclusion criterion.
from gaitlink.wba import MaxBreakCriteria, NStridesCriteria, WbAssembly

rules = [
    ("max_break", MaxBreakCriteria(max_break=10, remove_last_ic="per_foot", consider_end_as_break=True)),
    ("min_strides", NStridesCriteria(min_strides=5)),
]

wb_assembly = WbAssembly(rules)
# Note, that we use the filtered stride list from above.
wb_assembly.assemble(filtered_stride_list)

# %%
# The wb_assembly object now contains a grouping of each stride to a WB.
# Depending on how we want to further process the data, we can either use the `wbs_` attribute, which is a dictionary
# mapping a WB id to a list of strides, or the `annotated_stride_list_` attribute, which is a copy of the stride list
# containing an additional column `wb_id` that contains the id of the WB the stride belongs to.
# Note, that both outputs only contain the strides that belong to a final WB.
# Strides belonging to a WB that was ultimately filtered out, or never belonged to a WB in the first place are not
# included.
# They can be accessed via the `excluded_stride_list_` and `excluded_wbs` attributes.
wb_assembly.annotated_stride_list_

# %%
print(f"The method identified {len(wb_assembly.wbs_)} WBs.")

# %%
# We can also get an idea of how the final results looks, by plotting the wba outputs.
from gaitlink.wba.plot import plot_wba_results

plot_wba_results(wb_assembly, stride_selection=ss)
plt.show()
