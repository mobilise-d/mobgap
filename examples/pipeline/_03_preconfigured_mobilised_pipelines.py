"""
Preconfigured Mobilised Pipelines
=================================

As part of the Mobilise-D project two separate pipelines have been developed depending on the patient characteristics.
The first pipeline :class:`~mobgab.pipeline.MobilisedPipelineHealthy` (P1 in [1]_) is designed for people that likely
still have a somewhat normal gait pattern.
In Mobilise-D, this pipeline is used for healthy controls and patients with "COPD" and "CHF".
The second pipeline :class:`~mobgab.pipeline.MobilisedPipelineImpaired` (P2 in [1]_) is designed for patients with
likely significantly impaired gait patterns.
In Mobilise-D, this pipeline is used for patients with "PD", "PFF" and "MS".

In this example we will show how to use these preconfigured pipelines.
If you want to understand the details of the pipelines, please refer to the
`step-by-step example <mobilised_pipeline_step_by_step>`_.

.. [1] Kirk, C., Küderle, A., Micó-Amigo, M.E. et al. Mobilise-D insights to estimate real-world walking speed in
       multiple conditions with a wearable device. Sci Rep 14, 1754 (2024).
       https://doi.org/10.1038/s41598-024-51766-5
"""

# %%
# Data
# ----
# For this example, we will use the provided example data.
# It contains data from Lab tests from MS patients and healthy controls.
from mobgap.data import LabExampleDataset

data_ha = LabExampleDataset().get_subset(cohort="HA")
data_ha

# %%
data_ms = LabExampleDataset().get_subset(cohort="MS")
data_ms

# %%
# Mobilised Pipeline Healthy
# --------------------------
from mobgap.pipeline import MobilisedPipelineHealthy, MobilisedPipelineUniversal

pipeline_ha = MobilisedPipelineHealthy()

# %%
# We just apply the pipeline to the first long test in the data.
long_test_ha = data_ha.get_subset(test="Test11")[0]

pipeline_ha = pipeline_ha.safe_run(long_test_ha)

# %%
# Now we can access the results.
# Note, that the pipelines contain a large number of results.
# Not all of them are relevant for every use case.
# We only show the main outputs here:
#
# The main output are the aggregated parameters.
# By default, this is just a single output for each recording.
# It describes the overall statistics and aggregated parameters over all WBs.
pipeline_ha.aggregated_parameters_

# %%
# On level below, we have the WB parameters.
# This is a DataFrame with the parameters per WB.
pipeline_ha.per_wb_parameters_

# %%
# Even more granular are the stride level parameters.
# They contain only the strides that are also part of a valid WB.
pipeline_ha.per_stride_parameters_

# %%
# For other results, see the documentation of the pipeline class itself.

# %%
# Mobilised Pipeline Impaired
# ---------------------------
from mobgap.pipeline import MobilisedPipelineImpaired

pipeline_ms = MobilisedPipelineImpaired()

# %%
# We just apply the pipeline to the first long test in the data.
long_test_ms = data_ms.get_subset(test="Test11")[0]

pipeline_ms = pipeline_ms.safe_run(long_test_ms)

# %%
# Like before we can access the results.
pipeline_ms.aggregated_parameters_

# %%
pipeline_ms.per_wb_parameters_

# %%
pipeline_ms.per_stride_parameters_

# %%
# That's it.
# As you can see, it is super simple to run the preconfigured pipelines on your data, if they are structured as a valid
# gait dataset.
#
# However, when running a larger study, you might want to process all data at once.
# Then it become "inconvenient" to run the pipeline for each recording separately manually.
#
# Luckily, it is relatively easy to implement a loop that runs the pipeline for each recording.
# We can even use :class:`~mobgap.pipeline.MobilisedPipelineUniversal` to automatically process all MS participants
# with the impaired pipeline and all HA participants with the healthy pipeline.

meta_pipeline = MobilisedPipelineUniversal(
    pipelines=[
        ("healthy", MobilisedPipelineHealthy()),
        ("impaired", MobilisedPipelineImpaired()),
    ]
)

# %%
# The meta-pipeline uses the ``recommended_cohorts`` parameter of the respective pipeline to determine which pipeline to
# use.
MobilisedPipelineHealthy().recommended_cohorts

# %%
MobilisedPipelineImpaired().recommended_cohorts

# %%
# So we can simply loop over all the data and run the meta-pipeline.
# We add a little bit of logic to deal with trials that for which we might not detect a valid WB.
# Then we aggregate the results.
#
# For the ``aggreate_parameters`` we modify the index, so that we have rows with NaNs for the trials that did not have
# any valid WBs.
import pandas as pd
from tqdm.auto import tqdm

per_wb_paras = {}
aggregated_paras = {}

for trial in tqdm(LabExampleDataset()):
    pipe = meta_pipeline.clone().safe_run(trial)
    if not (per_wb := pipe.per_wb_parameters_).empty:
        per_wb_paras[trial.group_label] = per_wb
    if not (agg := pipe.aggregated_parameters_).empty:
        aggregated_paras[trial.group_label] = agg

per_wb_paras = pd.concat(per_wb_paras)
aggregated_paras = (
    pd.concat(aggregated_paras)
    .reset_index(-1, drop=True)
    .rename_axis(LabExampleDataset().index.columns)
    .reindex(pd.MultiIndex.from_tuples(LabExampleDataset().group_labels))
)

# %%
# And now we can simply access the results.
#
# First the per WB parameters.
# Each row represents a WB and the multi-index tells us from which participant and which test it is.
per_wb_paras

# %%
# And the aggregated parameters.
# Note, that many values are NaN, because only a single WB was detected per trial.
# So we can not calculate the standard deviation or other statistics.
# To learn more about what the different aggregated values mean, check :class:`~mobgap.aggregation.MobilisedAggregator`.
aggregated_paras


# Modifying Parameters
# --------------------
# Both pipelines are basically the same, but the algorithms used for certain steps are different.
# Both just reimplement :class:`~mobgap.pipeline.BaseMobilisedPipeline` with the respective algorithms as default
# parameters.
# So we can easily modify the parameters of the pipeline either using the ``set_params`` method or by passing different
# parameters/algorithms to the constructor.
#
# .. warning:: As part of Mobilise-D we only validated the pipelines with their default values in exactly the cohorts we
#              recommend them for.
#              If you change the parameters, or use them in a different cohort, we ask you to not call this approach
#              "the Mobilised Pipeline" anymore, when communicating your results.
#
# Starting simple, let's say we simply don't want to filter and aggregate the final DMOs.
# We just set the respective parameters to None.
from mobgap.pipeline import MobilisedPipelineHealthy

pipe_no_agg = MobilisedPipelineHealthy(
    dmo_thresholds=None, dmo_aggregation=None
)
pipe_no_agg.safe_run(long_test_ha)

# %%
# Now, the aggregated parameters are empty.
pipe_no_agg.aggregated_parameters_

# %%
# And the per WB parameters are still there.
pipe_no_agg.per_wb_parameters_

# %%
# If you want to change the algorithm used for a certain step, you can simply pass a different algorithm to the
# constructor.
# For example, let's say you want to use the Adaptive Ionescu GSD algorithm instead of the GSDIluz (which is the
# default for the healthy pipeline).
#
# For the sake of this example, we will also modify the default parameters of the algorithm.
from mobgap.gait_sequences import GsdAdaptiveIonescu

pipe_adaptive_gsd = MobilisedPipelineHealthy(
    gait_sequence_detection=GsdAdaptiveIonescu(min_n_steps=3)
)
pipe_adaptive_gsd.safe_run(long_test_ha)
# %%
# This works as before and all parameters of the pipeline are still available.
pipe_adaptive_gsd.aggregated_parameters_

# %%
pipe_adaptive_gsd.per_wb_parameters_

# %%
# When you are planning to modify many algorithms, we would recommend to not use the specific pipeline classes anymore,
# to avoid the association (is it really still the Healthy pipeline if you change all algorithms?).
# In this case, we recommend the un-configured :class:`~mobgap.pipeline.GenericMobilisedPipeline`.
# This class is also used as the base class for the preconfigured pipelines.
# It has no algorithms set by default, so you have to set all algorithms yourself.
# See the end of the `step-by-step example <mobilised_pipeline_step_by_step>`_ for a demonstration.
#
# If you want to reuse some of the defaults of the preconfigured pipelines, you can still use the
# ``PreconfiguredParameters``.
# For example, we could get the same pipeline as before like this:
from mobgap.pipeline import GenericMobilisedPipeline

pipe_custom = GenericMobilisedPipeline(
    **dict(
        GenericMobilisedPipeline.PredefinedParameters.regular_walking,
        gait_sequence_detection=GsdAdaptiveIonescu(min_n_steps=3),
    )
)
pipe_adaptive_gsd.safe_run(long_test_ha)
# %%
pipe_adaptive_gsd.aggregated_parameters_

# %%
pipe_adaptive_gsd.per_wb_parameters_

# %%
# On the other end, if you are only planning to change a single sub-parameter of a pipeline, it might be easier to use
# the ``set_params`` method, instead of passing all parameters to the constructor.
#
# We show the extreme example of this here, by using the Universal-Pipeline as starting point and changing the
# filter order of the pre-processing filter of the GSD algorithm of the healthy pipeline used internally in the
# MetaPipeline.
#
# .. note:: The MobilisedPipelineUniversal is a special case, as it makes use of a tpcp feature called
#           ``composite_params``.
#           This allows us to target the ``pipelines__healthy`` parameters, even tough ``pipelines`` is not an object,
#           but a list of tuples.
#           Learn more about this feature in the `tpcp documentation
#           <https://tpcp.readthedocs.io/en/latest/auto_examples/recipies/_03_composite_objects.html>`_.
#
from mobgap.pipeline import MobilisedPipelineUniversal

meta_pipeline_modified = MobilisedPipelineUniversal().set_params(
    pipelines__healthy__gait_sequence_detection__pre_filter__order=50
)

# %%
# This parameter name is a bit long, but it demonstrates that it is possible to change even deeply nested parameters.
# This might be in particular useful, when you want to run approaches like GridSearch.
#
# The algorithm works as before (note we don't expect any change in output for this parameter change).
meta_pipeline_modified.safe_run(long_test_ha)
meta_pipeline_modified.aggregated_parameters_
