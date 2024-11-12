"""
GSD Evaluation Challenges
-------------------------
The :ref:`gsd_evaluation` example demonstrates how to evaluate the performance of a GSD algorithm on a single datapoint
 and explains the individual performance metrics that are calculated.

With that you could set up a custom evaluation pipeline to run and then score the output of a GSD algorithm
multiple datapoints and then aggregate the results.
To make this process easier, we set up opinionated evaluation challenges that can be used to quickly perform the same
evaluation with multiple algorithms and datasets.

Below, we will show how to use them on the example dataset.
"""
# TODO: Update based on new Scorer API

# %%
# Dataset
# -------
# To use the challenges, we need to dataset with reference information in the expected format.
# We will use the :class:`~mobgap.data.LabExampleDataset` for this purpose.
from mobgap.data import LabExampleDataset

long_test = LabExampleDataset(reference_system="INDIP").get_subset(
    test="Test11"
)

# %%
# Algorithm
# ---------
# Next we need to create an instance of a valid GSD algorithm.
from mobgap.gait_sequences import GsdIluz

algo = GsdIluz()

# %%
# This algorithm needs to be wrapped in a :class:`~mobgap.gait_sequences.pipeline.GsdEmulationPipeline` to be used in
# the challenges.
# This pipeline takes care of extracting the correct data from the dataset and running the algorithm on it.
from mobgap.gait_sequences.pipeline import GsdEmulationPipeline

pipe = GsdEmulationPipeline(algo)

# %%
# Let's demonstrate that quickly on a single datapoint.
pipe_with_results = pipe.clone().run(long_test[0])
pipe_with_results.gs_list_

# %%
# Evaluation Challenge
# --------------------
# This pipeline can now be used as part of an evaluation challenge.
# An evaluation challenge takes care of two things:
#
# - Running the pipeline on multiple datapoints
# - Scoring the results per datapoint and then aggregating the results
#
# We provide two challenges:
#
# - :class:`~mobgap.gait_sequences.evaluation.GsdEvaluation`: This challenge simply runs the pipeline on all datapoints and then scores the results.
# - :class:`~mobgap.gait_sequences.evaluation.GsdEvaluationCV`: This challenge runs a cross-validation on the dataset and then scores the results per fold.
#
# Before we run the entire pipeline, let's look at the scoring.
# Scoring is built based on tpcp's validation framework.
# As we have relativly complex scoring, scoring is split across two functions:
#
# - :func:`~mobgap.gait_sequences.evaluation.gsd_per_datapoint_score`: Run and score a single datapoint
# - :func:`~mobgap.gait_sequences.evaluation.gsd_final_agg`: Perform final aggreagtion and scoring based on the results
#                                                            per datapoint.
#
# Let's look at the code of it first.
from inspect import getsource

from mobgap.gait_sequences.evaluation import (
    gsd_final_agg,
    gsd_per_datapoint_score,
)

print(getsource(gsd_per_datapoint_score))

# %%
print(getsource(gsd_final_agg))

# %%
# We can see that these method is relatively simple, using the lower level gsd evaluation functions that we provide.
# `gsd_per_datapoint_score` calculates the raw results and all scores that can be calculated per datapoint.
# `gsd_final_agg` handles the calculation of all scores, that require the raw results from all datapoints at once.
# The remaining aggregation is handled by the :class:`~tpcp.validate.Scorer` class (see below).
# So if you want to run your own scoring function, it should be straightforward to do so.
#
# Note, the :func:`~tpcp.validate.no_agg` wrapping some of the return values.
# This is a special aggregator that tells the challenge to not try to aggregate the respective values.
# For all other values, the challenge will try average the values across all datapoints.
#
# To learn more about these special aggregators, check out the `tpcp example
# <https://tpcp.readthedocs.io/en/latest/auto_examples/validation/_03_custom_scorer.html>`_.
#
# The scoring function takes care of running the pipeline.
# So we can test the scorer, by just providing it with a pipeline and a datapoint.
from pprint import pprint

single_dp_results = gsd_per_datapoint_score(pipe, long_test[0])
single_dp_results.pop("detected")
single_dp_results.pop("reference")
pprint(single_dp_results)

# %%
# To use the two functions with a challenge, we need to wrap them into a :class:`~tpcp.validate.Scorer` instance.
from tpcp.validate import Scorer

gsd_evaluation_scorer = Scorer(
    gsd_per_datapoint_score, final_aggregator=gsd_final_agg
)

# %%
# The challenge will call this scorer for each group in the dataset.
# The scorer itself will then call `gsd_per_datapoint_score` for each datapoint and then `gsd_final_agg` with the
# combined results.
#
# For these two default scoring functions, we also provide the scorer directly, so that you don't have to construct it
# yourself.
# However, in case you want to modify the scoring functions, you can do so by creating your own scorer.
# We will continue to use the default scorer for the challenges.
from mobgap.gait_sequences.evaluation import gsd_score

gsd_evaluation_scorer = gsd_score

# %%
# Let's put everything together and run the challenge.
from mobgap.utils.evaluation import Evaluation

eval_challenge = Evaluation(long_test, scoring=gsd_evaluation_scorer)

# %%
# We can now run the challenge.
eval_challenge = eval_challenge.run(pipe)

# %%
# The results are stored in the `results_` attribute and contain the aggregated and the raw results per datapoint.
# To learn more about the results, check the :func:`~tpcp.validate.validate` documentation.
#
# Note, that we remove the :class:`~tpcp.validate.no_agg` parameters from the results, as they don't visualize well.
import pandas as pd

validate_results = pd.DataFrame(eval_challenge.results_)
validate_results
# %%
# As you can see, this is a very messy dataframe with a lot of information.
# To make this easier to digest, the evaluation object has methods for extracting the different groups of information.
# The first group is the aggregated results, which represent only a "single value" over the entire dataset.
agg_results = eval_challenge.get_aggregated_results_as_df()
agg_results.T

# %%
# You might have seen, that many metrics appear twice, once with a `combined__` prefix and once without.
# These represent two different things.
# If you check in the source code of the scorer above, the metric without prefix is calculated per datapoint and then
# averaged.
# The metric with the prefix is calculated over the raw detected gait sequences of all datapoints combined.
# Effectively, this is equivalent to different "weightings".
# In the aggregated results without prefix, each recording has the same weight, independent of its length.
# In the second case, each individual imu-sample has the same weight.
# It does not matter, in which recording this sample was classified correctly or not, it has the same impact on the
# combined metric.
#
# Both approaches are valid, but you should be aware of the differences when comparing algorithms.
# The way how you aggregate here, can have a big impact on the results.
combined_metrics = agg_results.filter(like="combined__").rename(
    columns=lambda x: x.replace("combined__", "")
)
combined_vs_per_datapoint = pd.concat(
    {
        "combined": combined_metrics,
        "per_datapoint": agg_results[combined_metrics.columns],
    },
    axis=0,
)
combined_vs_per_datapoint.reset_index(level=-1, drop=True).T
# %%
# The "single" results represent the values per datapoint.
single_results = eval_challenge.get_single_results_as_df()
single_results.T

# %%
# And finally, we had a couple "raw" results in the scoring, that we passed through without calculating any error
# metrics.
# These are available as a dictionary of raw results.
raw_results = eval_challenge.get_raw_results()
list(raw_results.keys())
# %%
raw_results["detected"]

# %%
raw_results["reference"]

# %%
# Further, there are some runtime information available (i.e. when the challenge was started, and how long it took).
eval_challenge.perf_["start_datetime"], eval_challenge.perf_["end_datetime"]

# %%
eval_challenge.perf_["runtime_s"]


# %%
# Using :class:`~mobgap.utils.evaluation.Evaluation` is great, if you are only comparing (or planning to
# compare) non-ML algorithms, or algorithms that don't require further optimization (e.g. through GridSearch).
#
# Therefore, it is generally recommended to run a cross-validation with
# :class:`~mobgap.utils.evaluation.EvaluationCV`.
# This allows you to evaluate the performance of the algorithm on multiple folds of the dataset and through the use
# of :class:`~tpcp.optimize.DummyOptimize` you can also use algorithms without optimization in the same pipeline for
# comparison.
#
# Let's demonstrate the use of :class:`~mobgap.utils.evaluation.GsdEvaluationCV` on the example dataset using
# the same algorithm once with and once without GridSearch.
#
# For the CV-based challenge, we need to set up a cross-validation.
# As we only have 3 datapoints here, we will use a 3-fold cross-validation without grouping or stratification.
# In a real-world scenario, you would use a more sophisticated cross-validation strategy.
# You can learn more about cross-validation in the `tpcp example
# <https://tpcp.readthedocs.io/en/latest/auto_examples/validation/_04_advanced_cross_validation.html>`_.
#
# Further, to speed things up, we are going to use multi-processing.
# We can configure this using the ``n_jobs`` parameter that we pass to the internal
# :func:`~tpcp.validate.cross_validate` function via the ``cv_params`` parameters
from mobgap.utils.evaluation import EvaluationCV

eval_challenge_cv = EvaluationCV(
    long_test,
    cv_iterator=3,
    scoring=gsd_evaluation_scorer,
    cv_params={"n_jobs": 2, "return_optimizer": True},
)

# %%
# To use our pipeline from above, we need to wrap it in a :class:`~tpcp.optimize.DummyOptimize` instance.
# This will basically skip any optimization on the train set and just apply the pipeline to the test set.
from tpcp.optimize import DummyOptimize

eval_challenge_cv = eval_challenge_cv.run(
    DummyOptimize(pipe, ignore_potential_user_error_warning=True)
)

# %%
# The results now are a little bit more complex, as they contain the results for each fold.
# In addition, we have information for the train and the test set.
# The test set results, are what we are usually looking for.
# The train set results, are only calculated when providing the ``return_train_score`` parameter to the ``cv_params``.
#
# As before all results are stored in the `results_` attribute, but it is usually recommended to use the helper methods
# to access the data.
#
# Note, that compared to the results above, we now have mutliple CV folds and the aggregated results present one value
# per fold.
# These parameters could be further aggregated, e.g. by calculating the mean of these values over all folds.
agg_results_cv = eval_challenge_cv.get_aggregated_results_as_df()
agg_results_cv.T

# %%
# The single results contain the CV fold as an additional index.
# Otherwise, the output is identical to before.
# Note, that if you use anything else then a KFold, splitter, you might have some datapoints duplicated across folds.
single_results_cv = eval_challenge_cv.get_single_results_as_df()
single_results_cv

# %%
# And the raw outputs:
raw_results_cv = eval_challenge_cv.get_raw_results()
raw_results_cv["detected"]

# %%
# If we compare these results to the ones from the non-CV challenge, we can see that "single" results are identical,
# just that they were called in multiple folds.
# This is expected, as we used :class:`~tpcp.optimize.DummyOptimize` and thus didn't optimize the algorithm.
#
# Let's try a :class:`~tpcp.optimize.GridSearch` on the algorithm to see how the results change.
# For the gridsearch, we will re-use the same scoring function as before, but we need to specify, which scoring result
# we want to optimize for.
from sklearn.model_selection import ParameterGrid
from tpcp.optimize import GridSearch

para_grid = ParameterGrid({"algo__window_length_s": [2, 3, 4]})
optimizer = GridSearch(
    pipe, para_grid, scoring=gsd_evaluation_scorer, return_optimized="precision"
)

# %%
# The optimizer can now be used in the same CV challenge as before.
# This way we can guarantee that the same folds are used for the optimization and the evaluation and ensure the best
# possible comparison between the algorithms versions.
eval_challenge_gs = eval_challenge_cv.clone().run(optimizer)

# %%
# The results we are seeing now are generated by the internally optimized version of the algorithm.
agg_results_cv = eval_challenge_gs.get_aggregated_results_as_df()
agg_results_cv.T

# %%
# Because we used ``cv_params={"return_optimizer": True}`` we can also access the optimizer per fold directly from
# the ``results_`` attribute.`
# This can be useful to get more insights into the optimization process and what the optimal parameters were.
opt_results = pd.Series(eval_challenge_gs.results_["optimizer"])
opt_results

# %%
# We can get the best parameters per fold by directly interacting with the optimizer instances.
best_params = opt_results.apply(lambda x: pd.Series(x.best_params_))
best_params

# %%
# Or we can go much deeper, by getting all information about the optimization process.
# Let's just look at the keys of the information that is available.
all_opti_results_fold0 = pd.DataFrame(opt_results.loc[0].gs_results_)
all_opti_results_fold0.columns.to_list()

# %%
# With that, we hope it becomes clear, how these challenges can be extremely valuable, when benchmarking algorithms
# across datasets.
# To see how we evaluate the performance of the algorithms available in mobgap, check out the other gsd evaluation
# examples.
