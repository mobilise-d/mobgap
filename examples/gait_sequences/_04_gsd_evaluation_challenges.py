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
# We provide a default scoring function that calculates all the relevant performance metrics.
#
# Let's look at the code of it first.
from inspect import getsource

from mobgap.gait_sequences.evaluation import gsd_evaluation_scorer

print(getsource(gsd_evaluation_scorer))

# %%
# We can see that this method is relatively simple, using the gsd evaluation functions that we provide.
# So if you want to run your own scoring function, it should be straightforward to do so.
#
# Note, the :class:`~tpcp.validate.NoAgg` wrapping some of the return values.
# This is a special aggregator that tells the challenge to not try to aggregate the respective values.
# For all other values, the challenge will try average the values across all datapoints.
#
# To learn more about these special aggregators, check out the `tpcp example
# <https://tpcp.readthedocs.io/en/latest/auto_examples/validation/_03_custom_scorer.html>`_.
#
# The scoring function takes care of running the pipeline.
# So we can test the scorer, by just providing it with a pipeline and a datapoint.
#
# Note, that we remove the :class:`~tpcp.validate.NoAgg` parameters from the results, as they don't visualize well.
from pprint import pprint

single_dp_results = gsd_evaluation_scorer(pipe, long_test[0])
single_dp_results.pop("detected")
single_dp_results.pop("reference")
pprint(single_dp_results)

# %%
# The challenge will call this scoring method for each datapoint in the dataset.
# Let's test this with the `GsdEvaluation` challenge.
from mobgap.gait_sequences.evaluation import GsdEvaluation

eval_challenge = GsdEvaluation(long_test, scoring=gsd_evaluation_scorer)
# %%
# We can now run the challenge.
eval_challenge = eval_challenge.run(pipe)

# %%
# The results are stored in the `results_` attribute and contain the aggregated and the raw results per datapoint.
# To learn more about the results, check the :func:`~tpcp.validate.validate` documentation.
import pandas as pd

validate_results = pd.DataFrame(eval_challenge.results_)

# %%
# The aggregated results across all datapoints are available in all columns not starting with ``agg__``
agg_results = validate_results.filter(like="agg__")
agg_results.T

# %%
# The raw results are stored in the columns starting with ``single__``.
single_results = validate_results.filter(like="single__")

# %%
# And it is often helpful to explode them to get a better overview.
exploded_results = (
    single_results.explode(single_results.columns.to_list())
    .rename_axis("fold")
    .set_index(
        pd.MultiIndex.from_tuples(
            (dl := validate_results["data_labels"].explode().to_list()),
            names=list(dl[0]._fields),
        ),
        append=True,
    )
)
exploded_results.columns = exploded_results.columns.str.removeprefix("single__")
exploded_results.T

# %%
# The ``detected`` and ``reference`` columns in this dataframe contain the raw un-aggregated gait-sequences.
# So if we want to perform further evaluation on them (e.g. visualize them), we can use them.
raw_gs_list = pd.concat(
    exploded_results.loc[:, ["detected", "reference"]].stack().to_dict(),
    names=[*exploded_results.index.names, "system"],
).unstack("system")
raw_gs_list

# %%
# Further there are some runtime information available (i.e. when the challenge was started, and how long it took).
eval_challenge.start_datetime_, eval_challenge.end_datetime_

# %%
eval_challenge.runtime_s_


# %%
# Using :class:`~mobgap.gait_sequences.evaluation.GsdEvaluation` is great, if you are only comparing (or planning to
# compare) non-ML algorithms, or algorithms that don't require further optimization (e.g. through GridSearch).
#
# Therefore, it is generally recommended to run a cross-validation with
# :class:`~mobgap.gait_sequences.evaluation.GsdEvaluationCV`.
# This allows you to evaluate the performance of the algorithm on multiple folds of the dataset and through the use
# of :class:`~tpcp.optimize.DummyOptimize` you can also use algorithms without optimization in the same pipeline for
# comparison.
#
# Let's demonstrate the use of :class:`~mobgap.gait_sequences.evaluation.GsdEvaluationCV` on the example dataset using
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
from mobgap.gait_sequences.evaluation import GsdEvaluationCV

eval_challenge_cv = GsdEvaluationCV(
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
# As before our main results are the aggregated results.
cv_results = pd.DataFrame(eval_challenge_cv.cv_results_)
agg_results_cv = cv_results.filter(like="test__agg__")
agg_results_cv.T

# %%
# The raw results are stored in the columns starting with ``single__``.
single_results = cv_results.filter(like="test__single__")
exploded_results_cv = (
    single_results.explode(single_results.columns.to_list())
    .rename_axis("fold")
    .set_index(
        pd.MultiIndex.from_tuples(
            (dl := cv_results["test__data_labels"].explode().to_list()),
            names=list(dl[0]._fields),
        ),
        append=True,
    )
)
exploded_results_cv.columns = exploded_results_cv.columns.str.removeprefix(
    "test__single__"
)
exploded_results_cv.T

# %%
# And the raw outputs:
raw_gs_list_cv = pd.concat(
    exploded_results_cv.loc[:, ["detected", "reference"]].stack().to_dict(),
    names=[*exploded_results_cv.index.names, "system"],
).unstack("system")
raw_gs_list_cv

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
    pipe, para_grid, return_optimized="precision", scoring=gsd_evaluation_scorer
)

# %%
# The optimizer can now be used in the same CV challenge as before.
# This way we can guarantee that the same folds are used for the optimization and the evaluation and ensure the best
# possible comparison between the algorithms versions.
eval_challenge_cv = eval_challenge_cv.clone().run(optimizer)

# %%
# The results we are seeing now are generated by the internally optimized version of the algorithm.
cv_results_gs = pd.DataFrame(eval_challenge_cv.cv_results_)
agg_results_gs = cv_results.filter(like="test__agg__")

# %%
# Because we used ``cv_params={"return_optimizer": True}`` we can also access the optimizer per fold directly.
# This can be useful to get more insights into the optimization process and what the optimal parameters were.
opt_results = cv_results_gs["optimizer"]
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
