r"""

.. _lrC_evaluation:

LRD Evaluation
==============

This example demonstrates how to evaluate an LRC algorithm.
As left-right classification, is a balanced binary classification problem, we can apply simple metrics like accuracy to
evaluate the performance of the algorithm.
"""

import pandas as pd

from mobgap.data import LabExampleDataset
from mobgap.lrc import LrcUllrich
from mobgap.pipeline import GsIterator

# %%
# Loading some example data
# --------------------------
# First, we load some example data and apply the LrcUllrich algorithm with its default pre-trained model to it.
# We use the reference initial contacts as input for the algorithm so that we can focus on the evaluation of the
# L/R classification independently of the detection of the initial contacts.
# However, you can use any other algorithm as well.


def load_data():
    lab_example_data = LabExampleDataset(reference_system="INDIP")
    single_test = lab_example_data.get_subset(cohort="MS", participant_id="001", test="Test11", trial="Trial1")
    return single_test


def calculate_output(single_test_data):
    """Calculate the GSD Iluz output per WB."""
    iterator = GsIterator()
    ref_paras = single_test_data.reference_parameters_relative_to_wb_

    for (gs, data), r in iterator.iterate(single_test_data.data_ss, ref_paras.wb_list):
        ref_ics = ref_paras.ic_list.loc[gs.id]
        r.ic_list = LrcUllrich().predict(data, ref_ics, sampling_rate_hz=single_test_data.sampling_rate_hz).ic_lr_list_

    return iterator.results_.ic_list


def load_reference(single_test_data):
    """Load the reference gait sequences from the test data."""
    ref_gsd = single_test_data.reference_parameters_.ic_list
    return ref_gsd


test_data = load_data()
calculated_ic_lr_list = calculate_output(test_data)
reference_ic_lr_list = load_reference(test_data)


# %%
# We can see that the calculated and the reference ic_list have the same structure with the ``lr_label`` column
# providing the detected label per initial contact.
calculated_ic_lr_list

# %%
reference_ic_lr_list

# %%
# Calculating evaluation metrics
# ------------------------------
# Using the two ``lr_label`` columns, we can calculate the accuracy of the L/R detection using the simple metric
# functions from sklearn.
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(reference_ic_lr_list["lr_label"], calculated_ic_lr_list["lr_label"])
accuracy

# %%
# Similarly, we could create a confusion matrix to get more insights into the performance of the algorithm.
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay.from_predictions(reference_ic_lr_list["lr_label"], calculated_ic_lr_list["lr_label"])
disp.figure_.show()

# %%
# Running a full evaluation pipeline
# ----------------------------------
# Instead of manually evaluating and investigating the performance of an algorithm on a single piece of data, we
# often want to run a full evaluation on an entire dataset.
# This can be done using the :class:`~mobgap.lrc.base.LrdPipeline` class and some ``tpcp`` functions.
#
# But let's start with selecting some data.
# We want to use all the simulated real-world walking data from the INDIP reference system (Test11).
simulated_real_world_walking = LabExampleDataset(reference_system="INDIP").get_subset(test="Test11")

simulated_real_world_walking

# %%
# Now we can create a pipeline instance and directly run it on of the datapoints of the dataset.
from mobgap.lrc.pipeline import LrcEmulationPipeline

pipeline = LrcEmulationPipeline(LrcUllrich())

pipeline.safe_run(simulated_real_world_walking[0]).ic_lr_list_

# %%
# This is exactly what we did before, just on a pipeline level, without manually extracting the data from the dataset.
# To now actually run a validation, we need to iterate over all datapoints and calculate the accuracy for each of them.
# This can be done using the :func:`~tpcp.validate.validate` function.
#
# Note, that the ``LrdPipeline`` class already has a ``score`` method that returns the accuracy.
# This is used by default, but you could supply your own scoring method as well.
from tpcp.validate import validate

evaluation_results_with_opti = pd.DataFrame(validate(pipeline, simulated_real_world_walking))
evaluation_results_with_opti.drop(["single_raw_results"], axis=1).T

# %%
# The accuracy provided is the mean accuracy over all datapoints.
# The accuracy per datapoint can be found in the ``single_accuracy`` column.
#
# In addition to the metrics, we also provide the raw results for each datapoint in the ``single_raw_results`` column.
# This could be used for further analysis.
# For example to calculate the confusion matrix over all ICs of all datapoints.
raw_results = pd.concat(
    evaluation_results_with_opti["single_raw_results"][0], keys=evaluation_results_with_opti["data_labels"][0], axis=0
)

raw_results.head()

# %%
# The confusion matrix can be calculated using the same functions as before.
disp = ConfusionMatrixDisplay.from_predictions(raw_results["ref_lr_label"], raw_results["lr_label"])
disp.figure_.show()

# %%
# If you want to calculate additional metrics, you can either create a custom score function or subclass the pipeline
# and overwrite the score function.
#
# Parameter Optimization and Model Training
# -----------------------------------------
# Simply applying an algorithm for evaluation is one thing, but often we want to optimize the parameters of the
# algorithm, train internal models, or both and evalute the performance of this optimization approach and not just a
# fixed algorithm/model.
#
# In this case, we need to create a train test split on the dataset and to ensure we have independent data for the
# optimization.
# In general, we would recommend using a cross-validation approach.
# This can be done using the :func:`~tpcp.validate.cross_validate` function.
#
# In the example below, we show the "most complicated" case, where we retrain the internal model of the ``LrcUllrich``
# algorithm and optimize one of the Hyperparmeters of the internal SVM.
# As we retrain the model and optimize hyperparameters, we need to use a :class:`~tpcp.optimize.GridSearchCV` nested
# within the cross-validation loop.
#
# Let's set this up first.
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from tpcp.optimize import GridSearchCV

# %%
# We initialize the pipeline with an untrained model and an untrained scaler as a new pipeline.
clf_pipeline = Pipeline([("scaler", MinMaxScaler()), ("clf", SVC(kernel="linear"))])
pipeline = LrcEmulationPipeline(LrcUllrich(clf_pipe=clf_pipeline))

# %%
# Then we can create a parameter Grid for the gridsearch.
# Note, that we use ``__`` to set nested parameters.
para_grid = ParameterGrid({"algo__clf_pipe__clf__C": [0.1, 1.0, 10.0]})

# %%
# Then we path the pipeline to the optimizer.
# We only select a 2-fold cross-validation for this example, as we will only have 2 datapoints per train set and we want
# to minimize run time for this example.
optimizer = GridSearchCV(pipeline, para_grid, return_optimized="accuracy", cv=2)

# %%
# Let's test the optimizer first on a manual train set.
optimizer.optimize(simulated_real_world_walking[:2])

# %%
# We can inspect the results:
results = pd.DataFrame(optimizer.cv_results_)
results.loc[:, ~results.columns.str.endswith("raw_results")].T

# %%
# And apply/score the best performing and retrained model directly on the test set.
optimizer.score(simulated_real_world_walking[2])["accuracy"]

# %%
# Let's run everything combined with the external cross-validate to actually validate our optimization approach.
from tpcp.validate import cross_validate

evaluation_results_with_opti = pd.DataFrame(cross_validate(optimizer, simulated_real_world_walking, cv=3))
evaluation_results_with_opti.loc[:, ~evaluation_results_with_opti.columns.str.endswith("raw_results")].T

# %%
# We can compare these results with the performance of the pre-trained model that was not optimized for the given
# dataset, by using :class:`~tpcp.optimize.DummyOptimize`, to run a cross-validation, but without any optimization.
# We simply evaluate the pre-trained model on exactly the same test sets as the optimized model.
from tpcp.optimize import DummyOptimize

optimizer = DummyOptimize(LrcEmulationPipeline(LrcUllrich()))

evaluation_results_pre_trained = pd.DataFrame(cross_validate(optimizer, simulated_real_world_walking, cv=3))
evaluation_results_pre_trained.loc[:, ~evaluation_results_pre_trained.columns.str.endswith("raw_results")].T

# %%
# Note that using only so little data is not a good idea in practice.
# There are many parameters, that you should tweak to make this a robust validation.
# However, this example should provide a good starting point for your own experiments.
