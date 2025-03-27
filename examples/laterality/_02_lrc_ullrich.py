"""
Ullrich L/R Classifier
======================

The Ullrich L/R classifier is a general approach of differentiating left from right foot contacts using signal features
extracted from the gyroscopic data of a single IMU sensor placed on the lower back.
The feature vectors at the timepoints of the pre-detected initial contacts are then used in a typical binary
classification pipeline to predict the left/right label of each initial contact.

This example shows how to use the algorithm and how to train your own classification model on custom data.

"""

from mobgap.data import LabExampleDataset

# %%
# Loading some example data
# -------------------------
# .. note :: More infos about data loading can be found in the :ref:`data loading example <data_loading_example>`.
#
# We load example data from the lab dataset together with the INDIP reference system.
# We will use the INDIP "InitialContact_Event" output as ground truth.
#
# We only use the data from the "simulated daily living" activity test from a single participant.
#
# Like most algorithms, the algorithm requires the data to be in body frame coordinates.
# As we know the sensor was well aligned, we can just use ``to_body_frame`` to transform the data.
from mobgap.utils.conversions import to_body_frame

example_data = LabExampleDataset(
    reference_system="INDIP", reference_para_level="wb"
)
single_test = example_data.get_subset(
    cohort="MS", participant_id="001", test="Test11", trial="Trial1"
)

imu_data = to_body_frame(single_test.data_ss)
reference_wbs = single_test.reference_parameters_.wb_list

sampling_rate_hz = single_test.sampling_rate_hz
ref_ics = single_test.reference_parameters_.ic_list
ref_ics_rel_to_gs = single_test.reference_parameters_relative_to_wb_.ic_list

# %%
# Applying the algorithm using reference ICs
# ------------------------------------------
# We use algorithm to detect the laterality of the initial contacts.
# For this we need the IMU data and the indices of the initial contacts per GS.
# To focus this example on the L/R detection, we use the reference ICs from the INDIP system as input.
# In a real application, we would use the output of the IC-detectors as input.
#
# First, we need to set up an instance of our algorithm.
# For ``LrcUllrich`` we provide a pre-trained model, which we can use to predict the L/R labels.
# They are all trained on the MS-Project (University of Sheffield) dataset, just on different sub cohorts and can
# be accessed using ``LrcUllrich.PredefinedParameters``.
# We will use the model trained on all participants of the MS-Project dataset.
from mobgap.laterality import LrcUllrich

algo = LrcUllrich(**LrcUllrich.PredefinedParameters.msproject_all)

# %%
# As we want to apply the algorithm to each gait sequence/WB individually, use the `GsIterator` to iterate over the
# reference wbs and apply the algorithm to each wb.
# Note, that we use the ``ic_list`` result key, as the output of all L/R detectors is identical to the output of the
# IC-detectors, but with an additional ``lr_label`` column.
from mobgap.pipeline import GsIterator

iterator = GsIterator()

for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
    result.ic_list = algo.predict(
        data,
        ic_list=ref_ics_rel_to_gs.loc[gs.id].drop("lr_label", axis=1),
        sampling_rate_hz=sampling_rate_hz,
    ).ic_lr_list_

detected_ics = iterator.results_.ic_list
detected_ics.assign(ref_lr_label=ref_ics.lr_label)

# %%
# The output that we get provides us with an `lr_label` for each initial contact.
#
# Training a custom model
# -----------------------
# As ``LrcUllrich`` is machine-learning based, we provide the option to train a custom model on your own data.
# We even allow you to complete customize the classifier pipeline to test out different approaches.
#
# Here we will show how to train a custom model on the example data using the "low level" training interface.
# However, for most usecases (e.g. training with Hyperparatuning/evaulation with cross_validation/...) you will
# want to use the higher level Pipeline interface.
# A full example on how to use this is shown in the :ref:`evaluation example <lrc_evaluation>`.
#
# The low level interface involves directly calling the ``self_optimize`` method of ``LrcUllrich``.
# It takes a series of data sequences, their corresponding ICs and ground truth labels.
#
# But first, we need to define an untrained sklearn ML pipeline to provide to the algorithm for tuning.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

clf_pipe = Pipeline([("scaler", MinMaxScaler()), ("clf", SVC())])
algo = LrcUllrich(clf_pipe=clf_pipe)

# %%
# Then we need to prepare the data.
# We will extract the IMU data the ICs and the ground truth labels from the example data used above.
per_gs_data = []
per_gs_ic = []
per_gs_ic_lr = []

for (gs, data), _ in iterator.iterate(imu_data, reference_wbs):
    per_gs_data.append(data)
    ref_ics = ref_ics_rel_to_gs.loc[gs.id]
    per_gs_ic.append(ref_ics.drop("lr_label", axis=1))
    per_gs_ic_lr.append(ref_ics)

# %%
# We will use all sequences but the last as trainings data.
algo = algo.self_optimize(
    per_gs_data[:-1],
    per_gs_ic[:-1],
    per_gs_ic_lr[:-1],
    sampling_rate_hz=sampling_rate_hz,
)

# %%
# We can now use our trained model and make predictions on the sequence we did not train on.
# We will use the last sequence for this.
predictions = algo.predict(
    per_gs_data[-1], ic_list=per_gs_ic[-1], sampling_rate_hz=sampling_rate_hz
).ic_lr_list_.assign(ref_lr_label=per_gs_ic_lr[-1]["lr_label"])
predictions

# %%
# Note, that we don't expect particularly good performance, as we trained on very little data.
# But, because the data all from the same participant and recorded in a controlled lab environment, we can see that
# most predictions are correct.
#
# If you want to learn about evaluating and optimizing the algorithm, please refer to the
# :ref:`evaluation example <lrc_evaluation>`.
