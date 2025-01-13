"""
Mobilise-D TVS Dataset
======================

As part of the Mobilise-D technical validation study an extensive dataset containing 115 participants
(5 different indications + healthy adults) wearing a lower back sensor in the lab and during a 2.5 hour free-living
period was collected.
In the lab all trials were recorded with a synchronized motion capture system and the multi-modal wearable INDIP system
as reference.
During the 2.5 hour free-living period only the INDIP system was used as reference [1]_.

With that this dataset is one of the only datasets that contains data from patients with multiple indications AND
high-granular reference information from both lab and free-living settings.
This makes it a valuable resource for the development and validation of algorithms for lower back sensor data.
The dataset was already used extensivly to benchmark the Mobilise-D algorithms individually [2]_ and in
combination as a pipeline [3]_.

The dataset is published on Zenodo and can be accessed here: https://zenodo.org/records/13899386

The recommended way to work with the dataset is to use this library and the provided classes to load the data.
This will ensure that the data is loaded in a consistent way and that the data is correctly preprocessed.

This example demonstrate how to do this.

.. warning:: This example only shows the code, but not the output of the code, as this requires the dataset to be
   available, when this website is created.
   We highly recommend to run this code on your local machine to see the output.
   For this, replace the "Environmental Variable" loaded below with the actual path to the dataset on your local machine.

.. [1] Salis F, Bertuletti S, Bonci T, Caruso M, Scott K, Alcock L, Buckley E, Gazit E, Hansen , Schwickert L,
       Aminian K, Becker C, Brown P, Carsin A, Caulfield B, Chiari L, D’Ascanio I, Del Din S, Eskofier B,
       Garcia-Aymerich J, Hausdorff J, Hume E, Kirk C, Kluge F, Koch S, Kuederle A, Maetzler W, Micó-Amigo E, Mueller A,
       Neatrour I, Paraschiv-Ionescu A, Palmerini L, Yarnall A, Rochester L, Sharrack B, Singleton D, Vereijken B,
       Vogiatzis I, Della Croce U, Mazzà C, Cereatti A. A multi-sensor wearable system for the assessment of diseased
       gait in real-world conditions. Frontiers in Bioengineering and Biotechnology. 2023;11.
       doi: 10.3389/fbioe.2023.1143248.
.. [2] Micó-Amigo ME, Bonci T, Paraschiv-Ionescu A, Ullrich M, Kirk C, Soltani A, Küderle A, Gazit E, Salis F, Alcock L,
       Aminian K, Becker C, Bertuletti S, Brown P, Buckley E, Cantu A, Carsin AE, Caruso M, Caulfield B, Cereatti A,
       Chiari L, D'Ascanio I, Eskofier B, Fernstad S, Froehlich M, Garcia-Aymerich J, Hansen C, Hausdorff JM, Hiden H,
       Hume E, Keogh A, Kluge F, Koch S, Maetzler W, Megaritis D, Mueller A, Niessen M, Palmerini L, Schwickert L,
       Scott K, Sharrack B, Sillén H, Singleton D, Vereijken B, Vogiatzis I, Yarnall AJ, Rochester L, Mazzà C,
       Del Din S; Mobilise-D consortium. Assessing real-world gait with digital technology? Validation, insights and
       recommendations from the Mobilise-D consortium. J Neuroeng Rehabil. 2023 Jun 14;20(1):78.
       doi: 10.1186/s12984-023-01198-5. Erratum in: J Neuroeng Rehabil. 2024 May 3;21(1):71.
       doi: 10.1186/s12984-024-01361-6. PMID: 37316858; PMCID: PMC10265910.
.. [3] Kirk C, Küderle A, Micó-Amigo ME, Bonci T, Paraschiv-Ionescu A, Ullrich M, Soltani A, Gazit E, Salis F, Alcock L,
       Aminian K, Becker C, Bertuletti S, Brown P, Buckley E, Cantu A, Carsin A, Caruso M, Caulfield B, Cereatti A,
       Chiari L, D’Ascanio I, Garcia-Aymerich J, Hansen C, M. Hausdorff J, Hiden H, Hume E, Keogh A, Kluge F, Koch S,
       Maetzler W, Megaritis D, Mueller A, Niessen M, Palmerini L, Schwickert L, Scott K, Sharrack B, Sillén H,
       Singleton D, Vereijken B, Vogiatzis I, J. Yarnall A, Rochester L, Mazzà C, M. Eskofier B, Del Din S, Mobilise-D
       consortium. Mobilise-D insights to estimate real-world walking speed in multiple conditions with a wearable
       device. Sci Rep 14. 2024 Jan 19;1754. doi: https://doi.org/10.1038/s41598-024-51766-5
"""

# %%
# Dataset Path
# ------------
# For this example we get the path from an environmental variable, so that we don't have to hardcode the path here.
# When you run this on your local machine, you can set the environmental variable to the path of the dataset or just
# replace the path in the code below.
from pathlib import Path

from mobgap.utils.misc import get_env_var

dataset_path = Path(get_env_var("MOBGAP_TVS_DATASET_PATH").strip('"'))

# %%
# Load the Dataset
# ----------------
# We use the `TVSLabDataset` class to load the lab data and show most of the possible interactions.
# The `TVSFreeLivingDataset` can be used in the same way to load the free-living data.
#
# .. note:: Creating the dataset and selecting values from the index, does not actually load the data into RAM.
#           The data is only loaded once the actual (meta)data attributes are accessed.
#           Even then, data is only loaded per recording and not all at once.
#
# .. note:: Loading the data from the data.mat files can still be slow. If you're working with the TVS dataset a lot
#           (and have enough disk space available), it is highly recommended to use a diskcache by passing a joblib
#           Memory instance to the ``memory`` parameter of the dataset class.
#
from mobgap.data import TVSLabDataset

labdata = TVSLabDataset(dataset_path, reference_system="Stereophoto")
labdata

# %%
# Selecting data
# --------------
# The index of the dataset shows all available trials.
# This means each individual recording of a participant is represented by one row.
# By default, the dataset will also show trials that might not have valid reference data.
# If you want to skip them you can use the `missing_reference_error_type` parameter.
#
# If you compare the number of rows you can see that this removes a couple 100 trials.
# If you are planning to use the reference data, you should always set this parameter to "skip".
labdata = TVSLabDataset(
    dataset_path,
    reference_system="Stereophoto",
    missing_reference_error_type="skip",
)
labdata

# %%
# On the remaining data, we can easily filter by all columns that are in the index.
# For example, we could filter for only Test 11 (simulated activities of daily living) and Test 5
# (straight walking comfortable).
test_subset = labdata.get_subset(test=["Test5", "Test11"])
test_subset

# %%
# Note, that we can either filter by test or by test-name (or `test_name_pretty`).
# For the Free-Living dataset, the columns are named `recording` and `recording_name`.
test_subset = labdata.get_subset(
    test_name=["WalkComfortable", "DailyActivities"]
)
test_subset

# %%
# We could then further filter by cohort.
# For example only getting the data from Parkinson's patients.
# Note, this could also be done in a single call to `get_subset`.
test_subset_pd = test_subset.get_subset(cohort="PD")
test_subset_pd

# %%
# We can also filter based on information that is not directly in the index.
# For example, we could further filter for only the participants that are taller than 1.7m.
#
# For this, we access the ``participant_information`` (or ``participant_metadata``, more on the difference below)
# attribute of the dataset.
test_subset_pd_p_info = test_subset_pd.participant_information
test_subset_pd_p_info

# %%
# Note, that the participant information only contains the information of the participants that are in the current
# subset.
# So we can easily get the list of participants that are taller than 1.7m.
tall_participants = (
    test_subset_pd_p_info[test_subset_pd_p_info["height_m"] > 1.7]
    .reset_index()["participant_id"]
    .to_list()
)
tall_participants

# %%
# We can then use this list to get a subset of the data that only contains the data of the tall participants.
#
# Similar, we could filter based on any other information that is in the participant information (like disease status).
test_subset_tall_pd = test_subset_pd.get_subset(
    participant_id=tall_participants
)
test_subset_tall_pd

# %%
# We can also make more complex manipulations of the data, by directly manipulating the index.
# Let's say for all remaining test data, we want the last trial of each test.
# Note, that some tests for some participants have one, and others have two trials.
# However, as the index is sorted, we know that the last trial is always the last row of each test.
#
# We can use normal pandas operations on the dataset index and then provide the resulting index to get subset.
test_subset_tall_pd_last_trial = test_subset_tall_pd.get_subset(
    index=(
        test_subset_tall_pd.index.groupby(
            ["test", "participant_id", "time_measure"]
        ).tail(1)
    )
)
test_subset_tall_pd_last_trial

# %%
# Metadata
# --------
# Let's assume we now have the data, we want to work with, let's have a look at the actual data, we want to work with.
# We will rename the variable of our data subset to avoid typing the long name.
subset = test_subset_tall_pd_last_trial

# %%
# Then we are going to access the available meta information.
# The main demographic information is stored in the `participant_information` attribute (that we already saw above).
subset.participant_information

# %%
# It contains basic information about the participants, general clinical scores, disease specific clinical scores, and
# information about potential use of walking aids.
#
# In case, the entire list is too much, we can also access the respective subsets.
subset.demographic_information

# %%
subset.general_clinical_information

# %%
subset.cohort_clinical_information

# %%
subset.walking_aid_use_information

# %%
# When ever a data value was not recorded or not applicable for a participant, the value is set to NaN.
#
# All the information provided above is extracted from the ``participant_information.xlsx`` file provided with the
# dataset.
# A small subset of the information is also provided again within the ``infoForAlgo.mat`` files.
# The information there is the information deemed directly necessary for some of the algorithms.
#
# When using the standard pipelines, the information provided in the ``infoForAlgo.mat`` files is directly forwarded to
# the action method (e.g. ``detect`` or ``calculate``) of the algorithms as keyword arguments.
# In the dataset the information can be accessed via the ``participant_metadata_as_df`` attribute for multiple
# participants or the ``participant_metadata`` attribute for a single participant.
subset.participant_metadata_as_df

# %%
# Or for a single participant (just selecting the first row)
subset[0].participant_metadata

# %%
# Similarly, we can access recording metadata.
#
# Note, that the recording metadata is stored in the actual data.mat file.
# These files can be relatively large and hence, accessing the metadata (in particular for multiple participants) can
# be slow.
# To speed things up, you can use the caching mechanism provided by the dataset via the ``memory`` parameter.
# subset.recording_metadata_as_df

# %%
# Or for a single trial (just selecting the first row)
subset[0].recording_metadata

# %%
# The final piece of meta information that is available is the data quality of the SU (wearable sensor) and the
# reference data.
# This is a simple quality score (0-3) + additional comments that is provided for each recording.
#
# The numbers can be interpreted as follows:
#
# - 0: Recording discarded completely (these recordings are likely not included in the dataset in the first place)
# - 1: Recording has issues, but included in the dataset. Individual tests or trials might be missing, or might have
#   degraded quality.
# - 2: Recording had some issues, but they could be fixed. Actual data should be good (INDIP only)
# - 3: Recording is good
#
# Depending on your requirements, it might be necessary to filter out data with a quality score of 1.
# The comments can provide further inside into the issues.
subset.data_quality

# %%
# IMU and Reference Data
# ----------------------
# As explained in the other data loader examples, data can be accessed, once only a single trial is selected.
single_trial = subset[0]
single_trial

# %%
# The imu data is stored in the ``data_ss`` attribute.
# This is the data of the single sensor that was selected during the dataset creation.
single_trial.data_ss

# %%
# The reference data is stored in the ``reference_parameters_`` / ``reference_parameters_relative_to_wb_`` attribute.
single_trial.reference_parameters_

# %%
# Usage in Algorithms and Pipelines
# ---------------------------------
# The TVS datasets follow the exact same API as the other datasets in mobgap and hence, can be used in the same way.
# Everything that you can do with the :class:`~mobgap.data.LabExampleDataset` can also be done with the TVS datasets.
# This means, in all examples and tutorials that use the :class:`~mobgap.data.LabExampleDataset`, you can simply
# replace it with the TVS datasets.
# For more information check out the other data and algorithm examples.
