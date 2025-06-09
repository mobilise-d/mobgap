"""
.. _custom_datasets:

Custom Data and Datasets
========================

While it is nice to have the example data and access to the TVS and some other datasets right through mobgap, let's be
honest: you will probably want to use your own data at some point.
Let's discuss how to approach this.

First, it is important to understand that the API in Mobgap is split in two parts: Algorithms and Pipelines.
Algorithms by design only expect very simple data structures as input and only require inputs that they actually need.
Most of the time, this is a pandas DataFrame for the data, key-value pairs for sampling rate and other metadata,
and other dataframe-like structures for intermediate results (e.g. list of initial contacts) required as input for
algorithms further down the pipeline.

This makes algorithms extremely easy to use even outside of any common pipeline.
However, because of this, algorithms are hard to wrap in higher level functions (like GridSearch) as their individual
APIs and data requirements vary a lot.
So we need a second structure, that trades the simplicity of the inputs for a uniform call signature, that requires
a more complex data structure as input.

With that in mind, let's look at how to prepare your own data for Mobgap algorithms first, and then learn how to
build datasets that we can use in the pipelines.
"""

from typing import Optional, Union

# %%
# Step 1: Understanding the data we have
# --------------------------------------
# As part of the mobgap package, we ship a few example datasets in "csv" format.
# They should serve as examples for "any common" data you might have.
#
# The folders have the following structure:
# (Note that running this cell might trigger an automatic download)
from mobgap.data import get_example_csv_data_path

path = get_example_csv_data_path()
all_data_files = sorted(list(path.rglob("*.csv")))
all_data_files

# %%
# So we have folders that describe the cohort and participant id, and filenames that encode the time measure, test and
# trial.
# Each file contains the IMU data in a simple csv format.
# Let's load one of the files to see what it looks like.
import pandas as pd

data = pd.read_csv(all_data_files[0])
data.head()

# %%
# Normally, we would likely have additional files and documentation that describe the data.
# For this data, we simply know that the sampling rate is 100 Hz.
sampling_rate_hz = 100

# %%
# Step 2: Understanding the mobgap requirements
# ---------------------------------------------
# This is the point where you should read and understand the guides on
# `common data structures <../common_data_types.html>`_
# and the `expected coordinate systems <../coordinate_systems.html>`_ in mobgap.
#
# Come back, when you have done that!
#
# In mobgap, we expect the raw data to be a pandas Dataframe.
# This is already taken care of.
# However, when inspecting the data more closely you will realize that the acceleration data is in units of g.
# In mobgap, we expect the acceleration data to be in m/s^2.
#
# This is good time to create a function that loads and converts the data.
# If you have more complex transformations to apply for your data, this is the place to do it.
from pathlib import Path

from mobgap.consts import GRAV_MS2


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data[["acc_x", "acc_y", "acc_z"]] = (
        data[["acc_x", "acc_y", "acc_z"]] * GRAV_MS2
    )
    return data


data = load_data(all_data_files[0])
data.head()

# %%
# Step 3: Using the data with an algorithm
# ----------------------------------------
# Now that we have the data in the right format, we can use it with any algorithm that requires nothing else than
# IMU data as input.
# Let's use the GSD-Iluz algorithm as an example.
from mobgap.gait_sequences import GsdIluz
from mobgap.utils.conversions import to_body_frame

gsd = GsdIluz()
gsd.detect(data=to_body_frame(data), sampling_rate_hz=sampling_rate_hz)
gsd.gs_list_

# %%
# That's it! You can now use your own data with any algorithm in mobgap.
#
# If you have reference data (e.g. reference initial contacts, or reference walking bouts), you can also have a look
# at how we expect this data to be structured in the `common data structures <../common_data_types.html>`_ guide.
# In general, the structure of the reference data is always expected to be identical to the structure of the algorithm
# results that it should be compared to.
#
# Step 4: Metadata
# ----------------
# In addition to the raw IMU data, some algorithms (e.g. the :class:`~mobgap.stride_length.SlZijlstra`) require
# additional information about the participant.
# This additional information we refer to as "Participant Metadata".
# Each algorithm directly specifies what information it requires as keyword argument to it's "run"/"detect"/
# "calculate"/... method.
# Depending on what algorithms you want to use, you need this information available as well.
#
# In our example dataset, this information is stored in a "global" json file for all participants.
# Let's have a look at this.
#
# We load the file as json and collapse the "identifier levels" (in this case: cohort and participant_id)
# into a tuple as dict key and add the "cohort" as additional piece of metadata in the dict directly.
# We will see later, why this is a helpful format.
import json
from pprint import pprint


def load_particpant_metadata(path: Path):
    with path.open("r") as f:
        metadata = json.load(f)
    metadata_reformatted = {}
    for cohort_name, info in metadata.items():
        for participant_id, participant_metadata in info.items():
            metadata_reformatted[(cohort_name, participant_id)] = (
                participant_metadata
            )
            metadata_reformatted[(cohort_name, participant_id)]["cohort"] = (
                cohort_name
            )
    return metadata_reformatted


particpant_metadata = load_particpant_metadata(
    path / "participant_metadata.json"
)
pprint(particpant_metadata[("HA", "001")])

# %%
# We can see that our example data has quite a lot of metadata.
# This is not always required.
# The algorithms currently implemented, only require the sensor height in m and the cohort the participant belongs to.
#
# So if you are working with a custom data, make sure that this information is available to use all algorithms without
# issues.
#
# Next to participant metadata we also have the concept of "recording metadata".
# This is required only by the :func:`~mobgap.aggregation.apply_thresholds` function so far.
# It needs additional information about the recording environment,
# i.e., whether the recording was in a `free_living` or in a `laboratory` environment.
# Here, we only have laboratory data for all recordings.
# So we can define constant recording metadata for all recordings.
recording_metadata = {"measurement_condition": "laboratory"}

# %%
# Step 5: Building a custom dataset
# ---------------------------------
# Dataset classes are more complicated structures that encapsulate the meta information of an entire dataset
# (so potentially multiple participants, multiple cohorts, etc.) and provide a uniform API to access the data.
# The first dataset you might have seen in other mobgap examples of already
# is the :class:`~mobgap.data.LabExampleDataset`.
from mobgap.data import LabExampleDataset

example_data = LabExampleDataset()
example_data

# %%
# We can see that it contains the information about all the recordings in the example data.
# and we can access it, by iterating over it/taking an index slice.
single_trial = example_data[4]
single_trial

# %%
# We can access all the metadata and imu data from it.
single_trial.participant_metadata

# %%
single_trial.data_ss.head()

# %%
# For more information see the `dedicated example <data_loading_example>`_ on this.
#
# Now let's build a similar dataset for our own data.
#
# .. note:: Before reading this section, it would help to skim the
#    `tpcp dataset guide <https://tpcp.readthedocs.io/en/latest/auto_examples/datasets/_01_datasets_basics.html>`_.
#    We will not explain all the cool things you can do with datasets here, to not duplicate this information.
#
# For all mobgap pipelines we expect datapoints (a dataset with a single row) as input that are subclasses of
# :class:`~mobgap.data.BaseGaitDataset` or :class:`~mobgap.data.BaseGaitDatasetWithReference`.
# These baseclasses basically just define what attributes and methods a dataset should have at least to be compatible
# with the mobgap pipelines.
#
# The normal way would be to create a dataset subclass and implement all the required methods.
# However, for simple datasets or just single recordings, this might be overkill.
#
# Step 6: Custom Dataset - the shortcut
# -------------------------------------
# When we have just a single (or a couple) recordings that can all be comfortably loaded at once, we can use the
# :class:`~mobgap.data.GaitDatasetFromData` class to quickly create a valid dataset that can be used with any pipeline.
#
# For this we first preload all the data and identifier information and then pass it to the class.
# At the same time, we create a version of our metadata, that copies the participant metadata for each recording.
loaded_data = {}
participant_metadata_for_dataset_from_data = {}
recording_metadata_for_dataset_from_data = {}

for d in all_data_files:
    recording_identifier = d.name.split(".")[0].split("_")
    cohort, participant_id = d.parts[-3:-1]
    loaded_data[(cohort, participant_id, *recording_identifier)] = {
        "LowerBack": load_data(d)
    }
    participant_metadata_for_dataset_from_data[
        (cohort, participant_id, *recording_identifier)
    ] = particpant_metadata[(cohort, participant_id)]
    recording_metadata_for_dataset_from_data[
        (cohort, participant_id, *recording_identifier)
    ] = recording_metadata

# %%
from mobgap.data import GaitDatasetFromData

dataset_from_data = GaitDatasetFromData(
    loaded_data,
    sampling_rate_hz,
    participant_metadata_for_dataset_from_data,
    recording_metadata_for_dataset_from_data,
)
dataset_from_data

# %%
# We can make this a little easier to work with by providing better index column names.
dataset_from_data = GaitDatasetFromData(
    loaded_data,
    sampling_rate_hz,
    participant_metadata_for_dataset_from_data,
    recording_metadata_for_dataset_from_data,
    index_cols=["cohort", "participant_id", "time_measure", "test", "trial"],
)
dataset_from_data

# %%
# Now we can work with our custom dataset in the same way, as we worked with the example datasets.
#
# For example, we can get a subset
single_trial = dataset_from_data.get_subset(
    cohort="HA", participant_id="001", test="Test5"
)[0]
# %%
# And then use it to access the IMU data
single_trial.data_ss.head()

# And the participant metadata
single_trial.participant_metadata

# %%
# To show that this works as expected, we run one of the datapoints through the Mobilise-D Pipeline.
# (Note, we only expect a single WB within the "Test5" recordings)
from mobgap.pipeline import MobilisedPipelineHealthy

pipe = MobilisedPipelineHealthy().run(single_trial)
pipe.per_wb_parameters_.drop(columns="rule_obj").T


# %%
# Step 7: Custom Dataset - doing it properly
# ------------------------------------------
# If you have more complex data and want to do anything more than a one of analysis, it makes sense to create a proper
# dataset class that encapsulates all the logic on how to find and load the specific data format that you are working
# with.
# These datasets can either be very specific (like the :class:`~mobgap.data.TVSLabDataset`) or very generic, like the
# :class:`~mobgap.data.GenericMobilisedDataset`, that can be used with any folder structure full of `data.mat` files.
#
# In the following, we are going to speed-run through the creation of a simple dataset class that can be used with the
# csv example data, that we showed above.
# For a little bit slower, but more detailed guide, see the `tpcp real-world-dataset guide
# <https://tpcp.readthedocs.io/en/latest/auto_examples/datasets/_02_datasets_real_world_example.html>`_.
#
# First thing that we need is an index of all files that exist in the dataset.
# We reuse the logic from above to extract the information from the path and the filename.
# This index creation happens in the ``create_index`` method in our custom class that subclasses
# :class:`~mobgap.data.base.BaseGaitDataset`.
# Note, that we sort the files before creating the index!
# This is important to ensure that we get exactly the same index on every operating system.
#
# We take the base path to our dataset as parameter in the init.
# Furthermore, we already implement the ``_path_from_index`` method that helps us to identify
# the correct data file for a given index.
from mobgap.data.base import BaseGaitDataset


class CsvExampleData(BaseGaitDataset):
    def __init__(
        self,
        base_path: Path,
        *,
        groupby_cols: Optional[Union[list[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ) -> None:
        self.base_path = base_path
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def _path_from_index(self) -> Path:
        self.assert_is_single(None, "_path_from_index")
        g = self.group_label
        return (
            self.base_path
            / g.cohort
            / g.participant_id
            / f"{g.time_measure}_{g.test}_{g.trial}.csv"
        )

    def create_index(self) -> pd.DataFrame:
        all_data_files = sorted(list(self.base_path.rglob("*.csv")))
        index = []
        for d in all_data_files:
            recording_identifier = d.name.split(".")[0].split("_")
            cohort, participant_id = d.parts[-3:-1]
            index.append((cohort, participant_id, *recording_identifier))
        return pd.DataFrame(
            index,
            columns=[
                "cohort",
                "participant_id",
                "time_measure",
                "test",
                "trial",
            ],
        )


# %%
# With this we can already represent the metadata and iterate over it.
csv_data = CsvExampleData(path)
csv_data


# %%
# To make this actually useful we need to integrate the data loading logic.
# The base class expects us to implement the following attributes:
#
# .. code:: python
#
#     sampling_rate_hz: float
#     data_ss: pd.DataFrame
#     participant_metadata: ParticipantMetadata
#     recording_metadata: RecordingMetadata
#
# Sampling rate and recording metadata are trivial, as they are constant for all recordings.
# The participant metadata is a little bit more complex, as we need to load it from the json file, but we already
# have the loading logic, and just going to reuse that here.
# Same for the data loading, we already have the logic to load the data, we just need to implement the data attribute.
from mobgap.data.base import ParticipantMetadata


class CsvExampleData(BaseGaitDataset):
    # Our constant values:
    sampling_rate_hz: float = 100
    measurement_condition = "laboratory"
    recording_metadata = {"measurement_condition": "laboratory"}

    def __init__(
        self,
        base_path: Path,
        *,
        groupby_cols: Optional[Union[list[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ) -> None:
        self.base_path = base_path
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def _path_from_index(self) -> Path:
        self.assert_is_single(None, "_path_from_index")
        g = self.group_label
        return (
            self.base_path
            / g.cohort
            / g.participant_id
            / f"{g.time_measure}_{g.test}_{g.trial}.csv"
        )

    def create_index(self) -> pd.DataFrame:
        all_data_files = sorted(list(self.base_path.rglob("*.csv")))
        index = []
        for d in all_data_files:
            recording_identifier = d.name.split(".")[0].split("_")
            cohort, participant_id = d.parts[-3:-1]
            index.append((cohort, participant_id, *recording_identifier))
        return pd.DataFrame(
            index,
            columns=[
                "cohort",
                "participant_id",
                "time_measure",
                "test",
                "trial",
            ],
        )

    @property
    def participant_metadata(self) -> ParticipantMetadata:
        self.assert_is_single(None, "participant_metadata")
        return particpant_metadata[
            (self.group_label.cohort, self.group_label.participant_id)
        ]

    # data loading:
    @property
    def data(self) -> dict[str, pd.DataFrame]:
        # Data loading is only allowed, once we have just a single recording selected.
        self.assert_is_single(None, "data")
        return {"LowerBack": load_data(self._path_from_index())}

    @property
    def data_ss(self) -> pd.DataFrame:
        self.assert_is_single(None, "data_ss")
        return self.data["LowerBack"]


# %%
# Now we can use this dataset with any pipeline as before!
csv_data = CsvExampleData(path)
csv_data

# %%
single_trial = csv_data.get_subset(
    cohort="HA", participant_id="001", test="Test5"
)[0]
single_trial

# %%
pipe = MobilisedPipelineHealthy().run(single_trial)
pipe.per_wb_parameters_.drop(columns="rule_obj").T

# %%
# Next Steps
# ----------
# There are several ways on how to improve this dataset further.
# You could look into performance improvements like "caching" to avoid reloading the files from disk too often.
# See `this guide <https://tpcp.readthedocs.io/en/latest/auto_examples/recipies/_01_caching.html>`_ for more
# information.
#
# Further, in case you have a dataset with reference data, you could change the base class of the dataset to
# :class:`~mobgap.data.base.BaseGaitDatasetWithReference` and implement the reference data loading attributes.
# This allows to use the dataset for DMO validation or optimization pipelines.
# See for example `lrc_evaluation`_ for more information.
#
# If you are interested into more examples in how datasets can be structure in general, have a look at the source of
# :class:`~mobgap.data.TVSLabDataset` or :class:`~mobgap.data.GenericMobilisedDataset`.
# For more examples outside mobgap, have a look at the source of the
# `gaitmap-dataset <https://github.com/mad-lab-fau/gaitmap-datasets>`_ package.
