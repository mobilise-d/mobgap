"""
Custom Data and Datasets
========================

While it is nice to have the example data and access to the TVS and some other datasets right through mobgap, let's be
honest: you will probably want to use your own data at some point.
For this we need to understand how to approach this.

First, it is important to understand that the API in Mobgap is split in two parts: Algorithms and Pipelines.
Algorithms by design only expect very simple data structures as input and only require inputs that they actually need.
Most of the time, this is a pandas DataFrame for the data, key-value pares for sampling rate and other metadata,
and other data frame like structures for intermediate results (e.g. list of initial contacts) required as input for
algorithms further down the pipeline.

This makes algorithms extremely easy to use even outside of any common pipeline.
However, because of this, algorithms are hard to wrap in higher level functions (like GridSearch) as their individual
APIs and data requirements are vary a lot.
So we need a second structure, that trades the simplicity of the inputs for a uniform call signature, that requires
a more complex data structure as input.

With that in mind, let's look at how to prepare your own data for Mobgap algorithms first, and then learn how to
build datasets that we can use in the pipelines.
"""

from typing import Optional, Union

# %%
# Step 1: Understanding the data we have
# ---------------------------------------
# As part of the mobgap package, we ship a few example datasets in "csv" format.
# They should serve as examples for "any common" data you might have.
#
# The folders have the following structure:
# (Note this might trigger an automatic download, when you run this cell)
from mobgap.data import get_example_csv_data_path

path = get_example_csv_data_path()
all_data_files = list(path.rglob("*.csv"))
all_data_files

# %%
# So we have folders that describe the cohort and participant id, and filenames that encode the time measure, test and
# trial.
# Each file contains the imu data in a simple csv format.
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
# Ok, in mobgap, we expect the raw data to be a pandas Dataframe.
# This is already taken care of.
# However, when inspecting the data more closely you will realize that the accelertion data is in units of g.
# In mobgap, we expect the acceleration data to be in m/s^2.
#
# This is good time to create a function that loads and converts the data.
# If you have more complex transforms to do for your data, this is the place to do it.
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
# Now that we have the data in the right format, we can use it with any algorithm that just requires IMU data as input.
# Let's use the GSD-Iluz algorithm as an example.
from mobgap.gsd import GsdIluz

gsd = GsdIluz()
gsd.detect(data=data, sampling_rate_hz=sampling_rate_hz)
gsd.gs_list_

# %%
# That's it! You can now use your own data with any algorithm in mobgap.
#
# If you have had reference data (e.g. reference initial contacts, or reference walking bouts), you could have a look
# at how we expect this data to be structured in the `common data structures <../common_data_types.html>`_ guide.
# In general, the structure of the reference data is always expected to be identical to the sturcture of the algorithm
# results, it should be compared to.
#
# Step 4: Building a custom dataset
# ---------------------------------
# Datasets are more complicated structures that encapsulate the meta information of an entire dataset (so potentially
# multiple participants, multiple cohorts, etc.) and provide a uniform API to access the data.
# The first dataset you might have seen in the context of mobgap is the :class:`~mobgap.data.LabExampleDataset`.
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
# For all mobgap pipelines we expect datapoint (a dataset with a single row) as input that are subclasses of
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
# For this we first preload all the data and metadata and then pass it to the class.
loaded_data = {}
for d in all_data_files:
    recording_identifier = d.name.split(".")[0].split("_")
    cohort, participant_id = d.parts[-3:-1]
    loaded_data[(cohort, participant_id, *recording_identifier)] = load_data(d)

# %%
from mobgap.data import GaitDatasetFromData

dataset_from_data = GaitDatasetFromData(loaded_data, sampling_rate_hz)
dataset_from_data

# %%
# We can make this a little easier to work with by providing better index column names.
dataset_from_data = GaitDatasetFromData(
    loaded_data,
    sampling_rate_hz,
    index_cols=["cohort", "participant_id", "time_measure", "test", "trial"],
)
dataset_from_data

# %%
# TODO: Show that the dataset can be used in pipeline.

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
# cvs example data, that we showed above.
# For a little bit slower, but more detailed guide, see the `tpcp rea-world-dataset guide
# <https://tpcp.readthedocs.io/en/latest/auto_examples/datasets/_02_datasets_real_world_example.html>`_.
#
# First thing that we need is an index of all files that exist in the dataset.
# We reuse the logic from above to extract the information from the path and the filename.
# This index creation happens in the ``create_index`` method in our custom class that subclasses
# :class:`~mobgap.data.BaseGaitDataset`.
#
# We take the base-path to our dataset as parameter in the init.
# And already implement the ``_path_from_index`` method that helps us to identify the correct file for a given index.
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
        all_data_files = list(self.base_path.rglob("*.csv"))
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
#     measurement_condition: Union[Literal["laboratory", "free_living"], str]
#
# We ignore the metadata stuff for now, and just implement the data loading and everything that can be represented as
# a constant.
class CsvExampleData(BaseGaitDataset):
    # Our constant values:
    sampling_rate_hz: float = 100
    measurement_condition = "laboratory"

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
        all_data_files = list(self.base_path.rglob("*.csv"))
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
# Now we can use this dataset with any pipeline that does not require metadata.
# TODO: Show example
