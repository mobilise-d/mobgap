"""
Loading example data
====================

This example shows how to use the loader functions to load example data.

Laboratory data
---------------
We provide a small subset of the TVS dataset of the Mobilise-D project as example data.
The data only contains a small number of participants and only two (instead of the ten) laboratory tests of each
participant.
The data is stored in the `examples/data` folder of the repository in the Mobilise-D matlab format [1]_.

.. [1] Palmerini L, Reggi L, Bonci T, Del Din S, Micó-Amigo ME, Salis F, Bertuletti S, Caruso M, Cereatti A, Gazit E,
       Paraschiv-Ionescu A, Soltani A, Kluge F, Küderle A, Ullrich M, Kirk C, Hiden H, D'Ascanio I, Hansen C,
       Rochester L, Mazzà C, Chiari L. Mobility recorded by wearable devices and gold standards: the Mobilise-D
       procedure for data standardization. Sci Data. 2023 Jan 19;10(1):38. doi: 10.1038/s41597-023-01930-9.
       PMID: 36658136; PMCID: PMC9852581.
"""
# %%
# Dataset Class
# +++++++++++++
# We provide a :class:`~gaitlink.data.LabExampleDataset` class to load the example data.
# This is the easiest way to access the example data and allows you to select and iterate over the data in an easy way.
from gaitlink.data import LabExampleDataset

example_data = LabExampleDataset()
# %%
# You can select the data you want using the ``get_subset`` method.
ha_example_data = example_data.get_subset(cohort="HA")
ha_example_data

# %%
# Once you selected only a single row of the dataset (either by repeated ``get_subset`` or by iteration), you can load
# the actual data.
single_test = ha_example_data.get_subset(participant_id="002", test="Test5", trial="Trial2")
single_test

# %%
# The raw IMU data:
imu_data = single_test.data["LowerBack"]
imu_data

# %%
import matplotlib.pyplot as plt

imu_data.filter(like="gyr").plot()
plt.show()

# %%
# Test-level metadata:
single_test.metadata

# %%
# Participant-level metadata:
single_test.participant_metadata

# %%
# You can also load the reference system data, by specifying the ``reference_system`` argument.
# All parameters related to the reference systems have a trailing underscore.
example_data_with_reference = LabExampleDataset(reference_system="Stereophoto")
single_trial_with_reference = example_data_with_reference.get_subset(
    cohort="HA", participant_id="002", test="Test5", trial="Trial2"
)
single_trial_with_reference.reference_parameters_


# %%
# Functional interface
# ++++++++++++++++++++
# We can get the local path to the example data using :func:`~gaitlink.data.get_all_lab_example_data_paths`
# and then use :func:`~gaitlink.data.load_mobilised_matlab_format` to load the data.
from gaitlink.data import load_mobilised_matlab_format, get_all_lab_example_data_paths

all_example_data_paths = get_all_lab_example_data_paths()
list(all_example_data_paths.keys())

# %%
# Then we can select the participant we want to load.
example_participant_path = all_example_data_paths[("HA", "002")]
data = load_mobilised_matlab_format(example_participant_path / "data.mat")

# %%
# Calling the loader function without any further arguments, will load the "SU" (normal lower-back sensor) only.
# The returned dictionary contains the test names as keys and the loaded data as
# :class:`~gaitlink.data.MobilisedTestData` objects.
# This allows for easy access to the data and metadata without traversing a nested data structure.
test_list = list(data.keys())
test_list

# %%
# We can access the data of a single test by using the test name as key.
test_11_data = data[test_list[2]]
imu_data = test_11_data.imu_data["LowerBack"]
imu_data

# %%
# We can also access the metadata of the test.
test_11_data.metadata

# %%
# To load reference data as well, we can use the ``reference_system`` argument.
# Note, that we don't have a way to load the raw data of the reference system.
# We only load the calculated parameters.
#
# The available reference systems will depend on the data.
data_with_reference = load_mobilised_matlab_format(example_participant_path / "data.mat", reference_system="INDIP")

# %%
# The returned :class:`~gaitlink.data.MobilisedTestData` objects now contain the reference parameters.
data_with_reference[test_list[2]].reference_parameters

# %%
# And metadata about the reference system is available as well.
data_with_reference[test_list[0]].metadata.reference_sampling_rate_hz
