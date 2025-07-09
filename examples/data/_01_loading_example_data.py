"""

.. _data_loading_example:

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
# We provide a :class:`~mobgap.data.LabExampleDataset` class to load the example data.
# This is the easiest way to access the example data and allows you to select and iterate over the data in an easy way.
from mobgap.data import LabExampleDataset

example_data = LabExampleDataset()
# %%
# You can select the data you want using the ``get_subset`` method.
ha_example_data = example_data.get_subset(cohort="HA")
ha_example_data

# %%
# Once you selected only a single row of the dataset (either by repeated ``get_subset`` or by iteration), you can load
# the actual data.
single_test = ha_example_data.get_subset(
    participant_id="002", test="Test5", trial="Trial2"
)
single_test

# %%
# The raw IMU data can be accessed in two ways:
#
# 1. ``.data`` which contains a dictionary with the data of all IMU sensors in the dataset.
#    Per default, we only load the data of the "LowerBack" sensor for performance reasons.
#    But, you can select the sensors to load using the ``raw_data_sensor`` and ``sensor_psotions`` argument of the
#    Dataset class.
imu_data = single_test.data["LowerBack"]
imu_data
# %%
# 2. ``.data_ss`` which contains only the data of the "single sensor".
#    This is the data used as input for all algorithms in the provided pipelines.
#    In most cases this is equivalent to the data of the "LowerBack" sensor, but a different position can be selected
#    using the ``single_sensor_position`` (or ``single_sensor_name`` in some Dataset classes) argument of the Dataset
#    class.
single_sensor_data = single_test.data_ss
single_sensor_data

# %%
import matplotlib.pyplot as plt

single_sensor_data.filter(like="gyr").plot()
plt.show()

# %%
# Test-level metadata:
single_test.recording_metadata

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
single_trial_with_reference.raw_reference_parameters_

# %%
# The data you can see above is the "raw" reference data.
# Including both the information for walking bouts and level-walking bouts.
# To access the data in format that can be directly compared to the output of the mobgap algorithms or used as input
# to algorithms further down the processing pipeline, you can use the ``reference_parameters_`` attribute.
# If the data is extracted from the normal walking bouts or the level walking bouts is controlled by the
# ``reference_para_level`` parameter of the Dataset class (default is ``wb``).
ref_paras = single_trial_with_reference.reference_parameters_

# %%
# This attribute contains the data for the outputs of the various steps of the processing pipeline.
ref_paras.wb_list

# %%
ref_paras.ic_list

# %%
ref_paras.turn_parameters

# %%
ref_paras.stride_parameters


# %%
# Functional interface
# ++++++++++++++++++++
# We can get the local path to the example data using :func:`~mobgap.data.get_all_lab_example_data_paths`
# and then use :func:`~mobgap.data.load_mobilised_matlab_format` to load the data.
from mobgap.data import (
    get_all_lab_example_data_paths,
    load_mobilised_matlab_format,
)

all_example_data_paths = get_all_lab_example_data_paths()
list(all_example_data_paths.keys())

# %%
# Then we can select the participant we want to load.
example_participant_path = all_example_data_paths[("HA", "002")]
data = load_mobilised_matlab_format(example_participant_path / "data.mat")

# %%
# Calling the loader function without any further arguments, will load the "SU" (normal lower-back sensor) only.
# The returned dictionary contains the test names as keys and the loaded data as
# :class:`~mobgap.data.MobilisedTestData` objects.
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

data_with_reference = load_mobilised_matlab_format(
    example_participant_path / "data.mat", reference_system="INDIP"
)
selected_test = data_with_reference[test_list[2]]

# %%
# The returned :class:`~mobgap.data.MobilisedTestData` objects now contain the reference parameters.
raw_reference_data = selected_test.raw_reference_parameters

# %%
# And metadata about the reference system is available as well.
ref_sampling_rate_hz = selected_test.metadata["reference_sampling_rate_hz"]
ref_sampling_rate_hz

# %%
# To parse the reference data into better data structures, we can use the
# :func:`~mobgap.data.parse_reference_parameters` function.
from mobgap.data import parse_reference_parameters

data_sampling_rate_hz = selected_test.metadata["sampling_rate_hz"]

ref_paras_functional = parse_reference_parameters(
    raw_reference_data["wb"],
    data_sampling_rate_hz=data_sampling_rate_hz,
    ref_sampling_rate_hz=ref_sampling_rate_hz,
    debug_info="Example Recording",
)

# %%
# They have the same structure the reference parameters of the Dataset class.
ref_paras_functional
